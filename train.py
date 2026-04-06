import argparse
import json
import os
import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

# Internal Qwen modules
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from dataset import TTSDataset

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="model/")
    parser.add_argument("--train_jsonl", type=str, default="shuffled_multilingual_entries.jsonl")
    parser.add_argument("--output_dir", type=str, default="output/")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-6) 
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    args = parser.parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=8, mixed_precision="bf16")

    # 1. Load Model, Config, and Processor
    qwen3tts = Qwen3TTSModel.from_pretrained(
        args.init_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    config = qwen3tts.model.config

    next_lang_id = max(config.talker_config.codec_language_id.values()) + 1
    config.talker_config.codec_language_id["uzbek"] = next_lang_id
    print(f"Registered 'uzbek' with ID: {next_lang_id}")

    # --- APPLY LORA TO THE TALKER ---
    lora_config = LoraConfig(
        r=32,         
        lora_alpha=32,             
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj" 
        ],
        lora_dropout=0.1,       
        bias="none",
        use_rslora=True,           
        use_dora=True,        
        modules_to_save=["language_embedding"] 
    )
    
    # Wrap the talker model
    qwen3tts.model.talker = get_peft_model(qwen3tts.model.talker, lora_config)
    qwen3tts.model.talker.print_trainable_parameters()

    # qwen3tts.model.talker.base_model.model.model.text_embedding.weight.requires_grad = True

    # Freeze the speaker encoder 
    for param in qwen3tts.model.speaker_encoder.parameters():
        param.requires_grad = False

    # 2. Data Loading
    with open(args.train_jsonl, 'r', encoding='utf-8') as f:
        train_data = [json.loads(line) for line in f]
    
    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=dataset.collate_fn
    )

    # 3. Optimizer & Scheduler Setup
    optimizer = AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=0.1)

    num_update_steps_per_epoch = len(train_dataloader) // accelerator.gradient_accumulation_steps
    max_train_steps = args.epochs * num_update_steps_per_epoch
    num_warmup_steps = int(max_train_steps * args.warmup_ratio)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader, lr_scheduler
    )

    model.eval()
    last_grad_norm = 0.0

    talker_model = model.talker.base_model.model

    # 4. Training Loop
    model.train()
    last_grad_norm = 0.0
    
    for epoch in range(args.epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                input_ids = batch['input_ids']
                codec_ids = batch['codec_ids']
                ref_mels = batch['ref_mels']
                text_embedding_mask = batch['text_embedding_mask']
                codec_embedding_mask = batch['codec_embedding_mask']
                attention_mask = batch['attention_mask']
                codec_0_labels = batch['codec_0_labels']
                codec_mask = batch['codec_mask']

                speaker_embedding = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype)).detach()

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                input_text_embedding = model.talker.text_projection(model.talker.get_text_embeddings()(input_text_ids)) * batch['text_embedding_mask']
                input_codec_embedding = talker_model.model.codec_embedding(input_codec_ids) * batch['codec_embedding_mask']
                input_codec_embedding[:, 6, :] = speaker_embedding

                input_embeddings = input_text_embedding + input_codec_embedding

                for i in range(1, 16):
                    codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                outputs = model.talker(
                    inputs_embeds=input_embeddings,
                    attention_mask=attention_mask,
                    labels=codec_0_labels,
                    output_hidden_states=True
                )
                # Sub-talker loss: hidden_states shape [B, T, D]，codec_mask shape [B, T]
                hidden_states = outputs.hidden_states[0][-1]
                target_codec_mask = codec_mask[:, 1:]
                talker_hidden_states = hidden_states[:, :-1][target_codec_mask]
                talker_codec_ids = codec_ids[:, 1:][target_codec_mask]

                sub_talker_logits, sub_talker_loss = model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)
                main_loss= outputs.loss
                loss = main_loss+ 0.3 * sub_talker_loss

                accelerator.backward(loss)

                # Gradient Clipping (Must stay within sync_gradients check)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    last_grad_norm = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
    
            # Healthy Logging
            if step % 20 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                accelerator.print(
                    f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f} | "
                    f"Main: {main_loss.item():.2f} | Sub: {sub_talker_loss.item():.2f} | "
                    f"LR: {current_lr:.2e} | GradNorm: {last_grad_norm:.4f}"
                )

        os.makedirs(args.output_dir, exist_ok=True)
        unwrapped_model = accelerator.unwrap_model(model)
        
        # Save PEFT adapters (this is the most important part)
        unwrapped_model.talker.save_pretrained(args.output_dir)
        
        # Manual Config Save to avoid the KeyError: 'dtype'
        # Instead of save_pretrained, we save the raw dict which is safer
        config_path = os.path.join(args.output_dir, "config.json")
        try:
            # We use to_dict() to get all parameters without the 'diff' logic crashing
            config_dict = unwrapped_model.config.to_dict()
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Standard config save failed, trying fallback: {e}")
            # Fallback: just save the talker config if the main one is corrupted
            unwrapped_model.config.talker_config.save_pretrained(args.output_dir)
        
        print(f"LoRA weights and config successfully saved to {args.output_dir}")

if __name__ == "__main__":
    train()