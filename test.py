
import torch
import soundfile as sf
import time
import json
from peft import PeftModel
from qwen_tts import Qwen3TTSModel
import re

def normalize_uzbek_text(text):
    text = text.replace('‘', "'").replace('’', "'").replace('`', "'").replace('´', "'")
    # Clean up the specific mojibake if it's baked into your files
    text = text.replace('âĢĺ', "'")
    # Official Uzbek Unicode modifier (U+02BB) is often better, 
    # but check if your tokenizer prefers the standard straight ' (U+0027)
    return text

# Paths
# epi = epitran.Epitran('uzb-Latn')
base_model_path = "model/"  
adapter_path = "output/"  

# 1. Load the Base Model
print("Loading base model...")
model = Qwen3TTSModel.from_pretrained(
    base_model_path,
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)

# 2. Inject the LoRA Adapters into the Talker
print("Loading LoRA adapters...")
model.model.talker = PeftModel.from_pretrained(
    model.model.talker, 
    adapter_path
)

with open(f"{adapter_path}/config.json", "r") as f:
    updated_config = json.load(f)
    uzbek_id = updated_config.get("talker_config", {}).get("codec_language_id", {}).get("uzbek")

if uzbek_id is not None:
    model.model.config.talker_config.codec_language_id["uzbek"] = uzbek_id

    if "Uzbek" in model.model.config.talker_config.codec_language_id:
        del model.model.config.talker_config.codec_language_id["Uzbek"]
    
    # 3. Apply the Monkey Patch (Using the lowercase key)
    original_supported_langs_method = model._supported_languages_set

    def patched_supported_languages_set():
        # Handle cases where the original method might return a tuple or list
        langs = original_supported_langs_method()
        if langs is not None:
            if isinstance(langs, set):
                langs.add("uzbek")
            elif isinstance(langs, list):
                if "uzbek" not in langs:
                    langs.append("uzbek")
        return langs

    model._supported_languages_set = patched_supported_languages_set
    print(f"Uzbek language registered and validator patched with ID: {uzbek_id}")
else:
    print("Warning: Could not find 'Uzbek' in the saved config.json!")

# Reference audio for cloning
ref_audio = "example_og.wav"
ref_text  = "o'zbekiston respublikasi prezidentining ikkiming onyttinchi-yil onikkinchi-apreldagi pq-ikkiming sakkizyuz sakson ikkinchi-sonli qaroriga birinchi-ilova"

inp_text="O‘zbekiston Respublikasi poytaxti Toshkent shahrida yozgi kunlar issiq va quyoshli bo‘ladi. Har bir odam o‘z ishlariga shoshiladi, ammo qishloqda hayot tinchroq kechadi. Qaldirgochlar chirillab, bolalar esa kitob o‘qishni yoqtiradilar. Yomg‘ir yog‘sa, ko‘chalarda hidlar yanada yoqimli bo‘ladi."
# inp_text="Hi my name is Muxlisa, how can I assist you today?"

# phonemes = epi.transliterate(inp_text)
phonemes =normalize_uzbek_text(inp_text)
print("Input of the model G2P model: ", phonemes)
tokens = model.processor.tokenizer.tokenize(phonemes)
print(tokens)

# Generate speech
print("Start of generation!")
st = time.time()

wavs, sr = model.generate_voice_clone(
    text=phonemes,
    language="english", 
    ref_audio=ref_audio,
    ref_text=ref_text,
    max_new_tokens=300,       # Limits to ~12 seconds
    repetition_penalty=1.2,   # Helps stop the "looping" behavior
    temperature=0.7,          # Lowering temp makes it more stable/less "creative"
    top_p=0.8,
)

# Save the resulting audio
sf.write("output_voice_clone_uzbek.wav", wavs[0], sr)
print(f"Finished in: {time.time() - st:.2f} seconds")
