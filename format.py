import json
import random
from pathlib import Path
import torch
import torchaudio
import soundfile as sf
from datasets import load_dataset, concatenate_datasets
from qwen_tts import Qwen3TTSTokenizer
from huggingface_hub import login

login('...')

# Settings
TOKENIZER_MODEL = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
OUTPUT_JSONL = "shuffled_multilingual_entries.jsonl"
TMP_WAV_DIR = Path("tmp_wavs")
TMP_WAV_DIR.mkdir(exist_ok=True)
BATCH_ENCODE_SIZE = 32
TARGET_SR = 24000

act_langs = ["russian", "english", "uzbek", "uzbek"]
langs = ["ru_ru", "en_us", "uz_uz", "uz", "ru"]
repos = ["google/fleurs", "google/fleurs", "google/fleurs", "fsicoli/common_voice_18_0"]

tokenizer = Qwen3TTSTokenizer.from_pretrained(TOKENIZER_MODEL)

# --- STEP 1: Collect all raw data into one list ---
raw_samples = []

for repo, lang, act_lang in zip(repos, langs, act_langs):
    print(f"Loading {repo} ({lang})...")
    try:
        ds = load_dataset(repo, lang, split="train", trust_remote_code=True)
        for row in ds:
            text = row.get("sentence") or row.get("text") or row.get("transcription")
            audio = row.get("audio")
            if text and audio:
                raw_samples.append({
                    "audio_data": audio, # Keep the raw dict for now
                    "text": text.strip(),
                    "act_lang": act_lang,
                    "subset": lang
                })
    except Exception as e:
        print(f"Skipping {lang}: {e}")

# --- STEP 2: Shuffle the entire collection ---
print(f"Total samples collected: {len(raw_samples)}. Shuffling...")
random.seed(42) # For reproducibility
random.shuffle(raw_samples)

# --- STEP 3: Process and Save ---
entries = []
batch_paths = []
batch_entries = []

for i, sample in enumerate(raw_samples):
    # 1. Process Audio
    wav_path = TMP_WAV_DIR / f"shuffled_{i}.wav"
    audio = sample["audio_data"]
    arr, sr = audio["array"], audio["sampling_rate"]
    
    wave = torch.from_numpy(arr).float().unsqueeze(0)
    if sr != TARGET_SR:
        wave = torchaudio.functional.resample(wave, orig_freq=sr, new_freq=TARGET_SR)
    
    sf.write(str(wav_path), wave.squeeze(0).numpy(), TARGET_SR, subtype="PCM_16")

    # 2. Prepare Entry
    entry = {
        "audio": str(wav_path),
        "text": sample["text"],
        "language": sample["act_lang"],
        "ref_audio": str(wav_path) # Self-reference
    }

    batch_paths.append(str(wav_path))
    batch_entries.append(entry)

    # 3. Batch Tokenization
    if len(batch_paths) >= BATCH_ENCODE_SIZE:
        enc = tokenizer.encode(batch_paths)
        for e, codes in zip(batch_entries, enc.audio_codes):
            e["audio_codes"] = codes.cpu().tolist()
            entries.append(e)
        batch_paths.clear()
        batch_entries.clear()

    if i % 100 == 0:
        print(f"Processed {i}/{len(raw_samples)} samples...")

# Final flush
if batch_paths:
    enc = tokenizer.encode(batch_paths)
    for e, codes in zip(batch_entries, enc.audio_codes):
        e["audio_codes"] = codes.cpu().tolist()
        entries.append(e)

# Save to JSONL
print(f"Saving to {OUTPUT_JSONL}...")
with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    for e in entries:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")

print("Done!")