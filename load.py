from huggingface_hub import snapshot_download
import os

def download_qwen_tts(model_id, local_dir):
    print(f"Starting download for {model_id}...")
    
    # This downloads the full repo including weights, configs, and tokenizers
    try:
        path = snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # Set to True if using a shared cache
            revision="main"
        )
        print(f"\nSuccess! Model weights are located at: {path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    SAVE_PATH = "./model"
    
    download_qwen_tts(MODEL_ID, SAVE_PATH)
