import os
import shutil
from pathlib import Path

def verify_model_files():
    print("Verifying model files setup...")
    
    # Define required files
    required_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors"
    ]
    
    # Set up model cache directory
    cache_dir = Path(__file__).parent / "model_cache"
    cache_dir.mkdir(exist_ok=True)
    
    # Check if files exist in the current directory
    missing_files = []
    for file in required_files:
        if not (Path(__file__).parent / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("\nMissing required files. Please download these files from https://huggingface.co/mistralai/Mistral-7B-v0.1:")
        for file in missing_files:
            print(f"- {file}")
        print("\nSteps to download:")
        print("1. Go to https://huggingface.co/mistralai/Mistral-7B-v0.1")
        print("2. Click 'Files and versions' tab")
        print("3. Download each missing file")
        print("4. Place the files in the same directory as this script")
        return False
    
    # Move files to cache directory
    print("\nMoving files to cache directory...")
    for file in required_files:
        src = Path(__file__).parent / file
        dst = cache_dir / file
        shutil.copy2(src, dst)
        print(f"Moved {file} to cache")
    
    print(f"\nAll files verified and moved to: {cache_dir}")
    print("\nCache directory contents:")
    for item in os.listdir(cache_dir):
        print(f"- {item}")
    
    return True

if __name__ == "__main__":
    verify_model_files() 