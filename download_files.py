import os
import subprocess
from pathlib import Path
from dotenv import load_dotenv

def download_files():
    print("Starting file download process...")
    
    # Load environment variables
    load_dotenv()
    
    # Get Hugging Face token
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("\nError: HUGGINGFACE_TOKEN not found in .env file")
        print("\nPlease follow these steps:")
        print("1. Go to https://huggingface.co/settings/tokens")
        print("2. Generate a new token")
        print("3. Create a .env file in this directory with:")
        print("   HUGGINGFACE_TOKEN=your_token_here")
        return False
    
    # Set up download directory
    download_dir = Path(__file__).parent
    os.makedirs(download_dir, exist_ok=True)
    
    # Files to download
    files_to_download = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors"
    ]
    
    print("\nDownloading files...")
    for file in files_to_download:
        print(f"\nDownloading {file}...")
        url = f"https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/{file}"
        output_path = download_dir / file
        
        # Use curl to download the file
        cmd = [
            "curl",
            "-L",
            "-H", f"Authorization: Bearer {hf_token}",
            url,
            "-o", str(output_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"Successfully downloaded {file}")
            else:
                print(f"Error downloading {file}: {result.stderr}")
                return False
        except Exception as e:
            print(f"Error downloading {file}: {str(e)}")
            return False
    
    print("\nAll files downloaded successfully!")
    print("\nVerifying files...")
    return verify_files()

def verify_files():
    required_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors"
    ]
    
    download_dir = Path(__file__).parent
    missing_files = []
    
    for file in required_files:
        if not (download_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("\nMissing files:")
        for file in missing_files:
            print(f"- {file}")
        return False
    
    print("\nAll files verified successfully!")
    return True

if __name__ == "__main__":
    download_files() 