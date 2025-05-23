from huggingface_hub import login, HfApi, snapshot_download
import os
from dotenv import load_dotenv
import subprocess
import sys
import json
import logging
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_model():
    logger.info("Starting model download process...")
    
    # Load environment variables
    load_dotenv()
    
    # Get Hugging Face token
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        logger.error("HUGGINGFACE_TOKEN not found in .env file")
        print("\nPlease follow these steps:")
        print("1. Go to https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        print("2. Click 'Access repository' and accept the terms")
        print("3. Go to https://huggingface.co/settings/tokens")
        print("4. Generate a new token")
        print("5. Create a .env file in this directory with:")
        print("   HUGGINGFACE_TOKEN=your_token_here")
        raise ValueError("Missing Hugging Face token")
    
    # Set up cache directory
    cache_dir = os.path.join(os.path.dirname(__file__), "model_cache")
    if os.path.exists(cache_dir):
        logger.info("Cleaning existing cache directory...")
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        # First, try to login using the CLI
        logger.info("Logging in to Hugging Face...")
        login_result = subprocess.run(
            ["huggingface-cli", "login", "--token", hf_token],
            capture_output=True,
            text=True
        )
        
        if login_result.returncode != 0:
            logger.warning("CLI login failed, trying Python API login...")
            try:
                login(token=hf_token)
            except Exception as e:
                logger.error(f"Python API login failed: {str(e)}")
                raise
        
        logger.info("Login successful!")
        
        # Initialize the API
        api = HfApi()
        
        # Verify access to the model
        logger.info("Verifying model access...")
        try:
            model_info = api.model_info("TinyLlama/TinyLlama-1.1B-Chat-v1.0", token=hf_token)
            # Convert model_info to a dictionary of basic types
            model_info_dict = {
                "id": model_info.id,
                "modelId": model_info.modelId,
                "tags": getattr(model_info, 'tags', []),
                "pipeline_tag": getattr(model_info, 'pipeline_tag', None),
                "private": getattr(model_info, 'private', False),
                "downloads": getattr(model_info, 'downloads', 0),
                "likes": getattr(model_info, 'likes', 0)
            }
            logger.info(f"Access verified! Model info: {json.dumps(model_info_dict, indent=2)}")
        except Exception as e:
            logger.error(f"Error accessing model: {str(e)}")
            print("\nPlease make sure you have:")
            print("1. Accepted the terms at https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            print("2. Used a token with read access")
            raise
        
        logger.info("Downloading model files...")
        # Download the complete model
        try:
            snapshot_download(
                repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                repo_type="model",
                cache_dir=cache_dir,
                token=hf_token,
                local_files_only=False,
                force_download=True
            )
        except Exception as e:
            logger.error(f"Error during model download: {str(e)}")
            raise
        
        # Find the snapshots directory
        snapshots_dir = None
        for root, dirs, files in os.walk(cache_dir):
            if "snapshots" in dirs:
                snapshots_dir = os.path.join(root, "snapshots")
                break
        
        if not snapshots_dir:
            raise ValueError("Could not find snapshots directory in cache")
        
        # Find the latest snapshot
        snapshot_dirs = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
        if not snapshot_dirs:
            raise ValueError("No snapshots found")
        
        latest_snapshot = os.path.join(snapshots_dir, snapshot_dirs[0])
        logger.info(f"Found snapshot directory: {latest_snapshot}")
        
        # Verify the files in the snapshot
        json_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
        logger.info("Verifying configuration files...")
        
        for file in json_files:
            file_path = os.path.join(latest_snapshot, file)
            if not os.path.exists(file_path):
                raise ValueError(f"Required file {file} not found in snapshot directory")
            
            # Verify file is not empty and contains valid JSON
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if not content.strip():
                        raise ValueError(f"File {file} is empty")
                    json.loads(content)  # Verify it's valid JSON
                    logger.info(f"Successfully verified {file}")
            except json.JSONDecodeError:
                raise ValueError(f"File {file} contains invalid JSON")
            except Exception as e:
                raise ValueError(f"Error reading {file}: {str(e)}")
        
        # Verify model files exist
        model_files = [f for f in os.listdir(latest_snapshot) if f.endswith('.safetensors') or f.endswith('.bin')]
        if not model_files:
            raise ValueError("No model weight files found in snapshot directory")
        logger.info(f"Found model weight files: {model_files}")
        
        logger.info(f"All files verified successfully in: {latest_snapshot}")
        
        # Show the contents of the snapshot directory
        logger.info("Snapshot contents:")
        for item in os.listdir(latest_snapshot):
            item_path = os.path.join(latest_snapshot, item)
            size = os.path.getsize(item_path)
            logger.info(f"- {item} ({size} bytes)")
            
    except Exception as e:
        logger.error(f"Error during model download: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you've accepted the terms at https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        print("2. Verify your token is correct")
        print("3. Check your internet connection")
        print("4. Try running: pip install --upgrade huggingface_hub")
        raise

if __name__ == "__main__":
    # First ensure huggingface_hub is up to date
    logger.info("Updating huggingface_hub...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "huggingface_hub"])
    
    download_model() 