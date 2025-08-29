#!/usr/bin/env python3
"""
Script to download the correct Unsloth base model for LoRA merging.
This ensures compatibility with the fine-tuned LoRA adapter.
"""

import os
import sys
import argparse
from pathlib import Path
import subprocess
import logging
from huggingface_hub import snapshot_download, login
import torch

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_requirements():
    """Check if required packages are installed."""
    required_packages = [
        'torch',
        'transformers',
        'huggingface_hub',
        'peft',
        'accelerate'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.info("Please install them using: pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_disk_space(download_path, required_gb=15):
    """Check if there's enough disk space for the model."""
    try:
        statvfs = os.statvfs(download_path)
        available_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
        
        if available_gb < required_gb:
            logger.error(f"Insufficient disk space. Required: {required_gb}GB, Available: {available_gb:.1f}GB")
            return False
        
        logger.info(f"Disk space check passed. Available: {available_gb:.1f}GB")
        return True
    
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")
        return True

def download_base_model(output_dir, use_auth_token=None):
    """Download the Unsloth Qwen2.5-VL base model."""
    model_id = "unsloth/qwen2.5-vl-7b-instruct-bnb-4bit"
    
    logger.info(f"Starting download of {model_id}")
    logger.info(f"Download location: {output_dir}")
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Check disk space
        if not check_disk_space(output_dir):
            return False
        
        # Login to Hugging Face if token provided
        if use_auth_token:
            logger.info("Logging in to Hugging Face...")
            login(token=use_auth_token)
        
        # Download the model
        logger.info("Downloading model files...")
        downloaded_path = snapshot_download(
            repo_id=model_id,
            local_dir=output_dir,
            local_dir_use_symlinks=False,  # Use actual files, not symlinks
            resume_download=True,  # Resume if interrupted
            token=use_auth_token
        )
        
        logger.info(f"Model downloaded successfully to: {downloaded_path}")
        
        # Verify download
        essential_files = [
            "config.json",
            "tokenizer_config.json",
            "tokenizer.json"
        ]
        
        missing_files = []
        for file in essential_files:
            if not os.path.exists(os.path.join(output_dir, file)):
                missing_files.append(file)
        
        if missing_files:
            logger.error(f"Download incomplete. Missing files: {', '.join(missing_files)}")
            return False
        
        logger.info("Download verification passed!")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download Unsloth Qwen2.5-VL base model for LoRA merging")
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./base_model",
        help="Directory to download the model to (default: ./base_model)"
    )
    parser.add_argument(
        "--auth-token",
        type=str,
        help="Hugging Face authentication token (optional)"
    )
    parser.add_argument(
        "--skip-requirements-check",
        action="store_true",
        help="Skip checking for required packages"
    )
    
    args = parser.parse_args()
    
    # Convert to absolute path
    output_dir = os.path.abspath(args.output_dir)
    
    logger.info("=" * 60)
    logger.info("Unsloth Qwen2.5-VL Base Model Download Script")
    logger.info("=" * 60)
    
    # Check requirements
    if not args.skip_requirements_check:
        logger.info("Checking requirements...")
        if not check_requirements():
            sys.exit(1)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        logger.info(f"CUDA available with {torch.cuda.device_count()} GPU(s)")
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
    else:
        logger.warning("CUDA not available. Model will run on CPU (much slower)")
    
    # Download model
    success = download_base_model(output_dir, args.auth_token)
    
    if success:
        logger.info("=" * 60)
        logger.info("✅ BASE MODEL DOWNLOAD COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"Model location: {output_dir}")
        logger.info("Next steps:")
        logger.info("1. Run 02_merge_lora.py to merge with your LoRA adapter")
        logger.info("2. Run 03_convert_to_gguf.py to convert for Ollama")
        logger.info("3. Use 04_create_modelfile.py to create Ollama Modelfile")
        logger.info("=" * 60)
    else:
        logger.error("❌ Model download failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()