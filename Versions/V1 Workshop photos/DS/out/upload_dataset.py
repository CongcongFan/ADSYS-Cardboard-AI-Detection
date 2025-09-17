#!/usr/bin/env python3
"""
Simple script to upload the cardboard QC dataset to Hugging Face Hub
Uses the provided token and sensible defaults.
"""

import subprocess
import sys
import os

def main():
    """Upload the dataset with predefined settings."""
    
    # Configuration
    dataset_path = "./datasets/qwen_cardboard_qc"
    repo_name = "cardboard-qc-dataset"
    hf_token = "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"
    
    print("Cardboard QC Dataset Uploader")
    print("="*50)
    print(f"Dataset path: {dataset_path}")
    print(f"Repository name: {repo_name}")
    print(f"Token: {hf_token[:10]}...")
    print()
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found at {dataset_path}")
        print("Please run the dataset construction script first.")
        return
    
    # Check if required packages are installed
    try:
        import datasets
        import huggingface_hub
        print("Required packages found: datasets, huggingface_hub")
    except ImportError as e:
        print(f"ERROR: Missing required package: {e}")
        print("Please install with: pip install datasets huggingface_hub")
        return
    
    # Run the upload script
    cmd = [
        sys.executable, "push_to_huggingface.py",
        "--dataset_path", dataset_path,
        "--repo_name", repo_name,
        "--hf_token", hf_token,
        "--commit_message", "Upload cardboard quality control dataset for vision-language model training"
    ]
    
    print("Starting upload...")
    print("Command:", " ".join(cmd[:6] + ["--hf_token", "***", "--commit_message", cmd[-1]]))
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\nUpload completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\nUpload failed with error code {e.returncode}")
        return
    except KeyboardInterrupt:
        print("\nUpload cancelled by user.")
        return

if __name__ == "__main__":
    main()