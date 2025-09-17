#!/usr/bin/env python3
"""
Script to push the cardboard QC dataset to Hugging Face Hub
"""

import os
import argparse
from datasets import load_from_disk
from huggingface_hub import HfApi, login
import json

def create_dataset_card(dataset_info):
    """Create a comprehensive dataset card for the repository."""
    card_content = f"""---
license: cc-by-4.0
task_categories:
- image-classification
- visual-question-answering
tags:
- cardboard
- quality-control
- computer-vision
- manufacturing
- vision-language
size_categories:
- n<1K
language:
- en
pretty_name: "Cardboard Quality Control Dataset"
---

# Cardboard Quality Control Dataset

## Dataset Description

This dataset contains cardboard bundle images with quality assessments for training vision-language models to perform quality control in manufacturing environments.

### Dataset Summary

- **Total samples**: {dataset_info['total_samples']}
- **Task**: Binary classification of cardboard bundles (Pass/Fail)
- **Format**: Vision-language conversation format
- **Image resolution**: 640x640 pixels
- **Labels**: Pass (bundle appears flat), Fail (bundle appears warped)

### Dataset Splits

| Split | Samples | Pass | Fail |
|-------|---------|------|------|
| Train | {dataset_info['splits']['train']} | 75 | 42 |
| Validation | {dataset_info['splits']['validation']} | 14 | 2 |
| Test | {dataset_info['splits']['test']} | 25 | 10 |

## Dataset Structure

### Data Fields

- `image`: PIL Image of cardboard bundle (640x640 RGB)
- `conversations`: List of conversation turns between user and assistant
- `filename`: Original image filename
- `label`: Binary label ("Pass" or "Fail")
- `reason`: Text explanation of the assessment

### Data Format

Each sample contains a conversation where:
- **User**: "Analyze this cardboard bundle image. Is the bundle flat or warped? Provide your assessment."
- **Assistant**: Provides assessment based on the label and reason

Example conversation:
```python
{{
  "conversations": [
    {{
      "role": "user",
      "content": "Analyze this cardboard bundle image. Is the bundle flat or warped? Provide your assessment."
    }},
    {{
      "role": "assistant", 
      "content": "The bundle appears flat. Bundle appears flat"
    }}
  ]
}}
```

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("your-username/cardboard-qc-dataset")

# Access splits
train_data = dataset["train"]
val_data = dataset["validation"] 
test_data = dataset["test"]

# Example usage
sample = train_data[0]
image = sample["image"]  # PIL Image
conversation = sample["conversations"]  # List of messages
label = sample["label"]  # "Pass" or "Fail"
```

## Dataset Creation

This dataset was created from:
- **Images**: Overlay images from cardboard bundle photography
- **Labels**: Quality control assessments in CSV format
- **Processing**: Automated conversation generation for vision-language training

### Data Collection

- Images captured from cardboard manufacturing environment
- Quality assessments performed by human experts
- Labels indicate whether bundles meet flatness requirements for packaging

### Annotation Process

- Binary classification: Pass (flat) vs Fail (warped)
- Expert reasoning provided for each assessment
- Conversation format generated automatically for model training

## Intended Use

### Primary Use Cases

1. **Quality Control Automation**: Train models to automatically assess cardboard bundle quality
2. **Vision-Language Model Training**: Fine-tune multimodal models for manufacturing inspection
3. **Computer Vision Research**: Benchmark for industrial quality control tasks

### Limitations

- Limited to cardboard bundle assessment
- Binary classification only (no severity grading)
- Single camera angle/lighting condition
- Small dataset size ({dataset_info['total_samples']} samples)

## Ethics and Bias

- Dataset represents specific manufacturing environment
- May not generalize to different lighting/camera setups
- No personal or sensitive information included
- Focus on industrial/manufacturing use case

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{{cardboard_qc_2024,
  title={{Cardboard Quality Control Dataset}},
  author={{ADSYS Manufacturing}},
  year={{2024}},
  publisher={{Hugging Face}},
  url={{https://huggingface.co/datasets/your-username/cardboard-qc-dataset}}
}}
```

## License

This dataset is released under CC BY 4.0 License.

## Contact

For questions about this dataset, please open an issue in the repository.
"""
    return card_content


def push_dataset_to_hub(
    dataset_path: str,
    repo_name: str,
    hf_token: str,
    private: bool = False,
    commit_message: str = "Upload cardboard QC dataset"
):
    """Push the dataset to Hugging Face Hub."""
    
    print(f"Loading dataset from {dataset_path}...")
    
    # Load the dataset
    try:
        dataset = load_from_disk(dataset_path)
        print(f"Dataset loaded successfully!")
        print(f"Dataset structure: {dataset}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False
    
    # Login to Hugging Face
    print("Logging in to Hugging Face...")
    try:
        login(token=hf_token, add_to_git_credential=True)
        print("Successfully logged in!")
    except Exception as e:
        print(f"Error logging in: {e}")
        return False
    
    # Load dataset info for the card
    info_path = os.path.join(dataset_path, "dataset_info.json")
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            dataset_info = json.load(f)
    else:
        # Create basic info if not available
        dataset_info = {
            'total_samples': sum(len(split) for split in dataset.values()),
            'splits': {name: len(split) for name, split in dataset.items()}
        }
    
    # Create dataset card
    print("Creating dataset card...")
    card_content = create_dataset_card(dataset_info)
    
    # Push dataset to hub
    print(f"Pushing dataset to hub as '{repo_name}'...")
    try:
        dataset.push_to_hub(
            repo_id=repo_name,
            token=hf_token,
            private=private,
            commit_message=commit_message
        )
        print(f"Dataset successfully uploaded to: https://huggingface.co/datasets/{repo_name}")
        
        # Upload dataset card separately
        api = HfApi()
        api.upload_file(
            path_or_fileobj=card_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_name,
            repo_type="dataset",
            token=hf_token,
            commit_message="Add dataset card"
        )
        print("Dataset card uploaded successfully!")
        
        return True
        
    except Exception as e:
        print(f"Error uploading dataset: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Push cardboard QC dataset to Hugging Face Hub")
    parser.add_argument("--dataset_path", default="./datasets/qwen_cardboard_qc",
                        help="Path to the local dataset")
    parser.add_argument("--repo_name", default="cardboard-qc-dataset",
                        help="Name of the repository on Hugging Face Hub")
    parser.add_argument("--hf_token", required=True,
                        help="Hugging Face API token")
    parser.add_argument("--private", action="store_true",
                        help="Make the dataset private")
    parser.add_argument("--commit_message", default="Upload cardboard QC dataset",
                        help="Commit message for the upload")
    
    args = parser.parse_args()
    
    # Verify dataset path exists
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path not found: {args.dataset_path}")
        return
    
    # Push to hub
    success = push_dataset_to_hub(
        dataset_path=args.dataset_path,
        repo_name=args.repo_name,
        hf_token=args.hf_token,
        private=args.private,
        commit_message=args.commit_message
    )
    
    if success:
        print("\n" + "="*60)
        print("UPLOAD COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Dataset URL: https://huggingface.co/datasets/{args.repo_name}")
        print("\nYou can now:")
        print("1. View your dataset on Hugging Face Hub")
        print("2. Use it in training scripts with:")
        print(f"   from datasets import load_dataset")
        print(f"   dataset = load_dataset('{args.repo_name}')")
    else:
        print("\nUpload failed. Please check the error messages above.")


if __name__ == "__main__":
    main()