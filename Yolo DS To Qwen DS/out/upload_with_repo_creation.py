#!/usr/bin/env python3
"""
Enhanced script to upload the cardboard QC dataset to Hugging Face Hub
Creates repository if it doesn't exist and handles all edge cases.
"""

import os
from datasets import load_from_disk
from huggingface_hub import HfApi, login, create_repo
import json

def create_dataset_card(dataset_info, repo_name):
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
dataset = load_dataset("{repo_name}")

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
  url={{https://huggingface.co/datasets/{repo_name}}}
}}
```

## License

This dataset is released under CC BY 4.0 License.

## Contact

For questions about this dataset, please open an issue in the repository.
"""
    return card_content


def main():
    """Upload the dataset with full repository management."""
    
    # Configuration
    dataset_path = "./datasets/qwen_cardboard_qc"
    repo_name = "cardboard-qc-dataset"
    hf_token = "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"
    
    print("Cardboard QC Dataset Uploader (Enhanced)")
    print("="*60)
    print(f"Dataset path: {dataset_path}")
    print(f"Repository name: {repo_name}")
    print(f"Token: {hf_token[:10]}...")
    print()
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found at {dataset_path}")
        return False
    
    print("Loading dataset...")
    try:
        dataset = load_from_disk(dataset_path)
        print(f"Dataset loaded successfully: {dataset}")
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        return False
    
    # Login to Hugging Face
    print("\\nLogging in to Hugging Face...")
    try:
        login(token=hf_token, add_to_git_credential=True)
        api = HfApi()
        print("Successfully logged in!")
    except Exception as e:
        print(f"ERROR logging in: {e}")
        return False
    
    # Create repository if it doesn't exist
    print(f"\\nCreating repository '{repo_name}' (if it doesn't exist)...")
    try:
        create_repo(
            repo_id=repo_name,
            repo_type="dataset",
            token=hf_token,
            private=False,
            exist_ok=True
        )
        print(f"Repository '{repo_name}' ready!")
    except Exception as e:
        print(f"Note: Repository creation result: {e}")
        # Continue anyway as it might already exist
    
    # Load dataset info
    info_path = os.path.join(dataset_path, "dataset_info.json")
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            dataset_info = json.load(f)
    else:
        dataset_info = {
            'total_samples': sum(len(split) for split in dataset.values()),
            'splits': {name: len(split) for name, split in dataset.items()}
        }
    
    # Push dataset
    print("\\nUploading dataset...")
    try:
        dataset.push_to_hub(
            repo_id=repo_name,
            token=hf_token,
            private=False,
            commit_message="Upload cardboard quality control dataset for vision-language model training"
        )
        print(f"Dataset uploaded successfully!")
    except Exception as e:
        print(f"ERROR uploading dataset: {e}")
        return False
    
    # Create and upload dataset card
    print("\\nCreating dataset card...")
    try:
        card_content = create_dataset_card(dataset_info, repo_name)
        
        # Write to temporary file first
        readme_path = "temp_README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(card_content)
        
        # Upload the README
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_name,
            repo_type="dataset",
            token=hf_token,
            commit_message="Add comprehensive dataset card"
        )
        
        # Clean up temp file
        os.remove(readme_path)
        print("Dataset card uploaded successfully!")
        
    except Exception as e:
        print(f"Warning: Could not upload dataset card: {e}")
        print("Dataset is still accessible, just without the README.")
    
    # Final success message
    print("\\n" + "="*60)
    print("UPLOAD COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Dataset URL: https://huggingface.co/datasets/{repo_name}")
    print("\\nYou can now use it with:")
    print(f"from datasets import load_dataset")
    print(f"dataset = load_dataset('{repo_name}')")
    print("\\nDataset includes:")
    print(f"- {dataset_info['total_samples']} total samples")
    print(f"- Train: {dataset_info['splits']['train']} samples")
    print(f"- Validation: {dataset_info['splits']['validation']} samples")
    print(f"- Test: {dataset_info['splits']['test']} samples")
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        print("\\nUpload failed. Please check the error messages above.")
        exit(1)