#!/usr/bin/env python3
"""
Hugging Face Dataset Constructor for Cardboard QC
Creates a vision-language dataset from overlay images and QC labels for fine-tuning.
"""

import os
import pandas as pd
from PIL import Image
from datasets import Dataset, DatasetDict, Features, Value, Image as HFImage
import json
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


def load_and_process_labels(csv_path: str) -> pd.DataFrame:
    """Load the QC labels CSV and process it."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} labels from {csv_path}")
    
    # Display label distribution
    label_counts = df['label'].value_counts()
    print(f"Label distribution: {dict(label_counts)}")
    
    return df


def create_conversation_format(label: str, reason: str) -> List[Dict]:
    """Create a conversation format for the dataset."""
    user_message = "Analyze this cardboard bundle image. Is the bundle flat or warped? Provide your assessment."
    
    if label == "Pass":
        assistant_message = f"The bundle appears flat. {reason}"
    else:  # Fail
        assistant_message = f"The bundle appears warped. {reason}"
    
    return [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_message}
    ]


def process_dataset_entries(images_dir: str, labels_df: pd.DataFrame) -> List[Dict]:
    """Process all entries to create the dataset."""
    dataset_entries = []
    missing_images = []
    
    for idx, row in labels_df.iterrows():
        filename = row['file']
        label = row['label']
        reason = row['reason']
        
        image_path = os.path.join(images_dir, filename)
        
        if not os.path.exists(image_path):
            missing_images.append(filename)
            continue
        
        # Load and verify image
        try:
            image = Image.open(image_path)
            image.verify()  # Verify the image can be opened
            image = Image.open(image_path)  # Reopen after verify
        except Exception as e:
            print(f"Error loading image {filename}: {e}")
            continue
        
        # Create conversation format
        conversations = create_conversation_format(label, reason)
        
        # Create dataset entry
        entry = {
            'image': image,
            'conversations': conversations,
            'filename': filename,
            'label': label,
            'reason': reason
        }
        
        dataset_entries.append(entry)
    
    if missing_images:
        print(f"Warning: {len(missing_images)} images not found:")
        for img in missing_images[:5]:  # Show first 5
            print(f"  - {img}")
        if len(missing_images) > 5:
            print(f"  ... and {len(missing_images) - 5} more")
    
    return dataset_entries


def create_huggingface_dataset(
    images_dir: str,
    labels_csv: str,
    output_dir: str,
    dataset_name: str = "cardboard_qc_dataset",
    test_size: float = 0.2,
    val_size: float = 0.1
) -> DatasetDict:
    """Create a Hugging Face dataset from images and labels."""
    
    # Load labels
    labels_df = load_and_process_labels(labels_csv)
    
    # Process entries
    print("Processing dataset entries...")
    dataset_entries = process_dataset_entries(images_dir, labels_df)
    print(f"Successfully processed {len(dataset_entries)} entries")
    
    # Define dataset features
    features = Features({
        'image': HFImage(),
        'conversations': [
            {
                'role': Value('string'),
                'content': Value('string')
            }
        ],
        'filename': Value('string'),
        'label': Value('string'),
        'reason': Value('string')
    })
    
    # Create dataset
    dataset = Dataset.from_list(dataset_entries, features=features)
    
    # Split dataset
    print("Splitting dataset...")
    
    # First split: train + temp
    train_test_split = dataset.train_test_split(test_size=test_size + val_size, seed=42)
    train_dataset = train_test_split['train']
    temp_dataset = train_test_split['test']
    
    # Second split: validation + test
    val_test_ratio = val_size / (test_size + val_size)
    val_test_split = temp_dataset.train_test_split(test_size=1-val_test_ratio, seed=42)
    val_dataset = val_test_split['train']
    test_dataset = val_test_split['test']
    
    # Create dataset dict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })
    
    # Print split statistics
    print(f"\nDataset splits:")
    for split_name, split_data in dataset_dict.items():
        labels = [entry['label'] for entry in split_data]
        pass_count = labels.count('Pass')
        fail_count = labels.count('Fail')
        print(f"  {split_name}: {len(split_data)} samples (Pass: {pass_count}, Fail: {fail_count})")
    
    # Save dataset
    print(f"\nSaving dataset to {output_dir}/{dataset_name}...")
    dataset_path = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_path, exist_ok=True)
    dataset_dict.save_to_disk(dataset_path)
    
    # Save dataset info
    info = {
        'dataset_name': dataset_name,
        'total_samples': len(dataset),
        'splits': {split_name: len(split_data) for split_name, split_data in dataset_dict.items()},
        'features': ['image', 'conversations', 'filename', 'label', 'reason'],
        'conversation_format': 'User asks for cardboard bundle assessment, assistant provides analysis',
        'labels': ['Pass', 'Fail'],
        'source_images_dir': images_dir,
        'source_labels_csv': labels_csv
    }
    
    info_path = os.path.join(dataset_path, 'dataset_info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"Dataset creation completed!")
    print(f"Dataset saved to: {dataset_path}")
    print(f"Dataset info saved to: {info_path}")
    
    return dataset_dict


def main():
    """Main function to create the dataset."""
    parser = argparse.ArgumentParser(description="Create Hugging Face dataset for cardboard QC")
    parser.add_argument("--images_dir", default="./overlays_overlay", 
                        help="Directory containing overlay images")
    parser.add_argument("--labels_csv", default="./qc_labels.csv", 
                        help="CSV file with labels")
    parser.add_argument("--output_dir", default="./datasets", 
                        help="Output directory for the dataset")
    parser.add_argument("--dataset_name", default="cardboard_qc_dataset", 
                        help="Name of the dataset")
    parser.add_argument("--test_size", type=float, default=0.2, 
                        help="Proportion of data for test set")
    parser.add_argument("--val_size", type=float, default=0.1, 
                        help="Proportion of data for validation set")
    
    args = parser.parse_args()
    
    # Verify inputs exist
    if not os.path.exists(args.images_dir):
        print(f"Error: Images directory not found: {args.images_dir}")
        return
    
    if not os.path.exists(args.labels_csv):
        print(f"Error: Labels CSV not found: {args.labels_csv}")
        return
    
    # Create dataset
    dataset_dict = create_huggingface_dataset(
        images_dir=args.images_dir,
        labels_csv=args.labels_csv,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        test_size=args.test_size,
        val_size=args.val_size
    )
    
    # Show example
    print("\n" + "="*50)
    print("Example entry from training set:")
    print("="*50)
    example = dataset_dict['train'][0]
    print(f"Filename: {example['filename']}")
    print(f"Label: {example['label']}")
    print(f"Reason: {example['reason']}")
    print("Conversation:")
    for msg in example['conversations']:
        print(f"  {msg['role']}: {msg['content']}")
    print(f"Image size: {example['image'].size}")


if __name__ == "__main__":
    main()