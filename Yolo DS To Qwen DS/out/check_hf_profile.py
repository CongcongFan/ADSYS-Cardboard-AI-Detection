#!/usr/bin/env python3
"""
Check Hugging Face profile and find the correct dataset name
"""

from huggingface_hub import HfApi, login

def check_profile_and_datasets():
    """Check the HF profile and list datasets."""
    
    hf_token = "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"
    
    try:
        # Login and get API
        login(token=hf_token, add_to_git_credential=True)
        api = HfApi()
        
        # Get user info
        user_info = api.whoami(token=hf_token)
        username = user_info['name']
        
        print(f"Logged in as: {username}")
        print(f"User type: {user_info.get('type', 'user')}")
        
        # List user's datasets
        print(f"\nDatasets owned by {username}:")
        datasets = api.list_datasets(author=username, token=hf_token)
        
        for i, dataset in enumerate(datasets):
            print(f"{i+1}. {dataset.id}")
            if hasattr(dataset, 'tags') and dataset.tags:
                print(f"   Tags: {dataset.tags}")
        
        # Check for our specific dataset
        target_datasets = [d for d in datasets if 'cardboard' in d.id.lower()]
        
        if target_datasets:
            print(f"\nFound cardboard-related datasets:")
            for dataset in target_datasets:
                full_name = dataset.id
                print(f"  - {full_name}")
                print(f"    URL: https://huggingface.co/datasets/{full_name}")
                
                # Test loading this dataset
                try:
                    from datasets import load_dataset
                    test_ds = load_dataset(full_name)
                    print(f"    ✅ Can be loaded: {test_ds}")
                except Exception as e:
                    print(f"    ❌ Loading error: {e}")
        else:
            print("\nNo cardboard datasets found. The upload might still be processing.")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_profile_and_datasets()