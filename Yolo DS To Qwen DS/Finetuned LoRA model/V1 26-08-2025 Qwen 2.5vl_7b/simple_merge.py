#!/usr/bin/env python3
"""
Simple LoRA merge script using unsloth for memory efficiency
"""

import os
import torch
from unsloth import FastLanguageModel

def merge_lora():
    """Merge LoRA adapter with base model"""
    
    print("🔧 Starting LoRA merge with Unsloth...")
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Paths
    base_model_name = "unsloth/qwen2.5-vl-7b-instruct-bnb-4bit"
    lora_path = "./lora_model"
    output_path = "./merged_model"
    
    try:
        # Load model with LoRA adapter
        print("📥 Loading model with LoRA adapter...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_name,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        
        # Load LoRA weights
        print("🔄 Applying LoRA weights...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )
        
        # Load trained LoRA weights
        print("📂 Loading trained LoRA weights...")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)
        
        # Merge and unload
        print("🔀 Merging LoRA with base model...")
        model = model.merge_and_unload()
        
        # Save merged model
        print(f"💾 Saving merged model to {output_path}...")
        os.makedirs(output_path, exist_ok=True)
        
        # Save in HuggingFace format
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        
        print("✅ LoRA merge completed successfully!")
        print(f"Merged model saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during merge: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = merge_lora()
    exit(0 if success else 1)