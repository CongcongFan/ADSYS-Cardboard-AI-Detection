#!/usr/bin/env python3
"""
Script to merge LoRA adapter with the base Unsloth model.
This creates a full model with the fine-tuned weights integrated.
"""

import os
import sys
import argparse
from pathlib import Path
import logging
import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLProcessor,
    AutoTokenizer,
    AutoConfig
)
from peft import PeftModel, PeftConfig
import gc

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_requirements():
    """Check if required packages are installed."""
    required_packages = [
        'torch',
        'transformers', 
        'peft',
        'accelerate',
        'bitsandbytes'
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

def check_paths(base_model_path, lora_path):
    """Verify that required paths and files exist."""
    logger.info("Checking paths and files...")
    
    # Check base model path
    if not os.path.exists(base_model_path):
        logger.error(f"Base model path does not exist: {base_model_path}")
        return False
    
    # Check LoRA path
    if not os.path.exists(lora_path):
        logger.error(f"LoRA adapter path does not exist: {lora_path}")
        return False
    
    # Check essential base model files
    base_files = ["config.json", "tokenizer_config.json"]
    for file in base_files:
        file_path = os.path.join(base_model_path, file)
        if not os.path.exists(file_path):
            logger.error(f"Missing base model file: {file_path}")
            return False
    
    # Check essential LoRA files
    lora_files = ["adapter_config.json", "adapter_model.safetensors"]
    for file in lora_files:
        file_path = os.path.join(lora_path, file)
        if not os.path.exists(file_path):
            logger.error(f"Missing LoRA file: {file_path}")
            return False
    
    logger.info("✅ All required paths and files found!")
    return True

def get_memory_info():
    """Get current memory usage information."""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_allocated = torch.cuda.memory_allocated(0) / (1024**3)
        gpu_reserved = torch.cuda.memory_reserved(0) / (1024**3)
        logger.info(f"GPU Memory - Total: {gpu_memory:.1f}GB, Allocated: {gpu_allocated:.1f}GB, Reserved: {gpu_reserved:.1f}GB")
    
    import psutil
    ram = psutil.virtual_memory()
    logger.info(f"RAM - Total: {ram.total/(1024**3):.1f}GB, Available: {ram.available/(1024**3):.1f}GB, Used: {ram.percent}%")

def merge_lora_with_base(base_model_path, lora_path, output_path, use_cpu=False):
    """Merge LoRA adapter with base model."""
    logger.info("Starting LoRA merge process...")
    
    try:
        # Set device
        device = "cpu" if use_cpu or not torch.cuda.is_available() else "cuda"
        logger.info(f"Using device: {device}")
        
        get_memory_info()
        
        # Load LoRA config first
        logger.info("Loading LoRA configuration...")
        peft_config = PeftConfig.from_pretrained(lora_path)
        logger.info(f"LoRA config loaded. Base model: {peft_config.base_model_name_or_path}")
        
        # Verify base model compatibility
        expected_base = "unsloth/qwen2.5-vl-7b-instruct-bnb-4bit"
        if expected_base.lower() not in peft_config.base_model_name_or_path.lower():
            logger.warning(f"Base model mismatch. Expected: {expected_base}, Got: {peft_config.base_model_name_or_path}")
            logger.warning("Proceeding anyway, but this may cause issues...")
        
        # Load base model
        logger.info("Loading base model...")
        logger.info("This may take several minutes depending on your hardware...")
        
        model_config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
        
        # Load model with appropriate settings
        load_kwargs = {
            "config": model_config,
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
        }
        
        if device == "cuda":
            load_kwargs["device_map"] = "auto"
            load_kwargs["low_cpu_mem_usage"] = True
        
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_path,
            **load_kwargs
        )
        
        logger.info("✅ Base model loaded successfully!")
        get_memory_info()
        
        # Load LoRA adapter
        logger.info("Loading LoRA adapter...")
        model_with_lora = PeftModel.from_pretrained(
            base_model,
            lora_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        
        logger.info("✅ LoRA adapter loaded successfully!")
        get_memory_info()
        
        # Merge LoRA weights into base model
        logger.info("Merging LoRA weights with base model...")
        logger.info("This process may take 10-15 minutes...")
        
        merged_model = model_with_lora.merge_and_unload()
        
        logger.info("✅ LoRA merge completed!")
        
        # Clear memory
        del model_with_lora
        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        get_memory_info()
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Save merged model
        logger.info(f"Saving merged model to: {output_path}")
        logger.info("This may take several minutes...")
        
        merged_model.save_pretrained(
            output_path,
            safe_serialization=True,
            max_shard_size="2GB"
        )
        
        logger.info("✅ Merged model saved!")
        
        # Copy tokenizer and processor files
        logger.info("Copying tokenizer and processor files...")
        
        try:
            # Load and save tokenizer
            tokenizer = AutoTokenizer.from_pretrained(lora_path, trust_remote_code=True)
            tokenizer.save_pretrained(output_path)
            logger.info("✅ Tokenizer saved!")
        except Exception as e:
            logger.warning(f"Could not copy tokenizer from LoRA path: {e}")
            try:
                # Fallback to base model tokenizer
                tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
                tokenizer.save_pretrained(output_path)
                logger.info("✅ Tokenizer saved from base model!")
            except Exception as e2:
                logger.error(f"Failed to save tokenizer: {e2}")
        
        try:
            # Load and save processor
            processor = Qwen2VLProcessor.from_pretrained(lora_path)
            processor.save_pretrained(output_path)
            logger.info("✅ Processor saved!")
        except Exception as e:
            logger.warning(f"Could not copy processor from LoRA path: {e}")
            try:
                # Fallback to base model processor
                processor = Qwen2VLProcessor.from_pretrained(base_model_path)
                processor.save_pretrained(output_path)
                logger.info("✅ Processor saved from base model!")
            except Exception as e2:
                logger.error(f"Failed to save processor: {e2}")
        
        # Clean up
        del merged_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        logger.info("✅ Merge process completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during merge process: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with Unsloth base model")
    parser.add_argument(
        "--base-model-path",
        type=str,
        default="./base_model",
        help="Path to the downloaded base model (default: ./base_model)"
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default="./lora_model",
        help="Path to the LoRA adapter (default: ./lora_model)"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./merged_model",
        help="Path to save the merged model (default: ./merged_model)"
    )
    parser.add_argument(
        "--use-cpu",
        action="store_true",
        help="Force CPU usage (slower but uses less VRAM)"
    )
    parser.add_argument(
        "--skip-requirements-check",
        action="store_true",
        help="Skip checking for required packages"
    )
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    base_model_path = os.path.abspath(args.base_model_path)
    lora_path = os.path.abspath(args.lora_path)
    output_path = os.path.abspath(args.output_path)
    
    logger.info("=" * 60)
    logger.info("LoRA Merge Script")
    logger.info("=" * 60)
    logger.info(f"Base model path: {base_model_path}")
    logger.info(f"LoRA adapter path: {lora_path}")
    logger.info(f"Output path: {output_path}")
    logger.info("=" * 60)
    
    # Check requirements
    if not args.skip_requirements_check:
        logger.info("Checking requirements...")
        if not check_requirements():
            sys.exit(1)
    
    # Check paths
    if not check_paths(base_model_path, lora_path):
        sys.exit(1)
    
    # Check hardware
    if torch.cuda.is_available() and not args.use_cpu:
        logger.info(f"CUDA available with {torch.cuda.device_count()} GPU(s)")
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU Memory: {gpu_memory:.1f}GB")
        
        if gpu_memory < 12:
            logger.warning("GPU has less than 12GB memory. Merge may be slow or fail.")
            logger.warning("Consider using --use-cpu flag if you encounter memory issues.")
    else:
        logger.warning("Using CPU for merge. This will be significantly slower!")
    
    # Perform merge
    success = merge_lora_with_base(
        base_model_path, 
        lora_path, 
        output_path,
        args.use_cpu
    )
    
    if success:
        logger.info("=" * 60)
        logger.info("✅ LORA MERGE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"Merged model location: {output_path}")
        logger.info("Next steps:")
        logger.info("1. Run 03_convert_to_gguf.py to convert for Ollama")
        logger.info("2. Use 04_create_modelfile.py to create Ollama Modelfile")
        logger.info("3. Test with 05_test_model.py")
        logger.info("=" * 60)
    else:
        logger.error("❌ LoRA merge failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()