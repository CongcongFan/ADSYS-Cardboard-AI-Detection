#!/usr/bin/env python3
"""
Memory-efficient LoRA merge script that handles low-memory systems.
Includes gradient checkpointing, sequential processing, and memory optimization.
"""

import os
import sys
import argparse
from pathlib import Path
import logging
import torch
import gc
import psutil
import time
from typing import Optional, Dict, Any

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLProcessor,
    AutoTokenizer,
    AutoConfig
)
from peft import PeftModel, PeftConfig
import warnings

# Suppress warnings to reduce memory overhead
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryManager:
    """Manages memory usage and provides monitoring utilities."""
    
    def __init__(self):
        self.initial_memory = self.get_memory_info()
        self.peak_memory = self.initial_memory.copy()
        
    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage information."""
        info = {}
        
        # RAM info
        ram = psutil.virtual_memory()
        info['ram_total'] = ram.total / (1024**3)
        info['ram_available'] = ram.available / (1024**3)
        info['ram_used_percent'] = ram.percent
        info['ram_used'] = (ram.total - ram.available) / (1024**3)
        
        # GPU info if available
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            info['gpu_total'] = gpu_props.total_memory / (1024**3)
            info['gpu_allocated'] = torch.cuda.memory_allocated(0) / (1024**3)
            info['gpu_reserved'] = torch.cuda.memory_reserved(0) / (1024**3)
            info['gpu_free'] = info['gpu_total'] - info['gpu_reserved']
        else:
            info['gpu_total'] = 0
            info['gpu_allocated'] = 0
            info['gpu_reserved'] = 0
            info['gpu_free'] = 0
            
        return info
    
    def log_memory_usage(self, context: str = ""):
        """Log current memory usage with context."""
        info = self.get_memory_info()
        
        # Track peak memory
        for key in ['ram_used', 'gpu_allocated', 'gpu_reserved']:
            if key in info and info[key] > self.peak_memory.get(key, 0):
                self.peak_memory[key] = info[key]
        
        logger.info(f"Memory Usage {context}:")
        logger.info(f"  RAM: {info['ram_used']:.1f}GB/{info['ram_total']:.1f}GB ({info['ram_used_percent']:.1f}%)")
        if info['gpu_total'] > 0:
            logger.info(f"  GPU: {info['gpu_allocated']:.1f}GB allocated, {info['gpu_reserved']:.1f}GB reserved, {info['gpu_free']:.1f}GB free")
    
    def check_memory_requirements(self, required_ram_gb: float = 8.0, required_gpu_gb: float = 6.0) -> bool:
        """Check if system has sufficient memory for the operation."""
        info = self.get_memory_info()
        
        if info['ram_available'] < required_ram_gb:
            logger.warning(f"Low RAM: {info['ram_available']:.1f}GB available, {required_ram_gb}GB recommended")
            return False
            
        if torch.cuda.is_available() and info['gpu_free'] < required_gpu_gb:
            logger.warning(f"Low GPU memory: {info['gpu_free']:.1f}GB available, {required_gpu_gb}GB recommended")
            return False
            
        return True
    
    def aggressive_cleanup(self):
        """Perform aggressive memory cleanup."""
        # Python garbage collection
        collected = gc.collect()
        logger.debug(f"Garbage collected {collected} objects")
        
        # PyTorch cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Give system time to reclaim memory
        time.sleep(0.5)

class MemoryEfficientLoRAMerger:
    """Memory-efficient LoRA merger with advanced optimizations."""
    
    def __init__(self, memory_manager: MemoryManager, use_cpu: bool = False):
        self.memory_manager = memory_manager
        self.use_cpu = use_cpu
        self.device = "cpu" if use_cpu or not torch.cuda.is_available() else "cuda"
        
        # Memory optimization settings
        self.torch_dtype = torch.float32 if use_cpu else torch.float16
        self.low_cpu_mem_usage = True
        self.max_memory_per_gpu = None
        
        # Configure PyTorch for memory efficiency
        if torch.cuda.is_available() and not use_cpu:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            
    def configure_memory_settings(self):
        """Configure memory settings based on available hardware."""
        memory_info = self.memory_manager.get_memory_info()
        
        if not self.use_cpu and torch.cuda.is_available():
            # Reserve some GPU memory for system
            available_gpu = memory_info['gpu_free']
            if available_gpu < 8.0:
                self.max_memory_per_gpu = f"{max(1, int(available_gpu * 0.8))}GiB"
                logger.info(f"Limited GPU memory per device: {self.max_memory_per_gpu}")
        
        # Set conservative memory fraction
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.85)
    
    def load_model_with_memory_optimization(self, model_path: str, **kwargs):
        """Load model with memory optimizations."""
        logger.info(f"Loading model from {model_path} with memory optimization...")
        
        self.memory_manager.log_memory_usage("before model loading")
        
        # Configure loading parameters for memory efficiency
        load_kwargs = {
            "torch_dtype": self.torch_dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": self.low_cpu_mem_usage,
            **kwargs
        }
        
        if not self.use_cpu:
            device_map = "auto"
            if self.max_memory_per_gpu:
                device_map = {"": 0}
                load_kwargs["max_memory"] = {0: self.max_memory_per_gpu}
            load_kwargs["device_map"] = device_map
        
        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                **load_kwargs
            )
            
            self.memory_manager.log_memory_usage("after model loading")
            return model
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "not enough memory" in str(e).lower():
                logger.error("Out of memory while loading model")
                logger.info("Trying with CPU fallback...")
                
                # Force CPU loading
                load_kwargs = {
                    "torch_dtype": torch.float32,
                    "trust_remote_code": True,
                    "low_cpu_mem_usage": True
                }
                
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path,
                    **load_kwargs
                )
                
                self.device = "cpu"
                self.use_cpu = True
                logger.warning("Switched to CPU mode due to memory constraints")
                return model
            else:
                raise e
    
    def merge_lora_sequential(self, base_model, lora_path: str) -> torch.nn.Module:
        """Merge LoRA weights sequentially to minimize memory usage."""
        logger.info("Loading LoRA adapter...")
        self.memory_manager.log_memory_usage("before LoRA loading")
        
        try:
            # Load LoRA with memory optimization
            model_with_lora = PeftModel.from_pretrained(
                base_model,
                lora_path,
                torch_dtype=self.torch_dtype,
                is_trainable=False  # Important for memory efficiency
            )
            
            self.memory_manager.log_memory_usage("after LoRA loading")
            
            logger.info("Starting sequential LoRA merge...")
            logger.info("This process uses less memory but may take longer...")
            
            # Enable gradient checkpointing to save memory during merge
            if hasattr(model_with_lora, 'gradient_checkpointing_enable'):
                model_with_lora.gradient_checkpointing_enable()
            
            # Perform merge with memory cleanup
            with torch.no_grad():  # Disable gradients to save memory
                merged_model = model_with_lora.merge_and_unload()
            
            self.memory_manager.log_memory_usage("after merge")
            
            # Cleanup LoRA model
            del model_with_lora
            self.memory_manager.aggressive_cleanup()
            
            return merged_model
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "not enough memory" in str(e).lower():
                logger.error("Out of memory during LoRA merge")
                logger.info("Consider using CPU mode with --use-cpu flag")
                raise RuntimeError(
                    "Insufficient memory for LoRA merge. Try using --use-cpu or "
                    "use the direct inference mode (06_direct_lora_inference.py) instead."
                )
            else:
                raise e
    
    def save_model_efficiently(self, model, output_path: str):
        """Save model with memory-efficient settings."""
        logger.info(f"Saving merged model to: {output_path}")
        
        os.makedirs(output_path, exist_ok=True)
        
        # Save in chunks to reduce memory usage
        save_kwargs = {
            "safe_serialization": True,
            "max_shard_size": "1GB"  # Smaller shards for memory efficiency
        }
        
        self.memory_manager.log_memory_usage("before model saving")
        
        with torch.no_grad():
            model.save_pretrained(output_path, **save_kwargs)
        
        self.memory_manager.log_memory_usage("after model saving")
        logger.info("Model saved successfully!")

def check_requirements():
    """Check if required packages are installed."""
    required_packages = [
        'torch',
        'transformers', 
        'peft',
        'accelerate',
        'psutil'
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

def check_paths(base_model_path: str, lora_path: str) -> bool:
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
    
    logger.info("All required paths and files found!")
    return True

def merge_lora_memory_efficient(
    base_model_path: str, 
    lora_path: str, 
    output_path: str,
    use_cpu: bool = False,
    force_merge: bool = False
) -> bool:
    """Memory-efficient LoRA merge with advanced optimizations."""
    
    memory_manager = MemoryManager()
    merger = MemoryEfficientLoRAMerger(memory_manager, use_cpu)
    
    logger.info("Starting memory-efficient LoRA merge process...")
    memory_manager.log_memory_usage("initial")
    
    try:
        # Configure memory settings
        merger.configure_memory_settings()
        
        # Check memory requirements
        required_ram = 6.0 if use_cpu else 4.0
        required_gpu = 0.0 if use_cpu else 4.0
        
        if not force_merge and not memory_manager.check_memory_requirements(required_ram, required_gpu):
            logger.warning("System may not have sufficient memory for merge")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                logger.info("Merge cancelled. Consider using:")
                logger.info("1. --use-cpu flag for CPU-only merge")
                logger.info("2. 06_direct_lora_inference.py for direct inference")
                return False
        
        # Load LoRA config first
        logger.info("Loading LoRA configuration...")
        peft_config = PeftConfig.from_pretrained(lora_path)
        logger.info(f"LoRA config loaded. Base model: {peft_config.base_model_name_or_path}")
        
        # Load base model with optimizations
        logger.info("Loading base model with memory optimizations...")
        model_config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
        
        base_model = merger.load_model_with_memory_optimization(
            base_model_path,
            config=model_config
        )
        
        logger.info("Base model loaded successfully!")
        
        # Perform sequential LoRA merge
        merged_model = merger.merge_lora_sequential(base_model, lora_path)
        
        # Clean up base model
        del base_model
        memory_manager.aggressive_cleanup()
        
        logger.info("LoRA merge completed successfully!")
        
        # Save merged model efficiently
        merger.save_model_efficiently(merged_model, output_path)
        
        # Copy tokenizer and processor files
        logger.info("Copying tokenizer and processor files...")
        
        try:
            # Load and save tokenizer
            tokenizer = AutoTokenizer.from_pretrained(lora_path, trust_remote_code=True)
            tokenizer.save_pretrained(output_path)
            logger.info("Tokenizer saved!")
        except Exception as e:
            logger.warning(f"Could not copy tokenizer from LoRA path: {e}")
            try:
                # Fallback to base model tokenizer
                tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
                tokenizer.save_pretrained(output_path)
                logger.info("Tokenizer saved from base model!")
            except Exception as e2:
                logger.error(f"Failed to save tokenizer: {e2}")
        
        try:
            # Load and save processor
            processor = Qwen2VLProcessor.from_pretrained(lora_path)
            processor.save_pretrained(output_path)
            logger.info("Processor saved!")
        except Exception as e:
            logger.warning(f"Could not copy processor from LoRA path: {e}")
            try:
                # Fallback to base model processor
                processor = Qwen2VLProcessor.from_pretrained(base_model_path)
                processor.save_pretrained(output_path)
                logger.info("Processor saved from base model!")
            except Exception as e2:
                logger.error(f"Failed to save processor: {e2}")
        
        # Final cleanup
        del merged_model
        memory_manager.aggressive_cleanup()
        
        # Log memory usage summary
        memory_manager.log_memory_usage("final")
        logger.info("Peak memory usage during merge:")
        logger.info(f"  RAM: {memory_manager.peak_memory.get('ram_used', 0):.1f}GB")
        if memory_manager.peak_memory.get('gpu_allocated', 0) > 0:
            logger.info(f"  GPU: {memory_manager.peak_memory.get('gpu_allocated', 0):.1f}GB")
        
        logger.info("Memory-efficient merge process completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during memory-efficient merge process: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Suggest alternatives
        logger.info("\nTroubleshooting suggestions:")
        logger.info("1. Try --use-cpu flag for CPU-only merge")
        logger.info("2. Use 06_direct_lora_inference.py for inference without merge")
        logger.info("3. Close other applications to free up memory")
        logger.info("4. Use --force-merge to bypass memory checks")
        
        return False

def main():
    parser = argparse.ArgumentParser(description="Memory-efficient LoRA adapter merge")
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
        help="Force CPU usage (slower but uses less GPU memory)"
    )
    parser.add_argument(
        "--force-merge",
        action="store_true",
        help="Force merge even with low memory (risky)"
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
    
    logger.info("=" * 80)
    logger.info("Memory-Efficient LoRA Merge Script")
    logger.info("=" * 80)
    logger.info(f"Base model path: {base_model_path}")
    logger.info(f"LoRA adapter path: {lora_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Use CPU: {args.use_cpu}")
    logger.info(f"Force merge: {args.force_merge}")
    logger.info("=" * 80)
    
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
    else:
        logger.info("Using CPU for merge (this will be slower)")
    
    # Perform merge
    success = merge_lora_memory_efficient(
        base_model_path, 
        lora_path, 
        output_path,
        args.use_cpu,
        args.force_merge
    )
    
    if success:
        logger.info("=" * 80)
        logger.info("MEMORY-EFFICIENT LORA MERGE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"Merged model location: {output_path}")
        logger.info("Next steps:")
        logger.info("1. Run 03_convert_to_gguf.py to convert for Ollama")
        logger.info("2. Use 04_create_modelfile.py to create Ollama Modelfile")
        logger.info("3. Test with 05_test_model.py")
        logger.info("=" * 80)
    else:
        logger.error("Memory-efficient LoRA merge failed!")
        logger.info("Consider using 06_direct_lora_inference.py for inference without merge")
        sys.exit(1)

if __name__ == "__main__":
    main()