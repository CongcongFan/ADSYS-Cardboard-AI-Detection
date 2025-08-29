#!/usr/bin/env python3
"""
Direct LoRA inference script that bypasses the merge process entirely.
This allows using the LoRA adapter directly with the base model for inference.
Much more memory-efficient than merging and supports real-time inference.
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
import logging
from typing import Optional, List, Dict, Any, Union
import torch
import gc
import psutil
from PIL import Image
import requests
from io import BytesIO

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    AutoTokenizer,
    AutoConfig
)
from peft import PeftModel, PeftConfig
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DirectLoRAInferenceEngine:
    """
    Direct LoRA inference engine that loads LoRA adapter without merging.
    Provides memory-efficient inference for fine-tuned models.
    """
    
    def __init__(
        self, 
        base_model_path: str,
        lora_path: str,
        use_cpu: bool = False,
        torch_dtype: str = "auto"
    ):
        self.base_model_path = base_model_path
        self.lora_path = lora_path
        self.use_cpu = use_cpu
        self.device = "cpu" if use_cpu or not torch.cuda.is_available() else "cuda"
        
        # Configure torch dtype
        if torch_dtype == "auto":
            self.torch_dtype = torch.float32 if use_cpu else torch.float16
        elif torch_dtype == "float32":
            self.torch_dtype = torch.float32
        elif torch_dtype == "float16":
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.float16
        
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.is_loaded = False
        
        logger.info(f"Initialized DirectLoRAInferenceEngine")
        logger.info(f"Device: {self.device}, Dtype: {self.torch_dtype}")
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage information."""
        info = {}
        
        # RAM info
        ram = psutil.virtual_memory()
        info['ram_total'] = ram.total / (1024**3)
        info['ram_available'] = ram.available / (1024**3)
        info['ram_used_percent'] = ram.percent
        
        # GPU info if available
        if torch.cuda.is_available():
            info['gpu_total'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            info['gpu_allocated'] = torch.cuda.memory_allocated(0) / (1024**3)
            info['gpu_reserved'] = torch.cuda.memory_reserved(0) / (1024**3)
        
        return info
    
    def log_memory_usage(self, context: str = ""):
        """Log current memory usage."""
        info = self.get_memory_info()
        logger.info(f"Memory usage {context}:")
        logger.info(f"  RAM: {info['ram_used_percent']:.1f}% ({info['ram_available']:.1f}GB available)")
        if torch.cuda.is_available():
            logger.info(f"  GPU: {info['gpu_allocated']:.1f}GB allocated / {info['gpu_total']:.1f}GB total")
    
    def load_model(self):
        """Load the base model and LoRA adapter."""
        if self.is_loaded:
            logger.info("Model already loaded")
            return
        
        logger.info("Loading models for direct LoRA inference...")
        self.log_memory_usage("before loading")
        
        try:
            # Load LoRA config
            logger.info("Loading LoRA configuration...")
            peft_config = PeftConfig.from_pretrained(self.lora_path)
            logger.info(f"LoRA base model: {peft_config.base_model_name_or_path}")
            
            # Load model config
            logger.info("Loading model configuration...")
            model_config = AutoConfig.from_pretrained(self.base_model_path, trust_remote_code=True)
            
            # Configure loading parameters for memory efficiency
            load_kwargs = {
                "config": model_config,
                "torch_dtype": self.torch_dtype,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            
            if not self.use_cpu:
                load_kwargs["device_map"] = "auto"
            
            # Load base model
            logger.info("Loading base model...")
            logger.info("This may take a few minutes...")
            
            base_model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.base_model_path,
                **load_kwargs
            )
            
            logger.info("Base model loaded successfully!")
            self.log_memory_usage("after base model")
            
            # Load LoRA adapter
            logger.info("Loading LoRA adapter...")
            
            self.model = PeftModel.from_pretrained(
                base_model,
                self.lora_path,
                torch_dtype=self.torch_dtype,
                is_trainable=False  # Important for inference
            )
            
            logger.info("LoRA adapter loaded successfully!")
            self.log_memory_usage("after LoRA adapter")
            
            # Load tokenizer
            logger.info("Loading tokenizer...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.lora_path, 
                    trust_remote_code=True
                )
            except:
                # Fallback to base model tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.base_model_path, 
                    trust_remote_code=True
                )
            
            # Load processor
            logger.info("Loading processor...")
            try:
                self.processor = Qwen2VLProcessor.from_pretrained(self.lora_path)
            except:
                # Fallback to base model processor
                self.processor = Qwen2VLProcessor.from_pretrained(self.base_model_path)
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Move to device if using CPU
            if self.use_cpu:
                self.model = self.model.to("cpu")
            
            self.is_loaded = True
            
            logger.info("✅ All models loaded successfully!")
            self.log_memory_usage("final loading")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def unload_model(self):
        """Unload model to free memory."""
        if not self.is_loaded:
            return
        
        logger.info("Unloading models...")
        
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        if self.processor:
            del self.processor
        
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.is_loaded = False
        
        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Models unloaded successfully")
    
    def load_image(self, image_source: Union[str, Image.Image]) -> Image.Image:
        """Load image from file path, URL, or PIL Image."""
        if isinstance(image_source, Image.Image):
            return image_source
        
        if isinstance(image_source, str):
            if image_source.startswith(('http://', 'https://')):
                # Load from URL
                response = requests.get(image_source)
                image = Image.open(BytesIO(response.content))
            else:
                # Load from file path
                if not os.path.exists(image_source):
                    raise FileNotFoundError(f"Image file not found: {image_source}")
                image = Image.open(image_source)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
        
        raise ValueError("Image source must be a file path, URL, or PIL Image")
    
    def generate_response(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """Generate response using the LoRA-enhanced model."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Prepare conversation
            messages = []
            
            if image_path:
                # Load and prepare image
                image = self.load_image(image_path)
                messages.append({
                    "role": "user", 
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                })
            else:
                # Text-only conversation
                messages.append({
                    "role": "user",
                    "content": prompt
                })
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Process inputs
            if image_path:
                image = self.load_image(image_path)
                inputs = self.processor(
                    text=[text], 
                    images=[image], 
                    return_tensors="pt",
                    padding=True
                )
            else:
                inputs = self.processor(
                    text=[text], 
                    return_tensors="pt",
                    padding=True
                )
            
            # Move inputs to device
            if not self.use_cpu:
                inputs = inputs.to(self.device)
            
            # Generate response
            logger.debug("Generating response...")
            
            generation_config = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "temperature": temperature,
                "top_p": top_p,
                "pad_token_id": self.tokenizer.eos_token_id,
                **kwargs
            }
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    **generation_config
                )
            
            # Extract new tokens (skip input tokens)
            input_token_len = inputs.input_ids.shape[1]
            new_tokens = generated_ids[:, input_token_len:]
            
            # Decode response
            response = self.tokenizer.decode(
                new_tokens[0], 
                skip_special_tokens=True
            ).strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def interactive_chat(self):
        """Start an interactive chat session."""
        if not self.is_loaded:
            self.load_model()
        
        logger.info("🤖 Starting interactive chat with LoRA-enhanced model")
        logger.info("Type 'quit' or 'exit' to end the session")
        logger.info("Type 'image <path>' to analyze an image")
        logger.info("=" * 50)
        
        try:
            while True:
                # Get user input
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    break
                
                if not user_input:
                    continue
                
                # Check for image command
                image_path = None
                if user_input.lower().startswith('image '):
                    parts = user_input.split(' ', 1)
                    if len(parts) > 1:
                        image_path = parts[1].strip()
                        user_input = "Analyze this image and describe what you see."
                
                try:
                    # Generate response
                    start_time = time.time()
                    response = self.generate_response(
                        user_input,
                        image_path=image_path,
                        max_new_tokens=256,
                        temperature=0.7
                    )
                    end_time = time.time()
                    
                    # Display response
                    print(f"\nBot: {response}")
                    print(f"\n(Generated in {end_time - start_time:.1f}s)")
                    
                except Exception as e:
                    print(f"\nError generating response: {e}")
                    logger.error(f"Generation error: {e}")
        
        except KeyboardInterrupt:
            print("\nChat session interrupted by user")
        
        finally:
            logger.info("Chat session ended")

def check_requirements() -> bool:
    """Check if required packages are installed."""
    required_packages = [
        'torch',
        'transformers', 
        'peft',
        'pillow',
        'requests',
        'psutil'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'pillow':
                import PIL
            else:
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
    
    # Check essential files
    base_files = ["config.json"]
    for file in base_files:
        file_path = os.path.join(base_model_path, file)
        if not os.path.exists(file_path):
            logger.error(f"Missing base model file: {file_path}")
            return False
    
    lora_files = ["adapter_config.json", "adapter_model.safetensors"]
    for file in lora_files:
        file_path = os.path.join(lora_path, file)
        if not os.path.exists(file_path):
            logger.error(f"Missing LoRA file: {file_path}")
            return False
    
    logger.info("✅ All required paths and files found!")
    return True

def test_model_functionality(engine: DirectLoRAInferenceEngine) -> bool:
    """Test basic model functionality."""
    logger.info("Testing model functionality...")
    
    try:
        # Test text-only inference
        test_prompt = "What is cardboard quality control?"
        logger.info(f"Testing with prompt: '{test_prompt}'")
        
        response = engine.generate_response(
            test_prompt,
            max_new_tokens=100,
            temperature=0.7
        )
        
        logger.info(f"Response: {response}")
        
        if response and len(response.strip()) > 10:
            logger.info("✅ Model functionality test passed!")
            return True
        else:
            logger.warning("⚠️ Model returned short or empty response")
            return False
            
    except Exception as e:
        logger.error(f"❌ Model functionality test failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Direct LoRA Inference Engine")
    parser.add_argument(
        "--base-model-path",
        type=str,
        default="./base_model",
        help="Path to the base model (default: ./base_model)"
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default="./lora_model",
        help="Path to the LoRA adapter (default: ./lora_model)"
    )
    parser.add_argument(
        "--use-cpu",
        action="store_true",
        help="Force CPU usage (slower but uses less memory)"
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        choices=["auto", "float32", "float16"],
        default="auto",
        help="PyTorch data type (default: auto)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Single prompt to test (otherwise starts interactive mode)"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Image path for visual question answering"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate (default: 256)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--test-model",
        action="store_true",
        help="Run model functionality test"
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
    
    logger.info("=" * 80)
    logger.info("Direct LoRA Inference Engine")
    logger.info("=" * 80)
    logger.info(f"Base model path: {base_model_path}")
    logger.info(f"LoRA adapter path: {lora_path}")
    logger.info(f"Use CPU: {args.use_cpu}")
    logger.info(f"Torch dtype: {args.torch_dtype}")
    logger.info("=" * 80)
    
    # Check requirements
    if not args.skip_requirements_check:
        logger.info("Checking requirements...")
        if not check_requirements():
            sys.exit(1)
    
    # Check paths
    if not check_paths(base_model_path, lora_path):
        sys.exit(1)
    
    # Create inference engine
    try:
        engine = DirectLoRAInferenceEngine(
            base_model_path=base_model_path,
            lora_path=lora_path,
            use_cpu=args.use_cpu,
            torch_dtype=args.torch_dtype
        )
        
        # Load model
        engine.load_model()
        
        # Run tests if requested
        if args.test_model:
            if not test_model_functionality(engine):
                logger.error("Model functionality test failed")
                sys.exit(1)
        
        # Handle different modes
        if args.prompt:
            # Single prompt mode
            logger.info("Running single prompt inference...")
            response = engine.generate_response(
                args.prompt,
                image_path=args.image,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature
            )
            print(f"\nPrompt: {args.prompt}")
            if args.image:
                print(f"Image: {args.image}")
            print(f"Response: {response}")
        else:
            # Interactive mode
            engine.interactive_chat()
        
        # Cleanup
        engine.unload_model()
        
        logger.info("✅ Direct LoRA inference completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error in direct LoRA inference: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()