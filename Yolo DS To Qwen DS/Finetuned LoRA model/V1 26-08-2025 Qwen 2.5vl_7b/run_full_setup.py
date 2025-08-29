#!/usr/bin/env python3
"""
Master script to run the complete LoRA to Ollama setup process.
This orchestrates all the individual scripts in the correct order.
"""

import os
import sys
import argparse
import subprocess
import time
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SetupOrchestrator:
    """Orchestrates the complete setup process."""
    
    def __init__(self, config):
        self.config = config
        self.start_time = time.time()
        self.current_step = 0
        self.total_steps = 6  # Updated to include memory check
        self.memory_analysis = None
        
    def log_step(self, step_name, step_description):
        """Log the current step with progress."""
        self.current_step += 1
        logger.info("=" * 80)
        logger.info(f"STEP {self.current_step}/{self.total_steps}: {step_name}")
        logger.info(f"Description: {step_description}")
        logger.info("=" * 80)
        
    def run_script(self, script_name, args=None):
        """Run a Python script with error handling."""
        if args is None:
            args = []
            
        cmd = [sys.executable, script_name] + args
        logger.info(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.get('timeout', 3600)  # 1 hour default timeout
            )
            
            if result.returncode == 0:
                logger.info("✅ Script completed successfully!")
                if result.stdout.strip():
                    logger.info("Script output:")
                    for line in result.stdout.strip().split('\n'):
                        logger.info(f"  {line}")
                return True
            else:
                logger.error(f"❌ Script failed with return code {result.returncode}")
                if result.stderr.strip():
                    logger.error("Error output:")
                    for line in result.stderr.strip().split('\n'):
                        logger.error(f"  {line}")
                if result.stdout.strip():
                    logger.info("Standard output:")
                    for line in result.stdout.strip().split('\n'):
                        logger.info(f"  {line}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Script timed out after {self.config.get('timeout', 3600)} seconds")
            return False
        except Exception as e:
            logger.error(f"❌ Error running script: {e}")
            return False
    
    def run_memory_analysis(self):
        """Run memory analysis to determine optimal strategy."""
        if not self.config.get('skip_memory_check', False):
            self.log_step(
                "Memory Analysis",
                "Analyzing system resources to determine optimal processing strategy"
            )
            
            try:
                # Run memory check script
                result = subprocess.run(
                    [sys.executable, "07_memory_check.py", "--format", "json"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    import json
                    self.memory_analysis = json.loads(result.stdout)
                    logger.info("Memory analysis completed")
                    
                    # Log key findings
                    mem_info = self.memory_analysis.get('memory_info', {})
                    logger.info(f"Available RAM: {mem_info.get('ram_available_gb', 0):.1f}GB")
                    
                    gpu_info = self.memory_analysis.get('gpu_info', {})
                    if gpu_info.get('cuda_available', False):
                        gpus = gpu_info.get('gpus', [])
                        if gpus:
                            best_gpu = max(gpus, key=lambda g: g.get('total_memory_gb', 0))
                            logger.info(f"GPU Memory: {best_gpu.get('total_memory_gb', 0):.1f}GB available")
                    
                    # Update configuration based on analysis
                    analysis = self.memory_analysis.get('analysis', {})
                    merge_feas = analysis.get('merge_feasibility', {})
                    
                    if not merge_feas.get('overall_feasible', False):
                        logger.warning("System not optimal for LoRA merge - recommending direct inference mode")
                        self.config['use_direct_inference'] = True
                    elif merge_feas.get('use_cpu_recommended', False):
                        logger.info("Recommending CPU mode for merge")
                        self.config['use_cpu'] = True
                        self.config['use_memory_efficient_merge'] = True
                    
                    return True
                else:
                    logger.warning(f"Memory analysis failed: {result.stderr}")
                    return True  # Continue anyway
                    
            except Exception as e:
                logger.warning(f"Memory analysis error: {e}")
                return True  # Continue anyway
        else:
            logger.info("Memory analysis skipped")
            return True
    
    def check_prerequisites(self):
        """Check prerequisites before starting setup."""
        logger.info("Checking prerequisites...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            logger.error("Python 3.8+ required")
            return False
            
        # Check if LoRA adapter exists
        lora_path = self.config['lora_path']
        if not os.path.exists(lora_path):
            logger.error(f"LoRA adapter not found at: {lora_path}")
            return False
            
        essential_lora_files = ["adapter_config.json", "adapter_model.safetensors"]
        for file in essential_lora_files:
            file_path = os.path.join(lora_path, file)
            if not os.path.exists(file_path):
                logger.error(f"Missing LoRA file: {file_path}")
                return False
        
        # Check disk space (approximate)
        import shutil
        free_space = shutil.disk_usage('.').free / (1024**3)  # GB
        required_space = 20 if self.config.get('use_direct_inference', False) else 50  # GB
        
        if free_space < required_space:
            logger.warning(f"Low disk space: {free_space:.1f}GB available, {required_space}GB recommended")
            if not self.config.get('auto_continue', False):
                response = input("Continue anyway? (y/n): ")
                if response.lower() != 'y':
                    return False
        
        logger.info("✅ Prerequisites check passed!")
        return True
    
    def run_setup(self):
        """Run the complete setup process."""
        logger.info("🚀 Starting Fine-tuned Qwen2.5-VL LoRA to Ollama Setup")
        logger.info(f"Configuration: {self.config}")
        
        # Step 0: Memory analysis (if not skipped)
        if not self.run_memory_analysis():
            return False
        
        if not self.check_prerequisites():
            return False
        
        try:
            # Step 1: Download base model
            self.log_step(
                "Download Base Model",
                "Downloading unsloth/qwen2.5-vl-7b-instruct-bnb-4bit from Hugging Face"
            )
            
            download_args = [
                "--output-dir", self.config['base_model_path']
            ]
            if self.config.get('hf_token'):
                download_args.extend(["--auth-token", self.config['hf_token']])
            if self.config.get('skip_requirements'):
                download_args.append("--skip-requirements-check")
                
            if not self.run_script("01_download_base_model.py", download_args):
                logger.error("Failed to download base model")
                return False
            
            # Check if we should use direct inference mode
            if self.config.get('use_direct_inference', False):
                logger.info("Using direct inference mode - skipping merge and conversion steps")
                
                # Test direct inference
                self.log_step(
                    "Test Direct LoRA Inference",
                    "Testing LoRA adapter with direct inference (no merge required)"
                )
                
                inference_args = [
                    "--base-model-path", self.config['base_model_path'],
                    "--lora-path", self.config['lora_path'],
                    "--test-model"
                ]
                if self.config.get('use_cpu'):
                    inference_args.append("--use-cpu")
                
                if not self.run_script("06_direct_lora_inference.py", inference_args):
                    logger.error("Direct inference test failed")
                    return False
                
                # Success message for direct inference
                total_time = time.time() - self.start_time
                logger.info("=" * 80)
                logger.info("🎉 DIRECT INFERENCE SETUP COMPLETED SUCCESSFULLY!")
                logger.info("=" * 80)
                logger.info(f"Total setup time: {total_time/60:.1f} minutes")
                logger.info(f"LoRA model ready for direct inference")
                logger.info("")
                logger.info("Usage:")
                logger.info(f"python 06_direct_lora_inference.py --base-model-path '{self.config['base_model_path']}' --lora-path '{self.config['lora_path']}'")
                logger.info("")
                logger.info("For interactive chat:")
                logger.info("python 06_direct_lora_inference.py")
                logger.info("")
                logger.info("For single prompt:")
                logger.info("python 06_direct_lora_inference.py --prompt 'Your question here'")
                logger.info("=" * 80)
                return True
            
            # Step 2: Merge LoRA (with memory-efficient option)
            merge_script = "02_merge_lora_memory_efficient.py" if self.config.get('use_memory_efficient_merge', False) else "02_merge_lora.py"
            
            self.log_step(
                "Merge LoRA Adapter", 
                f"Merging your fine-tuned LoRA weights with the base model using {merge_script}"
            )
            
            merge_args = [
                "--base-model-path", self.config['base_model_path'],
                "--lora-path", self.config['lora_path'],
                "--output-path", self.config['merged_model_path']
            ]
            if self.config.get('use_cpu'):
                merge_args.append("--use-cpu")
            if self.config.get('skip_requirements'):
                merge_args.append("--skip-requirements-check")
            if self.config.get('force_merge', False) and merge_script == "02_merge_lora_memory_efficient.py":
                merge_args.append("--force-merge")
                
            if not self.run_script(merge_script, merge_args):
                logger.error(f"Failed to merge LoRA adapter with {merge_script}")
                if merge_script == "02_merge_lora.py":
                    logger.info("Trying memory-efficient merge as fallback...")
                    if not self.run_script("02_merge_lora_memory_efficient.py", merge_args + ["--force-merge"]):
                        logger.error("Memory-efficient merge also failed")
                        logger.info("Consider using direct inference mode instead:")
                        logger.info("python 06_direct_lora_inference.py")
                        return False
                else:
                    logger.info("Consider using direct inference mode instead:")
                    logger.info("python 06_direct_lora_inference.py")
                    return False
            
            # Step 3: Convert to GGUF
            self.log_step(
                "Convert to GGUF",
                f"Converting merged model to GGUF format with {self.config['quantization']} quantization"
            )
            
            convert_args = [
                "--merged-model-path", self.config['merged_model_path'],
                "--output-path", self.config['gguf_path'],
                "--quantization", self.config['quantization']
            ]
            if self.config.get('llama_cpp_path'):
                convert_args.extend(["--llama-cpp-path", self.config['llama_cpp_path']])
            if self.config.get('skip_requirements'):
                convert_args.append("--skip-requirements-check")
                
            if not self.run_script("03_convert_to_gguf.py", convert_args):
                logger.error("Failed to convert to GGUF")
                return False
            
            # Step 4: Create Modelfile
            self.log_step(
                "Create Ollama Modelfile",
                f"Creating Ollama Modelfile and import scripts for model '{self.config['model_name']}'"
            )
            
            modelfile_args = [
                "--gguf-dir", self.config['gguf_path'],
                "--model-name", self.config['model_name'],
                "--output-dir", self.config['ollama_output_path']
            ]
            if self.config.get('auto_import'):
                modelfile_args.append("--auto-import")
                
            if not self.run_script("04_create_modelfile.py", modelfile_args):
                logger.error("Failed to create Modelfile")
                return False
            
            # Step 5: Test Model
            if not self.config.get('skip_tests'):
                self.log_step(
                    "Test Model",
                    "Running comprehensive tests to verify the model works correctly"
                )
                
                test_args = [
                    "--model-name", self.config['model_name']
                ]
                if self.config.get('test_images'):
                    test_args.extend(["--test-images"] + self.config['test_images'])
                
                if not self.run_script("05_test_model.py", test_args):
                    logger.warning("Model tests failed, but setup may still be successful")
                    logger.info("You can run tests manually later with: python 05_test_model.py")
            
            # Success!
            total_time = time.time() - self.start_time
            logger.info("=" * 80)
            logger.info("🎉 SETUP COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            logger.info(f"Total setup time: {total_time/60:.1f} minutes")
            logger.info(f"Model name: {self.config['model_name']}")
            logger.info("")
            logger.info("Next steps:")
            logger.info(f"1. Start using your model: ollama run {self.config['model_name']}")
            logger.info("2. Test with images: ollama run <model> 'analyze this cardboard' --image image.jpg")
            logger.info("3. Integrate into applications using Ollama API")
            logger.info("")
            logger.info("Files created:")
            logger.info(f"- Base model: {self.config['base_model_path']}")
            logger.info(f"- Merged model: {self.config['merged_model_path']}")
            logger.info(f"- GGUF model: {self.config['gguf_path']}")
            logger.info(f"- Ollama files: {self.config['ollama_output_path']}")
            logger.info("=" * 80)
            
            return True
            
        except KeyboardInterrupt:
            logger.info("\nSetup interrupted by user")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during setup: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

def main():
    parser = argparse.ArgumentParser(description="Complete LoRA to Ollama setup orchestrator")
    
    # Path arguments
    parser.add_argument("--lora-path", type=str, default="./lora_model", help="Path to LoRA adapter")
    parser.add_argument("--base-model-path", type=str, default="./base_model", help="Path for base model")
    parser.add_argument("--merged-model-path", type=str, default="./merged_model", help="Path for merged model")
    parser.add_argument("--gguf-path", type=str, default="./gguf_model", help="Path for GGUF model")
    parser.add_argument("--ollama-output-path", type=str, default="./ollama_model", help="Path for Ollama files")
    parser.add_argument("--llama-cpp-path", type=str, help="Path to llama.cpp directory")
    
    # Model configuration
    parser.add_argument("--model-name", type=str, default="qwen2.5vl-cardboard-qc", help="Ollama model name")
    parser.add_argument("--quantization", type=str, default="Q4_K_M", 
                       choices=["f16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_0", "Q2_K"],
                       help="Quantization level")
    
    # Authentication and options
    parser.add_argument("--hf-token", type=str, help="Hugging Face authentication token")
    parser.add_argument("--use-cpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--auto-import", action="store_true", help="Auto-import model into Ollama")
    parser.add_argument("--skip-tests", action="store_true", help="Skip model testing")
    parser.add_argument("--skip-requirements", action="store_true", help="Skip requirements checks")
    parser.add_argument("--test-images", nargs="+", help="Test images for validation")
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout per script (seconds)")
    
    # Memory-aware options
    parser.add_argument("--skip-memory-check", action="store_true", help="Skip automatic memory analysis")
    parser.add_argument("--use-direct-inference", action="store_true", help="Use direct LoRA inference instead of merge")
    parser.add_argument("--use-memory-efficient-merge", action="store_true", help="Use memory-efficient merge script")
    parser.add_argument("--force-merge", action="store_true", help="Force merge even with low memory warnings")
    parser.add_argument("--auto-continue", action="store_true", help="Automatically continue without user prompts")
    
    # Dry run option
    parser.add_argument("--dry-run", action="store_true", help="Show configuration without running")
    
    args = parser.parse_args()
    
    # Convert paths to absolute
    config = {
        'lora_path': os.path.abspath(args.lora_path),
        'base_model_path': os.path.abspath(args.base_model_path),
        'merged_model_path': os.path.abspath(args.merged_model_path),
        'gguf_path': os.path.abspath(args.gguf_path),
        'ollama_output_path': os.path.abspath(args.ollama_output_path),
        'llama_cpp_path': os.path.abspath(args.llama_cpp_path) if args.llama_cpp_path else None,
        'model_name': args.model_name,
        'quantization': args.quantization,
        'hf_token': args.hf_token,
        'use_cpu': args.use_cpu,
        'auto_import': args.auto_import,
        'skip_tests': args.skip_tests,
        'skip_requirements': args.skip_requirements,
        'test_images': args.test_images,
        'timeout': args.timeout,
        # Memory-aware options
        'skip_memory_check': args.skip_memory_check,
        'use_direct_inference': args.use_direct_inference,
        'use_memory_efficient_merge': args.use_memory_efficient_merge,
        'force_merge': args.force_merge,
        'auto_continue': args.auto_continue
    }
    
    if args.dry_run:
        logger.info("Dry run - Configuration:")
        for key, value in config.items():
            logger.info(f"  {key}: {value}")
        logger.info("\nTo run setup: python run_full_setup.py (without --dry-run)")
        return
    
    # Run setup
    orchestrator = SetupOrchestrator(config)
    success = orchestrator.run_setup()
    
    if not success:
        logger.error("❌ Setup failed!")
        logger.info("Check the logs above for details.")
        logger.info("You can also run individual scripts manually if needed.")
        sys.exit(1)

if __name__ == "__main__":
    main()