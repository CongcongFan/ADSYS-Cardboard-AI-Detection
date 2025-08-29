#!/usr/bin/env python3
"""
Script to convert the merged model to GGUF format for Ollama.
This uses llama.cpp conversion tools to create Ollama-compatible model files.
"""

import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path
import shutil
import tempfile
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_llama_cpp_availability():
    """Check if llama.cpp conversion tools are available."""
    logger.info("Checking for llama.cpp conversion tools...")
    
    # Common locations for llama.cpp
    possible_paths = [
        "./llama.cpp",
        "../../../Claude-Code-App/llama.cpp",
        "C:/llama.cpp",
        "/usr/local/llama.cpp",
        "/opt/llama.cpp"
    ]
    
    llama_cpp_path = None
    
    # Check each possible path
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        convert_script = os.path.join(abs_path, "convert_hf_to_gguf.py")
        if os.path.exists(convert_script):
            llama_cpp_path = abs_path
            logger.info(f"Found llama.cpp at: {abs_path}")
            break
    
    if not llama_cpp_path:
        logger.error("llama.cpp not found in common locations!")
        logger.info("Please:")
        logger.info("1. Clone llama.cpp: git clone https://github.com/ggerganov/llama.cpp.git")
        logger.info("2. Or specify path using --llama-cpp-path argument")
        return None
    
    # Check for required Python script
    convert_script = os.path.join(llama_cpp_path, "convert_hf_to_gguf.py")
    if not os.path.exists(convert_script):
        logger.error(f"convert_hf_to_gguf.py not found at: {convert_script}")
        return None
    
    return llama_cpp_path

def check_python_packages():
    """Check if required Python packages are installed."""
    required_packages = [
        'torch',
        'transformers',
        'numpy',
        'sentencepiece',
        'gguf'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.info("Install missing packages:")
        logger.info("pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_disk_space(output_dir, required_gb=20):
    """Check if there's enough disk space."""
    try:
        statvfs = os.statvfs(output_dir)
        available_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
        
        if available_gb < required_gb:
            logger.error(f"Insufficient disk space. Required: {required_gb}GB, Available: {available_gb:.1f}GB")
            return False
        
        logger.info(f"Disk space check passed. Available: {available_gb:.1f}GB")
        return True
    
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")
        return True

def convert_to_gguf(merged_model_path, output_path, llama_cpp_path, quantization="Q4_K_M"):
    """Convert merged model to GGUF format."""
    logger.info("Starting GGUF conversion...")
    
    try:
        # Verify paths
        if not os.path.exists(merged_model_path):
            logger.error(f"Merged model path does not exist: {merged_model_path}")
            return False
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Check disk space
        if not check_disk_space(output_path):
            return False
        
        # Path to conversion script
        convert_script = os.path.join(llama_cpp_path, "convert_hf_to_gguf.py")
        
        # Temporary GGUF file (unquantized)
        temp_gguf = os.path.join(output_path, "model_fp16.gguf")
        
        # Convert to GGUF (unquantized first)
        logger.info("Converting to unquantized GGUF format...")
        logger.info("This may take 15-30 minutes depending on your hardware...")
        
        convert_cmd = [
            sys.executable,
            convert_script,
            merged_model_path,
            "--outtype", "f16",
            "--outfile", temp_gguf
        ]
        
        logger.info(f"Running command: {' '.join(convert_cmd)}")
        
        result = subprocess.run(
            convert_cmd,
            cwd=llama_cpp_path,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode != 0:
            logger.error("Conversion to GGUF failed!")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            return False
        
        logger.info("✅ Unquantized GGUF conversion completed!")
        
        # Quantize the model if requested
        if quantization and quantization != "f16":
            logger.info(f"Quantizing model to {quantization}...")
            
            # Path to quantization tool
            quantize_tool = None
            possible_quantize_tools = [
                os.path.join(llama_cpp_path, "llama-quantize.exe"),  # Windows
                os.path.join(llama_cpp_path, "llama-quantize"),     # Linux/Mac
                os.path.join(llama_cpp_path, "build", "bin", "llama-quantize.exe"),  # Windows build
                os.path.join(llama_cpp_path, "build", "bin", "llama-quantize"),     # Linux/Mac build
            ]
            
            for tool in possible_quantize_tools:
                if os.path.exists(tool):
                    quantize_tool = tool
                    break
            
            if not quantize_tool:
                logger.warning("Quantization tool not found. Keeping unquantized model.")
                logger.warning("To quantize later, build llama.cpp and use llama-quantize tool.")
                final_gguf = temp_gguf
            else:
                final_gguf = os.path.join(output_path, f"model_{quantization.lower()}.gguf")
                
                quantize_cmd = [
                    quantize_tool,
                    temp_gguf,
                    final_gguf,
                    quantization
                ]
                
                logger.info(f"Running quantization: {' '.join(quantize_cmd)}")
                
                result = subprocess.run(
                    quantize_cmd,
                    capture_output=True,
                    text=True,
                    timeout=1800  # 30 minutes timeout
                )
                
                if result.returncode != 0:
                    logger.error("Quantization failed!")
                    logger.error(f"STDOUT: {result.stdout}")
                    logger.error(f"STDERR: {result.stderr}")
                    logger.warning("Keeping unquantized model.")
                    final_gguf = temp_gguf
                else:
                    logger.info("✅ Quantization completed!")
                    # Remove temporary unquantized file to save space
                    try:
                        os.remove(temp_gguf)
                        logger.info("Removed temporary unquantized file.")
                    except:
                        pass
        else:
            final_gguf = temp_gguf
        
        # Create metadata file
        metadata = {
            "model_name": "qwen2.5-vl-7b-cardboard-qc",
            "base_model": "unsloth/qwen2.5-vl-7b-instruct-bnb-4bit",
            "quantization": quantization if quantization else "f16",
            "use_case": "Cardboard Quality Control",
            "created_with": "ollama-lora-setup-script",
            "file_size_gb": os.path.getsize(final_gguf) / (1024**3)
        }
        
        metadata_path = os.path.join(output_path, "model_info.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model metadata saved to: {metadata_path}")
        logger.info(f"Final GGUF model: {final_gguf}")
        logger.info(f"Model size: {metadata['file_size_gb']:.1f}GB")
        
        return True
        
    except subprocess.TimeoutExpired:
        logger.error("Conversion timed out! This may indicate insufficient memory or very slow hardware.")
        return False
    except Exception as e:
        logger.error(f"Error during GGUF conversion: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    parser = argparse.ArgumentParser(description="Convert merged model to GGUF format for Ollama")
    parser.add_argument(
        "--merged-model-path",
        type=str,
        default="./merged_model",
        help="Path to the merged model (default: ./merged_model)"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./gguf_model",
        help="Path to save GGUF model (default: ./gguf_model)"
    )
    parser.add_argument(
        "--llama-cpp-path",
        type=str,
        help="Path to llama.cpp directory (auto-detected if not specified)"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="Q4_K_M",
        choices=["f16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_0", "Q2_K"],
        help="Quantization level (default: Q4_K_M for good quality/size balance)"
    )
    parser.add_argument(
        "--skip-requirements-check",
        action="store_true",
        help="Skip checking for required packages"
    )
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    merged_model_path = os.path.abspath(args.merged_model_path)
    output_path = os.path.abspath(args.output_path)
    
    logger.info("=" * 60)
    logger.info("GGUF Conversion Script")
    logger.info("=" * 60)
    logger.info(f"Merged model path: {merged_model_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Quantization: {args.quantization}")
    logger.info("=" * 60)
    
    # Check requirements
    if not args.skip_requirements_check:
        logger.info("Checking Python packages...")
        if not check_python_packages():
            sys.exit(1)
    
    # Find or verify llama.cpp path
    if args.llama_cpp_path:
        llama_cpp_path = os.path.abspath(args.llama_cpp_path)
        if not os.path.exists(os.path.join(llama_cpp_path, "convert_hf_to_gguf.py")):
            logger.error(f"Invalid llama.cpp path: {llama_cpp_path}")
            sys.exit(1)
    else:
        llama_cpp_path = check_llama_cpp_availability()
        if not llama_cpp_path:
            sys.exit(1)
    
    logger.info(f"Using llama.cpp at: {llama_cpp_path}")
    
    # Check if merged model exists
    if not os.path.exists(merged_model_path):
        logger.error(f"Merged model not found at: {merged_model_path}")
        logger.error("Please run 02_merge_lora.py first!")
        sys.exit(1)
    
    # Perform conversion
    success = convert_to_gguf(
        merged_model_path,
        output_path,
        llama_cpp_path,
        args.quantization
    )
    
    if success:
        logger.info("=" * 60)
        logger.info("✅ GGUF CONVERSION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"GGUF model location: {output_path}")
        logger.info("Next steps:")
        logger.info("1. Use 04_create_modelfile.py to create Ollama Modelfile")
        logger.info("2. Import model into Ollama")
        logger.info("3. Test with 05_test_model.py")
        logger.info("=" * 60)
    else:
        logger.error("❌ GGUF conversion failed!")
        logger.error("Common issues:")
        logger.error("- Insufficient disk space")
        logger.error("- Insufficient RAM (need 16GB+ for 7B model)")
        logger.error("- Missing llama.cpp or conversion tools")
        logger.error("- Corrupted merged model files")
        sys.exit(1)

if __name__ == "__main__":
    main()