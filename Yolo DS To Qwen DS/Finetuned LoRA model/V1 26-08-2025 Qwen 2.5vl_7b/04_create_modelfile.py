#!/usr/bin/env python3
"""
Script to create an Ollama Modelfile for the converted GGUF model.
This sets up the model with appropriate parameters for cardboard quality control.
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
import subprocess
import glob

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_gguf_file(gguf_dir):
    """Find the GGUF model file in the directory."""
    gguf_pattern = os.path.join(gguf_dir, "*.gguf")
    gguf_files = glob.glob(gguf_pattern)
    
    if not gguf_files:
        logger.error(f"No GGUF files found in {gguf_dir}")
        return None
    
    if len(gguf_files) > 1:
        logger.info(f"Multiple GGUF files found: {gguf_files}")
        # Prefer quantized models over fp16
        for gguf_file in gguf_files:
            if "q4_k_m" in gguf_file.lower() or "q5" in gguf_file.lower():
                logger.info(f"Selected quantized model: {gguf_file}")
                return gguf_file
        # If no quantized, use the first one
        logger.info(f"Using first available: {gguf_files[0]}")
        return gguf_files[0]
    
    logger.info(f"Found GGUF file: {gguf_files[0]}")
    return gguf_files[0]

def load_model_metadata(gguf_dir):
    """Load model metadata if available."""
    metadata_path = os.path.join(gguf_dir, "model_info.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except:
            logger.warning("Could not load model metadata")
    return {}

def check_ollama_availability():
    """Check if Ollama is installed and available."""
    try:
        result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"Ollama found: {result.stdout.strip()}")
            return True
        else:
            logger.warning("Ollama command failed")
            return False
    except FileNotFoundError:
        logger.warning("Ollama not found in PATH")
        return False

def create_modelfile(gguf_file_path, output_dir, model_name="qwen2.5vl-cardboard-qc"):
    """Create Ollama Modelfile for the GGUF model."""
    logger.info("Creating Ollama Modelfile...")
    
    # Load metadata
    metadata = load_model_metadata(os.path.dirname(gguf_file_path))
    
    # Create Modelfile content
    modelfile_content = f"""# Ollama Modelfile for Fine-tuned Qwen2.5-VL Cardboard Quality Control Model
# Created by: ollama-lora-setup-script
# Base model: unsloth/qwen2.5-vl-7b-instruct-bnb-4bit
# Fine-tuned for: Cardboard Quality Control

FROM {gguf_file_path}

# Template for Qwen2.5-VL chat format
TEMPLATE \"\"\"
<|im_start|>system
You are a specialized AI assistant for cardboard quality control analysis. You have been fine-tuned to analyze cardboard images and assess their quality based on various factors such as warping, damage, surface quality, and overall condition.

When analyzing images:
1. Examine the cardboard surface for any visible defects
2. Look for warping, bending, or structural damage
3. Assess overall quality on a scale from 1-10
4. Provide specific observations about any issues found
5. Give clear recommendations for quality control decisions

Be precise, objective, and professional in your assessments.<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
\"\"\"

# System prompt
SYSTEM \"\"\"You are a specialized AI assistant for cardboard quality control analysis. You have been fine-tuned to analyze cardboard images and assess their quality based on various factors such as warping, damage, surface quality, and overall condition.

When analyzing images:
1. Examine the cardboard surface for any visible defects
2. Look for warping, bending, or structural damage  
3. Assess overall quality on a scale from 1-10
4. Provide specific observations about any issues found
5. Give clear recommendations for quality control decisions

Be precise, objective, and professional in your assessments.\"\"\"

# Model parameters optimized for quality control tasks
PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096
PARAMETER num_predict 512

# Stop tokens
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"

# Additional metadata
PARAMETER num_thread 8
"""
    
    # Add model info if available
    if metadata:
        modelfile_content += f"""
# Model Information
# Quantization: {metadata.get('quantization', 'unknown')}
# Model Size: {metadata.get('file_size_gb', 'unknown')} GB
# Created: {metadata.get('created_with', 'ollama-lora-setup-script')}
"""
    
    # Save Modelfile
    modelfile_path = os.path.join(output_dir, "Modelfile")
    
    try:
        with open(modelfile_path, 'w', encoding='utf-8') as f:
            f.write(modelfile_content)
        logger.info(f"✅ Modelfile created: {modelfile_path}")
        return modelfile_path
    except Exception as e:
        logger.error(f"Failed to create Modelfile: {e}")
        return None

def create_import_script(modelfile_path, model_name, output_dir):
    """Create script to import model into Ollama."""
    import_script_content = f"""#!/bin/bash
# Script to import the fine-tuned model into Ollama

echo "Importing {model_name} into Ollama..."
echo "This may take several minutes..."

cd "{os.path.dirname(modelfile_path)}"

# Create the model in Ollama
ollama create {model_name} -f Modelfile

if [ $? -eq 0 ]; then
    echo "✅ Model imported successfully!"
    echo ""
    echo "You can now use the model with:"
    echo "  ollama run {model_name}"
    echo ""
    echo "Or in your applications:"
    echo "  curl http://localhost:11434/api/generate -d '{{"model": "{model_name}", "prompt": "Analyze this cardboard image for quality issues."}}"'
    echo ""
    echo "Test the model with: python 05_test_model.py --model-name {model_name}"
else
    echo "❌ Failed to import model into Ollama"
    echo "Please check:"
    echo "1. Ollama is running (ollama serve)"
    echo "2. GGUF file exists and is not corrupted"
    echo "3. Sufficient disk space"
    exit 1
fi
"""
    
    # Windows batch version
    import_bat_content = f"""@echo off
REM Script to import the fine-tuned model into Ollama

echo Importing {model_name} into Ollama...
echo This may take several minutes...

cd /d "{os.path.dirname(modelfile_path)}"

REM Create the model in Ollama
ollama create {model_name} -f Modelfile

if %ERRORLEVEL% EQU 0 (
    echo ✅ Model imported successfully!
    echo.
    echo You can now use the model with:
    echo   ollama run {model_name}
    echo.
    echo Or in your applications:
    echo   curl http://localhost:11434/api/generate -d "{{"model": "{model_name}", "prompt": "Analyze this cardboard image for quality issues."}}"
    echo.
    echo Test the model with: python 05_test_model.py --model-name {model_name}
) else (
    echo ❌ Failed to import model into Ollama
    echo Please check:
    echo 1. Ollama is running ^(ollama serve^)
    echo 2. GGUF file exists and is not corrupted
    echo 3. Sufficient disk space
    pause
    exit /b 1
)

pause
"""
    
    # Save shell script
    import_script_path = os.path.join(output_dir, "import_model.sh")
    try:
        with open(import_script_path, 'w', encoding='utf-8') as f:
            f.write(import_script_content)
        # Make executable on Unix systems
        try:
            os.chmod(import_script_path, 0o755)
        except:
            pass
        logger.info(f"✅ Import script created: {import_script_path}")
    except Exception as e:
        logger.error(f"Failed to create import script: {e}")
        return None
    
    # Save batch script for Windows
    import_bat_path = os.path.join(output_dir, "import_model.bat")
    try:
        with open(import_bat_path, 'w', encoding='utf-8') as f:
            f.write(import_bat_content)
        logger.info(f"✅ Import batch script created: {import_bat_path}")
    except Exception as e:
        logger.error(f"Failed to create import batch script: {e}")
    
    return import_script_path

def create_usage_guide(output_dir, model_name):
    """Create a usage guide for the model."""
    usage_content = f"""# {model_name} Usage Guide

## Model Information
- **Model Name**: {model_name}
- **Base Model**: unsloth/qwen2.5-vl-7b-instruct-bnb-4bit
- **Fine-tuned For**: Cardboard Quality Control
- **Vision Capabilities**: Yes (can analyze images)

## Quick Start

### 1. Import Model into Ollama
Run the import script:
```bash
# On Linux/Mac
./import_model.sh

# On Windows
import_model.bat
```

### 2. Basic Usage
```bash
# Interactive chat
ollama run {model_name}

# Single query
ollama run {model_name} "Analyze this cardboard for quality issues"
```

### 3. With Images (Vision)
```bash
# Analyze an image
ollama run {model_name} "What quality issues do you see in this cardboard?" --image path/to/cardboard.jpg
```

### 4. API Usage
```bash
# Text-only request
curl http://localhost:11434/api/generate \\
  -d '{{"model": "{model_name}", "prompt": "Explain cardboard quality factors"}}'

# Image analysis request
curl http://localhost:11434/api/generate \\
  -d '{{"model": "{model_name}", "prompt": "Analyze this cardboard image", "images": ["base64_encoded_image"]}}'
```

### 5. Python Integration
```python
import ollama
import base64

# Text query
response = ollama.generate(
    model='{model_name}',
    prompt='What factors determine cardboard quality?'
)
print(response['response'])

# Image analysis
with open('cardboard.jpg', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode()

response = ollama.generate(
    model='{model_name}',
    prompt='Analyze this cardboard for quality issues',
    images=[image_data]
)
print(response['response'])
```

## Quality Control Prompts

### Basic Analysis
- "Analyze this cardboard image for quality issues"
- "Rate this cardboard quality on a scale of 1-10"
- "What defects do you see in this cardboard?"

### Detailed Assessment
- "Perform a comprehensive quality control analysis of this cardboard, including warping, damage, and surface quality"
- "Assess this cardboard for production acceptance and provide recommendations"
- "Compare this cardboard quality to industry standards"

### Batch Processing
- "Process these cardboard samples and categorize them by quality grade"
- "Identify which of these cardboard pieces should be rejected"

## Model Parameters
The model is configured with these optimal parameters for quality control:
- Temperature: 0.3 (more focused, less creative)
- Top-p: 0.9
- Top-k: 40
- Context length: 4096 tokens
- Max prediction: 512 tokens

## Troubleshooting

### Model Not Found
```bash
# List available models
ollama list

# Re-import if needed
./import_model.sh
```

### Poor Performance
- Ensure Ollama server is running: `ollama serve`
- Check available RAM (8GB+ recommended)
- Consider using a smaller quantization if performance is poor

### Image Analysis Issues
- Ensure image is in supported format (JPEG, PNG)
- Keep image size reasonable (<10MB)
- Use clear, well-lit images for best results

## Testing
Use the provided test script to verify everything works:
```bash
python 05_test_model.py --model-name {model_name}
```

## Support
For issues with:
- Model creation: Check the setup logs
- Ollama integration: Check Ollama documentation
- Performance: Consider hardware requirements
"""
    
    usage_path = os.path.join(output_dir, "USAGE_GUIDE.md")
    try:
        with open(usage_path, 'w', encoding='utf-8') as f:
            f.write(usage_content)
        logger.info(f"✅ Usage guide created: {usage_path}")
        return usage_path
    except Exception as e:
        logger.error(f"Failed to create usage guide: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Create Ollama Modelfile for GGUF model")
    parser.add_argument(
        "--gguf-dir",
        type=str,
        default="./gguf_model",
        help="Directory containing GGUF model (default: ./gguf_model)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./ollama_model",
        help="Directory to save Modelfile and scripts (default: ./ollama_model)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="qwen2.5vl-cardboard-qc",
        help="Name for the Ollama model (default: qwen2.5vl-cardboard-qc)"
    )
    parser.add_argument(
        "--auto-import",
        action="store_true",
        help="Automatically import model into Ollama after creating Modelfile"
    )
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    gguf_dir = os.path.abspath(args.gguf_dir)
    output_dir = os.path.abspath(args.output_dir)
    
    logger.info("=" * 60)
    logger.info("Ollama Modelfile Creation Script")
    logger.info("=" * 60)
    logger.info(f"GGUF directory: {gguf_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Model name: {args.model_name}")
    logger.info("=" * 60)
    
    # Check if GGUF directory exists
    if not os.path.exists(gguf_dir):
        logger.error(f"GGUF directory not found: {gguf_dir}")
        logger.error("Please run 03_convert_to_gguf.py first!")
        sys.exit(1)
    
    # Find GGUF file
    gguf_file = find_gguf_file(gguf_dir)
    if not gguf_file:
        logger.error("No GGUF files found!")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create Modelfile
    modelfile_path = create_modelfile(gguf_file, output_dir, args.model_name)
    if not modelfile_path:
        sys.exit(1)
    
    # Create import scripts
    import_script_path = create_import_script(modelfile_path, args.model_name, output_dir)
    if not import_script_path:
        logger.warning("Failed to create import script, but continuing...")
    
    # Create usage guide
    usage_guide_path = create_usage_guide(output_dir, args.model_name)
    if not usage_guide_path:
        logger.warning("Failed to create usage guide, but continuing...")
    
    # Check Ollama availability
    ollama_available = check_ollama_availability()
    
    if args.auto_import and ollama_available:
        logger.info("Auto-importing model into Ollama...")
        try:
            result = subprocess.run(
                ['ollama', 'create', args.model_name, '-f', 'Modelfile'],
                cwd=output_dir,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )
            
            if result.returncode == 0:
                logger.info("✅ Model imported into Ollama successfully!")
            else:
                logger.error(f"Failed to import model: {result.stderr}")
                logger.info("You can import manually using the provided scripts.")
        
        except subprocess.TimeoutExpired:
            logger.error("Model import timed out. Large models can take time to import.")
            logger.info("You can import manually using the provided scripts.")
        except Exception as e:
            logger.error(f"Error during auto-import: {e}")
            logger.info("You can import manually using the provided scripts.")
    
    logger.info("=" * 60)
    logger.info("✅ OLLAMA MODELFILE CREATION COMPLETED!")
    logger.info("=" * 60)
    logger.info(f"Files created in: {output_dir}")
    logger.info("- Modelfile (Ollama model configuration)")
    logger.info("- import_model.sh/.bat (Import scripts)")
    logger.info("- USAGE_GUIDE.md (Usage documentation)")
    logger.info("")
    logger.info("Next steps:")
    if not args.auto_import:
        logger.info(f"1. Import model: ./import_model.sh (or import_model.bat)")
    logger.info("2. Test model: python 05_test_model.py")
    logger.info("3. Use in applications: ollama run " + args.model_name)
    logger.info("=" * 60)

if __name__ == "__main__":
    main()