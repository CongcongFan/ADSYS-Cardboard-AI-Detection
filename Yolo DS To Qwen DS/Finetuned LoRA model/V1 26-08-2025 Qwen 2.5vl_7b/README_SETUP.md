# Fine-tuned Qwen2.5-VL LoRA to Ollama Setup Guide

This comprehensive guide will help you set up your fine-tuned LoRA adapter with Ollama for cardboard quality control.

## 📋 Prerequisites

### System Requirements
- **RAM**: 16GB+ recommended (minimum 12GB)
- **Storage**: 50GB+ free space
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional but recommended)
- **OS**: Windows 10/11, Linux, or macOS

### Software Requirements
- Python 3.8+ (3.9+ recommended)
- Git
- Ollama (latest version)
- CUDA 11.8+ (if using GPU)

## 🚀 Quick Start

### Step 1: Environment Setup
```bash
# Clone or navigate to your project directory
cd "C:\Users\76135\Desktop\ADSYS-Cardboard-AI-Detection\Yolo DS To Qwen DS\Finetuned LoRA model\V1 26-08-2025 Qwen 2.5vl_7b"

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### Step 2: Install Ollama
```bash
# Windows: Download from https://ollama.ai
# Linux:
curl -fsSL https://ollama.ai/install.sh | sh
# macOS:
brew install ollama
```

### Step 3: Setup llama.cpp (for GGUF conversion)
```bash
# Clone llama.cpp if not already available
git clone https://github.com/ggerganov/llama.cpp.git

# Build llama.cpp (optional, for quantization)
cd llama.cpp
make

# Return to project directory
cd ..
```

## 📖 Detailed Setup Process

### Phase 1: Download Base Model
```bash
python 01_download_base_model.py --output-dir ./base_model
```

**Options:**
- `--output-dir`: Directory to download model (default: ./base_model)
- `--auth-token`: Hugging Face token if needed
- `--skip-requirements-check`: Skip package verification

**Expected time:** 10-20 minutes
**Disk space:** ~13GB

### Phase 2: Merge LoRA with Base Model
```bash
python 02_merge_lora.py --base-model-path ./base_model --lora-path ./lora_model --output-path ./merged_model
```

**Options:**
- `--base-model-path`: Path to downloaded base model
- `--lora-path`: Path to your LoRA adapter (default: ./lora_model)
- `--output-path`: Where to save merged model (default: ./merged_model)
- `--use-cpu`: Force CPU usage (slower but uses less VRAM)

**Expected time:** 15-30 minutes
**Disk space:** ~13GB additional

### Phase 3: Convert to GGUF Format
```bash
python 03_convert_to_gguf.py --merged-model-path ./merged_model --output-path ./gguf_model --quantization Q4_K_M
```

**Options:**
- `--merged-model-path`: Path to merged model
- `--output-path`: Where to save GGUF files
- `--llama-cpp-path`: Path to llama.cpp (auto-detected)
- `--quantization`: Quantization level (Q4_K_M recommended)

**Quantization Options:**
- `f16`: No quantization (largest, best quality)
- `Q8_0`: 8-bit quantization (good quality, smaller)
- `Q6_K`: 6-bit (balanced)
- `Q5_K_M`: 5-bit medium (good balance)
- `Q4_K_M`: 4-bit medium (recommended - good quality/size)
- `Q4_0`: 4-bit (smaller, lower quality)
- `Q2_K`: 2-bit (smallest, lowest quality)

**Expected time:** 20-45 minutes
**Disk space:** 3-7GB (depending on quantization)

### Phase 4: Create Ollama Modelfile
```bash
python 04_create_modelfile.py --gguf-dir ./gguf_model --model-name qwen2.5vl-cardboard-qc --auto-import
```

**Options:**
- `--gguf-dir`: Directory containing GGUF files
- `--model-name`: Name for the Ollama model
- `--auto-import`: Automatically import into Ollama
- `--output-dir`: Where to save Modelfile and scripts

**Expected time:** 1-5 minutes + import time

### Phase 5: Test the Model
```bash
python 05_test_model.py --model-name qwen2.5vl-cardboard-qc
```

**Options:**
- `--model-name`: Name of the model to test
- `--test-images`: Paths to test images
- `--skip-text-tests`: Skip text-only tests
- `--skip-image-tests`: Skip vision tests

## 🔧 Troubleshooting

### Common Issues

#### 1. "Out of Memory" Errors
**Solutions:**
- Use `--use-cpu` flag in merge step
- Close other applications
- Use a more aggressive quantization (Q4_0 or Q2_K)
- Increase virtual memory/swap

#### 2. "Model not found" in Ollama
**Solutions:**
```bash
# List available models
ollama list

# Re-import model
./import_model.sh  # or import_model.bat on Windows

# Check Ollama is running
ollama serve
```

#### 3. llama.cpp Conversion Fails
**Solutions:**
```bash
# Ensure llama.cpp is properly cloned
git clone https://github.com/ggerganov/llama.cpp.git

# Update llama.cpp
cd llama.cpp && git pull

# Install Python dependencies
pip install gguf numpy

# Verify convert script exists
ls llama.cpp/convert_hf_to_gguf.py
```

#### 4. Slow Performance
**Solutions:**
- Ensure GPU drivers are updated
- Use CUDA if available
- Try different quantization levels
- Check system resources (RAM, CPU usage)

#### 5. "Permission Denied" Errors
**Solutions:**
```bash
# On Linux/Mac, make scripts executable
chmod +x *.sh

# On Windows, run as administrator
```

### Hardware-Specific Recommendations

#### High-End System (32GB+ RAM, RTX 4090)
- Use `f16` or `Q8_0` quantization for best quality
- Enable GPU acceleration for all steps
- Expected total time: 45-90 minutes

#### Mid-Range System (16GB RAM, RTX 3070)
- Use `Q5_K_M` or `Q4_K_M` quantization
- Mix GPU/CPU usage as needed
- Expected total time: 60-120 minutes

#### Lower-End System (8-12GB RAM, no GPU)
- Use `--use-cpu` flag for all steps
- Use `Q4_0` or `Q2_K` quantization
- Expected total time: 2-4 hours

## 🧪 Testing and Validation

### Manual Testing
```bash
# Start Ollama server
ollama serve

# Test basic functionality
ollama run qwen2.5vl-cardboard-qc "What factors determine cardboard quality?"

# Test with image
ollama run qwen2.5vl-cardboard-qc "Analyze this cardboard for defects" --image path/to/cardboard.jpg
```

### Automated Testing
```bash
# Run full test suite
python 05_test_model.py --model-name qwen2.5vl-cardboard-qc

# Test with specific images
python 05_test_model.py --model-name qwen2.5vl-cardboard-qc --test-images image1.jpg image2.jpg
```

## 📊 Expected Results

### Text Generation
The model should provide coherent responses about:
- Cardboard quality factors
- Defect identification
- Quality control recommendations

### Vision Analysis
The model should analyze cardboard images and identify:
- Warping and bending
- Surface damage
- Overall quality ratings
- Specific defect locations

### Performance Metrics
- **Text generation**: 1-3 seconds per response
- **Image analysis**: 10-30 seconds per image
- **Memory usage**: 4-8GB RAM (depending on quantization)
- **Model size**: 3-13GB (depending on quantization)

## 🔄 Model Updates

To update your fine-tuned model:
1. Replace the LoRA adapter in `./lora_model`
2. Re-run steps 2-5
3. Import the updated model with a new name

## 📚 Additional Resources

### Documentation
- [Ollama Documentation](https://ollama.ai/docs)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [PEFT Documentation](https://huggingface.co/docs/peft)

### Support
- Check logs for detailed error messages
- Verify system requirements are met
- Ensure all dependencies are installed
- Test with different quantization levels if performance is poor

## 🎯 Integration Examples

### Python API Usage
```python
import ollama

# Text query
response = ollama.generate(
    model='qwen2.5vl-cardboard-qc',
    prompt='Assess cardboard quality factors'
)
print(response['response'])

# Image analysis
import base64
with open('cardboard.jpg', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode()

response = ollama.generate(
    model='qwen2.5vl-cardboard-qc',
    prompt='Analyze this cardboard for quality issues',
    images=[image_data]
)
print(response['response'])
```

### REST API Usage
```bash
curl http://localhost:11434/api/generate \
  -d '{"model": "qwen2.5vl-cardboard-qc", "prompt": "What makes good quality cardboard?"}'
```

## ⚖️ Model License and Usage

- Base model: Qwen2.5-VL (Apache 2.0 License)
- LoRA adapter: Your fine-tuned weights
- Usage: Intended for cardboard quality control applications
- Commercial use: Check base model license terms

---

🎉 **Congratulations!** You now have a working fine-tuned Qwen2.5-VL model for cardboard quality control integrated with Ollama!

For support or questions, refer to the troubleshooting section or check the generated logs.