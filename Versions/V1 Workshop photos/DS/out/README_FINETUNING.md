# Qwen2.5-VL Fine-tuning for Cardboard Quality Control

This directory contains all the necessary scripts to fine-tune Qwen2.5-VL (7B) on your cardboard quality control dataset using Unsloth for efficient training.

## 📁 Files Overview

### Dataset Management
- `construct_hf_dataset.py` - Create Hugging Face dataset from images and CSV labels
- `push_to_huggingface.py` - Upload dataset to Hugging Face Hub
- `DATASET_SUMMARY.md` - Dataset documentation and usage guide

### Model Training
- `qwen2_5_vl_cardboard_qc_finetune.py` - **Main fine-tuning script**
- `evaluate_cardboard_model.py` - Model evaluation and metrics
- `cardboard_qc_inference.py` - Production inference script

### Original Reference
- `qwen2_5_vl_(7b)_vision.py` - Original Unsloth notebook (LaTeX OCR example)

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Install required packages
pip install unsloth torch datasets transformers accelerate bitsandbytes

# For Colab users (run in notebook):
# !pip install --no-deps bitsandbytes accelerate xformers peft trl triton
# !pip install sentencepiece protobuf "datasets>=3.4.1,<4.0.0" "huggingface_hub>=0.34.0"
# !pip install --no-deps unsloth
```

### 2. Dataset Setup

Your dataset is already uploaded to Hugging Face Hub as `Cong2612/cardboard-qc-dataset` with:
- **168 total samples** (117 train, 16 validation, 35 test)
- **Image + conversation format** ready for vision-language training
- **Binary classification**: Pass (flat bundles) vs Fail (warped bundles)

### 3. Fine-tuning

```bash
python qwen2_5_vl_cardboard_qc_finetune.py
```

**Key features:**
- ✅ Loads Qwen2.5-VL-7B-Instruct (4-bit quantized)
- ✅ Adds LoRA adapters for efficient training
- ✅ Converts your dataset to proper conversation format
- ✅ Trains for 3 epochs with optimal hyperparameters
- ✅ Tests model before and after training
- ✅ Saves LoRA adapters to `cardboard_qc_lora/`

**Memory requirements:**
- **Minimum**: 8GB VRAM (4-bit quantization)
- **Recommended**: 16GB+ VRAM for larger batch sizes

### 4. Evaluation

```bash
python evaluate_cardboard_model.py
```

This will:
- Load your fine-tuned model
- Test on validation and test sets
- Calculate accuracy and classification metrics
- Save detailed results to JSON files
- Show example predictions and errors

### 5. Production Inference

```bash
# Single image
python cardboard_qc_inference.py --image path/to/cardboard_image.jpg

# Batch processing
python cardboard_qc_inference.py --folder path/to/images/ --output results.json
```

## 📊 Dataset Format

The dataset uses conversation format optimized for vision-language models:

```python
{
  "image": PIL_Image,
  "conversations": [
    {
      "role": "user",
      "content": "Analyze this cardboard bundle image. Is the bundle flat or warped? Provide your assessment."
    },
    {
      "role": "assistant", 
      "content": "The bundle appears flat. Bundle appears flat"
    }
  ],
  "filename": "IMG_5495_JPG.rf.2c365f0e236c8517b608f6f2461f708b.jpg",
  "label": "Pass",
  "reason": "Bundle appears flat"
}
```

## ⚙️ Training Configuration

### Model Architecture
- **Base Model**: `unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit`
- **Vision Layers**: Fine-tuned ✅
- **Language Layers**: Fine-tuned ✅
- **LoRA Rank**: 16
- **LoRA Alpha**: 16

### Training Hyperparameters
- **Epochs**: 3 (configurable)
- **Batch Size**: 2 (per device)
- **Gradient Accumulation**: 4 steps
- **Learning Rate**: 2e-4
- **Optimizer**: AdamW 8-bit
- **Scheduler**: Linear
- **Max Length**: 2048 tokens

### Hardware Recommendations
| GPU | VRAM | Batch Size | Training Time |
|-----|------|------------|---------------|
| RTX 4090 | 24GB | 4 | ~30 min |
| RTX 4080 | 16GB | 2 | ~45 min |
| RTX 4070 | 12GB | 1 | ~60 min |
| T4 (Colab) | 16GB | 2 | ~45 min |

## 📈 Expected Performance

Based on the dataset characteristics:
- **Dataset Size**: 168 samples (small but focused)
- **Task**: Binary classification (Pass/Fail)
- **Expected Accuracy**: 80-90% (depending on data quality)
- **Strengths**: Clear visual differences between flat/warped bundles
- **Challenges**: Limited training data, single camera angle

## 🔧 Customization Options

### 1. Modify Training Parameters

In `qwen2_5_vl_cardboard_qc_finetune.py`:

```python
# Training epochs
num_train_epochs=5,  # Increase for more training

# Learning rate
learning_rate=1e-4,  # Lower for more stable training

# LoRA parameters
r=32,                # Higher rank for more capacity
lora_alpha=32,       # Match the rank
```

### 2. Change Instruction Text

```python
def convert_cardboard_to_conversation(sample):
    # Customize the instruction
    user_instruction = "Assess the quality of this cardboard bundle. Is it suitable for packaging?"
    # ... rest of function
```

### 3. Add Validation During Training

```python
args = SFTConfig(
    # ... other parameters
    evaluation_strategy="steps",
    eval_steps=25,
    eval_dataset=converted_val,  # Add validation dataset
)
```

## 🚀 Advanced Usage

### 1. Multi-GPU Training

```python
# In the training config
per_device_train_batch_size=4,  # Increase batch size
dataloader_num_workers=4,       # Parallel data loading
```

### 2. Weights & Biases Logging

```python
args = SFTConfig(
    # ... other parameters
    report_to="wandb",
    run_name="cardboard_qc_experiment_1",
)
```

### 3. Save Full Model for Production

```python
# After training
model.save_pretrained_merged("cardboard_qc_full_model", tokenizer)
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `per_device_train_batch_size` to 1
   - Enable `use_gradient_checkpointing=True`
   - Use smaller LoRA rank (r=8)

2. **Model Not Loading**
   - Check if `cardboard_qc_lora/` directory exists
   - Verify all required files are present
   - Try loading base model first to test setup

3. **Poor Performance**
   - Increase training epochs
   - Lower learning rate
   - Add more training data
   - Check data quality and labeling consistency

4. **Slow Training**
   - Use 4-bit quantization: `load_in_4bit=True`
   - Enable gradient checkpointing: `use_gradient_checkpointing="unsloth"`
   - Reduce sequence length if possible

### Getting Help

- 📚 **Unsloth Documentation**: https://docs.unsloth.ai/
- 💬 **Discord**: https://discord.gg/unsloth
- 🐛 **Issues**: Check the model outputs and error messages
- 📧 **Dataset Issues**: Review the dataset summary and examples

## 📝 Next Steps

1. **Evaluate Performance**: Run evaluation script and analyze results
2. **Collect More Data**: Add more diverse samples if accuracy is low
3. **Deploy Model**: Use inference script for production quality control
4. **Monitor Performance**: Track model performance over time
5. **Iterate**: Retrain with new data as needed

## 🎯 Production Deployment

For production use:

1. **Save 16-bit Model**: Better performance than LoRA adapters
2. **Batch Processing**: Process multiple images efficiently
3. **API Integration**: Wrap inference script in FastAPI/Flask
4. **Monitoring**: Log predictions and track accuracy
5. **Feedback Loop**: Collect new samples for continuous improvement

---

**Happy Fine-tuning! 🚀**

Your Qwen2.5-VL model will learn to assess cardboard bundle quality just like a human quality control inspector!