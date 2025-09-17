# Qwen2.5VL Cardboard QC Finetuning Workflow

## Overview

This workflow converts your YOLO dataset into a labeled training dataset for finetuning Qwen2.5VL:7b to classify cardboard bundles as **FLAT** (Pass) or **WARPED** (Fail).

## Dataset Status
- **Original dataset**: 128 training images + 8 validation + 12 test = **148 total images**
- **Bbox overlay images**: ✅ Already generated in `out/overlays_overlay/`
- **Labels**: 📝 Ready for workers to label (0/168 completed)

## Complete Workflow

### Step 1: Label Images (Local)
```cmd
cd "C:\Users\76135\Desktop\ADSYS-Cardboard-AI-Detection\Yolo DS To Qwen DS"

# Start labeling tool
python label_helper.py

# Check progress anytime
python check_labeling_progress.py
```

**For Workers**: See `LABELING_GUIDE.md` for detailed instructions

### Step 2: Generate Training Files (Local)
Once all images are labeled:
```cmd
# Run the training data generator
generate_training_data.bat

# Or manually:
python coco_to_qwen.py --coco train/_annotations.coco.json --images train --out out --only_class bundle --mode overlay --resize 640 --train_split 0.8
```

This creates:
- `out/train.jsonl` - Training data
- `out/eval.jsonl` - Validation data

### Step 3: Upload to Colab
Upload these files to your Colab environment:
```
out/
├── train.jsonl          # Training samples
├── eval.jsonl           # Validation samples  
└── overlays_overlay/    # All bbox overlay images
```

### Step 4: Finetune with Unsloth (Colab)
Use Unsloth notebook with Qwen2.5VL:7b model:

```python
# Key settings for your use case:
model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
max_seq_length = 2048
dtype = torch.float16
load_in_4bit = True

# LoRA config
r = 16
target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
lora_alpha = 16
lora_dropout = 0.1

# Training params
batch_size = 2
learning_rate = 2e-4
num_train_epochs = 3
```

## File Structure
```
Yolo DS To Qwen DS/
├── train/                      # Original YOLO training images
├── test/                       # Original YOLO test images  
├── valid/                      # Original YOLO validation images
├── out/
│   ├── overlays_overlay/       # Bbox overlay images (for labeling)
│   ├── qc_labels.csv          # Manual labels (FLAT/WARPED)
│   ├── train.jsonl            # Generated training data
│   └── eval.jsonl             # Generated validation data
├── coco_to_qwen.py            # Dataset converter
├── label_helper.py            # GUI labeling tool
├── check_labeling_progress.py # Progress checker
├── generate_training_data.bat # Batch script
├── LABELING_GUIDE.md          # Worker instructions
└── README_WORKFLOW.md         # This file
```

## Training Data Format
The JSONL files contain conversations in this format:
```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text", 
          "text": "Inspect the attached pallet cardboard bundle. Decide: Pass (flat) or Fail (warped). Respond exactly as 'Pass:' or 'Fail:' plus a short reason."
        },
        {
          "type": "image",
          "image": "file:///path/to/overlay_image.jpg"
        }
      ]
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "text",
          "text": "Pass: Bundle appears flat"
        }
      ]
    }
  ]
}
```

## Key Parameters

### Image Processing
- **Input size**: 640x640 (optimal for Qwen2.5VL)
- **Format**: JPEG bbox overlay images
- **Augmentation**: Handled by YOLO dataset (rotations, crops, etc.)

### Model Settings
- **Base model**: Qwen2.5-VL-7B-Instruct
- **Finetuning**: LoRA (Low-Rank Adaptation)
- **Target**: Vision encoder + language projection layers
- **GPU**: T4 (Colab) - sufficient for LoRA training

### Training Split
- **Train**: 80% (~110 samples)
- **Validation**: 20% (~30 samples)
- **Recommended minimum**: 50 samples total (25 Pass, 25 Fail)

## Quality Assurance

### During Labeling
- Clear instructions for workers
- Progress tracking with `check_labeling_progress.py`
- Consistent labeling criteria (FLAT vs WARPED)

### Before Training  
- Balanced dataset (similar PASS/FAIL counts)
- No missing labels
- Image quality check (overlays visible)

### After Training
- Validate on original cardboard production app
- Test edge cases (borderline warped bundles)
- Monitor false positive/negative rates

## Troubleshooting

### Labeling Issues
- **GUI doesn't start**: Fallback to text mode automatically
- **Images not found**: Check `out/overlays_overlay/` directory
- **Slow labeling**: Use keyboard shortcuts (P/F keys)

### Training Issues
- **Not enough data**: Minimum 20 samples per class
- **Imbalanced classes**: Aim for 40-60% Pass rate
- **Poor performance**: Consider more training epochs or lower learning rate

## Next Steps After Training
1. Save the finetuned model
2. Test on new cardboard images
3. Integrate back into production app
4. Monitor real-world performance
5. Collect additional training data as needed

---

**Estimated Timeline**:
- Labeling: 2-3 hours (168 images)
- Training setup: 30 minutes  
- Finetuning: 1-2 hours (T4 GPU)
- Integration: 1 hour