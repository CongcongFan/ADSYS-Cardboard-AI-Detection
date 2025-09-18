# Model Weights Directory

This directory contains the trained YOLO model weights for cardboard quality detection.

## Required Files

Place your trained model files in this directory:

### Primary Model (Default)
- **`model_- 6 september 2025 11_35.pt`** - Main YOLO11 model for cardboard detection
  - Size: ~114 MB
  - Format: PyTorch (.pt)
  - Used by default in the GUI application

### Additional Files (Optional)
- **`model_artifacts.json`** - Model metadata and configuration
- **`roboflow_deploy.zip`** - Roboflow deployment package
- **`state_dict.pt`** - PyTorch state dictionary

## Model Information

### Training Details
- **Architecture**: YOLO11 (Ultralytics)
- **Dataset**: Synthetic cardboard bundle dataset
- **Classes**: Cardboard defects, warping, pallet issues
- **Input Size**: 640x640 pixels
- **Framework**: PyTorch

### Performance Expectations
- **Accuracy**: Optimized for cardboard quality control
- **Speed**: 15-25 FPS on RTX 3060 laptop
- **Detection**: Warping, damage, pallet positioning

## Download Instructions

1. **Obtain Model Files**: Contact the project maintainer for access to trained models
2. **Place Files**: Copy model files directly into this `weights/` directory
3. **Verify Path**: Ensure the main model file is named `model_- 6 september 2025 11_35.pt`
4. **GUI Configuration**: The application will auto-detect the model in this location

## File Structure
```
weights/
├── README.md                           # This file
├── model_- 6 september 2025 11_35.pt   # Primary model (download required)
├── model_artifacts.json                # Model metadata (optional)
├── roboflow_deploy.zip                 # Roboflow package (optional)
└── state_dict.pt                       # State dictionary (optional)
```

## Usage in Application

The GUI application automatically looks for models in this directory:
- **Default Path**: `Versions/V2 Nanobanana Sythetic cardboard bundle dataset/models/weights/`
- **Auto-Detection**: GUI will find and load `model_- 6 september 2025 11_35.pt`
- **Custom Models**: Use the "Browse" button in GUI to select alternative models

## Model Requirements

- **File Format**: PyTorch (.pt) format
- **YOLO Version**: Compatible with Ultralytics YOLO11
- **Size**: Typically 100-150 MB for full models
- **GPU Memory**: Requires 2-4 GB VRAM for inference

## Notes

- Model files are excluded from git due to size (>100MB GitHub limit)
- Contact project maintainer for trained model access
- Custom models can be placed here and selected via GUI
- Ensure proper CUDA/PyTorch installation for GPU acceleration

---

**Important**: This directory structure is preserved in git, but actual model files must be downloaded separately due to file size limitations.