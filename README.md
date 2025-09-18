# ADSYS Cardboard AI Detection

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![YOLO](https://img.shields.io/badge/YOLO-v11-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.12-red.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

AI-powered quality control system for detecting cardboard defects including warping, damage, and pallet issues using YOLO11 object detection with a user-friendly GUI interface.

## 🚀 Features

- **Real-time Detection**: YOLO11-based cardboard quality detection
- **Multiple Input Sources**: Support for images, videos, and webcam feeds
- **GPU Acceleration**: Optimized for CUDA-enabled GPUs (RTX 3060+)
- **User-friendly GUI**: Tkinter-based interface with real-time controls
- **Performance Monitoring**: Live FPS and GPU utilization display
- **Video Recording**: Save detection results to MP4 files
- **Flexible Configuration**: Adjustable confidence and IoU thresholds

## 📋 Requirements

### System Requirements
- **OS**: Windows 10/11, macOS, or Linux
- **Python**: 3.8 or higher
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB free space

### Hardware Requirements
- **GPU**: RTX 3060 or better for optimal performance
- **CPU**: Intel i5/AMD Ryzen 5 or equivalent
- **Webcam**: Any USB camera for real-time detection

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/CongcongFan/ADSYS-Cardboard-AI-Detection.git
cd ADSYS-Cardboard-AI-Detection
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Model Weights
The repository includes the folder structure for model weights, but the actual model files must be downloaded separately due to GitHub's file size limitations.

**Model Location**: `Versions/V2 Nanobanana Sythetic cardboard bundle dataset/models/weights/`

#### Required Model File:
- **Primary Model**: `model_- 6 september 2025 11_35.pt` (~114 MB)
  - Place this file in the weights directory
  - The GUI application will auto-detect it at startup

#### Optional Files:
- `model_artifacts.json` - Model metadata
- `roboflow_deploy.zip` - Roboflow deployment package
- `state_dict.pt` - PyTorch state dictionary

**Download Instructions**:
1. Contact the project maintainer for trained model access
2. Download the model files to your local machine
3. Copy `model_- 6 september 2025 11_35.pt` to the `weights/` directory
4. The GUI will automatically detect and load the model

**Note**: The `weights/` directory is included in the repository with documentation, but actual model files are excluded due to size constraints.

## 🖥️ Usage

### Running the YOLO GUI Application

Navigate to the application directory and run:

```bash
python "Versions/V2 Nanobanana Sythetic cardboard bundle dataset/Inference/yolo_gui_app.py"
```

Or use the full path:
```bash
python "C:\Users\[YourUsername]\Desktop\ADSYS-Cardboard-AI-Detection\Versions\V2 Nanobanana Sythetic cardboard bundle dataset\Inference\yolo_gui_app.py"
```

### GUI Interface Guide

#### 1. **Model Configuration**
- **Model Path**: Select your trained YOLO model (.pt file)
- **Device**: Choose between CPU or CUDA (GPU)
- **Auto-load**: Model loads automatically on startup

#### 2. **Input Source Selection**
- **Image**: Process single images
- **Video**: Process video files with recording option
- **Webcam**: Real-time detection from camera feed

#### 3. **Detection Parameters**
- **Confidence Threshold**: 0.0-1.0 (default: 0.25)
- **IoU Threshold**: 0.0-1.0 (default: 0.20)
- **Frame Skip**: Optimize performance by skipping frames

#### 4. **Output Settings**
- **Save Path**: Directory for saving results
- **Auto-save Video**: Automatically record video processing results

### Example Usage Scenarios

#### 1. **Image Processing**
1. Select "Image" as source type
2. Browse and select your cardboard image
3. Click "Load Model" if not auto-loaded
4. Click "Process Image"
5. View results with bounding boxes and confidence scores

#### 2. **Video Processing**
1. Select "Video" as source type
2. Browse and select your video file
3. Enable "Auto-save video" if you want to record results
4. Click "Start Detection"
5. Use "Start/Stop Recording" for manual control

#### 3. **Real-time Webcam Detection**
1. Select "Webcam" as source type
2. Set webcam index (usually 0 for default camera)
3. Click "Start Detection"
4. Real-time detection with live FPS display

## 📊 Performance Optimization

### GPU Settings
- Ensure CUDA is properly installed
- Set device to "cuda" for GPU acceleration
- Monitor GPU utilization in the performance panel

### Performance Tuning
- **Frame Skip**: Use values 2-5 for better performance
- **Confidence Threshold**: Higher values (0.3-0.5) for cleaner results
- **Resolution**: Lower input resolution for faster processing

### Expected Performance
- **RTX 3060 Laptop**: ~15-25 FPS (1080p video)
- **RTX 4070**: ~30-45 FPS (1080p video)
- **CPU Only**: ~2-5 FPS (not recommended for real-time)

## 🗂️ Project Structure

```
ADSYS-Cardboard-AI-Detection/
├── Versions/
│   └── V2 Nanobanana Sythetic cardboard bundle dataset/
│       └── Inference/
│           └── yolo_gui_app.py          # Main GUI application
├── Claude-Code-App/                     # Alternative applications
├── Test photos/                         # Sample test images
├── trash/                              # Default output directory
├── requirements.txt                     # Python dependencies
├── CLAUDE.md                           # Development documentation
└── README.md                           # This file
```

## 🔧 Configuration

### Default Paths (can be customized in GUI):
- **Model Path**: `Versions/V2 Nanobanana Sythetic cardboard bundle dataset/models/weights/model_- 6 september 2025 11_35.pt`
- **Save Directory**: `trash/`
- **Image/Video Browser**: Opens to appropriate directories

### Environment Variables:
- `CUDA_VISIBLE_DEVICES`: Set specific GPU for multi-GPU systems
- `TORCH_HOME`: Custom PyTorch model cache location

## 🐛 Troubleshooting

### Common Issues

#### 1. **Model Loading Errors**
```
Error: Model file not found
```
**Solution**: Ensure model file exists at the specified path or browse to select correct model.

#### 2. **CUDA Out of Memory**
```
RuntimeError: CUDA out of memory
```
**Solution**:
- Use CPU device instead of CUDA
- Lower input resolution
- Increase frame skip value

#### 3. **Camera Access Issues**
```
Error: Cannot access camera
```
**Solution**:
- Check camera permissions
- Try different webcam index (0, 1, 2...)
- Ensure camera is not used by other applications

#### 4. **Low FPS Performance**
**Solutions**:
- Enable GPU acceleration (CUDA)
- Increase frame skip value
- Lower detection thresholds
- Close other GPU-intensive applications

### Performance Tips
- **GPU Memory**: Close other applications using GPU memory
- **CPU Usage**: Monitor CPU usage in Task Manager
- **Frame Skip**: Use frame skip for real-time applications
- **Resolution**: Process smaller images/videos for better performance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **YOLO**: Ultralytics YOLO for object detection
- **OpenCV**: Computer vision library
- **PyTorch**: Deep learning framework
- **Tkinter**: GUI framework

## 📞 Support

For questions, issues, or contributions:
- **GitHub Issues**: [Create an Issue](https://github.com/CongcongFan/ADSYS-Cardboard-AI-Detection/issues)
- **Documentation**: Check `CLAUDE.md` for development details

---

**Note**: This project is designed for cardboard quality control in industrial settings. Ensure proper hardware setup for optimal performance.