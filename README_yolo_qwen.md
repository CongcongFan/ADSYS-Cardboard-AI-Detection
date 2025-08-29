# YOLO11-seg + Qwen2.5VL Integration Script

This script provides real-time object detection and segmentation using YOLO11-seg, combined with AI-powered analysis using Qwen2.5VL.

## Features

✅ **Multiple Input Sources:**
- Image files
- Video files  
- Webcam
- DroidCam (phone over WiFi)

✅ **Real-time Visualization:**
- Segmentation masks
- Bounding boxes
- Class names and confidence scores
- FPS and inference timing display

✅ **Async Processing:**
- YOLO runs independently (150ms typical)
- Qwen2.5VL runs asynchronously (2000-3000ms typical)
- No blocking between YOLO and Qwen

✅ **Smart Display:**
- Qwen decision text with background color changes on update
- FPS and timing info in top-right corner
- Clean overlay of all information
- **Debug Window**: Shows exactly what image is sent to Qwen (with annotations)

## Requirements

```bash
pip install ultralytics opencv-python numpy pillow
pip install ollama
```

**Note**: Make sure you have the latest version of the Ollama Python client:
```bash
pip install --upgrade ollama
```

**Important**: The script automatically sets the `OLLAMA_HOST` environment variable to connect to your remote Ubuntu Ollama server.

## Usage Examples

### 1. Process a Single Image
```bash
python yolo_qwen_simple.py --source image --path ./test_img/IMG_5497.JPG
```

### 2. Process a Video File
```bash
python yolo_qwen_simple.py --source video --path ./path/to/video.mp4
```

### 3. Use Webcam
```bash
python yolo_qwen_simple.py --source webcam --cam-index 0
```

### 4. Use DroidCam (Phone Camera)
```bash
python yolo_qwen_simple.py --source droidcam --droidcam-ip 192.168.0.66 --droidcam-port 4747
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--source` | Input source type (image/video/webcam/droidcam) | Required |
| `--path` | Path to image/video file | "" |
| `--cam-index` | Webcam device index | 0 |
| `--droidcam-ip` | DroidCam IP address | 192.168.0.66 |
| `--droidcam-port` | DroidCam port | 4747 |
| `--model` | YOLO model path | yolo11l-seg.pt |
| `--ollama-host` | Ollama host URL | http://192.168.0.87:11434 |
| `--qwen-model` | Qwen model name | qwen2.5vl:7b |

## YOLO Configuration

- **Input Size**: Fixed at 640 pixels (as requested)
- **IoU Threshold**: 0.20 (20%) to avoid duplicate bounding boxes
- **Confidence Threshold**: 0.25 (25%)
- **Model**: Uses yolo11l-seg.pt for segmentation

## Qwen2.5VL Integration

- **Prompt**: Analyzes cardboard in green bounding boxes
- **Output**: JSON format with warp detection and quality assessment
- **Async Processing**: Runs independently from YOLO inference
- **Background Color**: Changes to green when response updates
- **Ollama Client**: Uses official Python client for reliable communication

## Testing

Before running the full script, test basic YOLO functionality:

```bash
python test_basic.py
```

This will load a test image and verify YOLO11-seg works correctly.

## Controls

- **'q' key**: Quit the application
- **'d' key**: Toggle debug window visibility (video/webcam mode)
- **Any key**: Close image window (in image mode)

## Performance Notes

- **YOLO**: ~150ms per frame
- **Qwen**: ~2000-3000ms per analysis
- **FPS**: Displayed in real-time
- **Memory**: Efficient async processing prevents blocking

## Troubleshooting

1. **Model not found**: Ensure `yolo11l-seg.pt` is in the current directory
2. **DroidCam connection**: Verify phone and computer are on same WiFi network
3. **Qwen API**: Ensure Ollama is running with qwen2.5vl:7b model on the specified host
4. **Camera access**: Check camera permissions and device availability
5. **Debug window**: Use the debug window to see exactly what image is sent to Qwen
6. **Ollama connection**: Verify network connectivity to the Ollama host (default: 192.168.0.87:11434)

### Testing Ollama Connection

Before running the main script, test the Ollama connection:

```bash
python test_ollama.py
```

This will verify that your Windows machine can connect to the Ubuntu Ollama server.

## Next Steps

This is the simple version. Future enhancements can include:
- Toggle controls for different visualization modes
- Recording capabilities
- Batch processing
- Advanced filtering options
- Custom prompt templates
