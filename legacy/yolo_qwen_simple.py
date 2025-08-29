#!/usr/bin/env python3
"""
Simple YOLO11-seg + Qwen2.5VL script with multiple input sources
- Supports: image, video, webcam, DroidCam
- Shows: masks, boxes, class + confidence, FPS, inference time
- Qwen runs async with YOLO
- Decision text with background color changes on update
"""

import argparse
import base64
import json
import time
import threading
from typing import Optional, Tuple
import cv2
import numpy as np
import ollama
from ultralytics import YOLO
import os



class QwenWorker(threading.Thread):
    """Async worker for Qwen2.5VL inference using Ollama client"""
    def __init__(self, host: str, model: str, prompt: str):
        super().__init__(daemon=True)
        self.host = host
        self.model = model
        self.prompt = prompt
        self._lock = threading.Lock()
        self._latest_frame: Optional[str] = None
        self.result = {
            "text": "Waiting for Qwen...",
            "updated_at": 0.0,
            "processing": False
        }
        
        # Set environment variable for Ollama host
        os.environ['OLLAMA_HOST'] = host

    def submit_frame(self, frame_b64: str):
        """Submit a new frame for processing"""
        with self._lock:
            self._latest_frame = frame_b64

    def run(self):
        while True:
            frame_b64 = None
            with self._lock:
                if self._latest_frame is not None:
                    frame_b64 = self._latest_frame
                    self._latest_frame = None
                    self.result["processing"] = True

            if frame_b64 is None:
                time.sleep(0.01)
                continue

            # Process with Qwen using Ollama client
            try:
                start_time = time.perf_counter()
                client = ollama.Client(host=self.host)
                # Use Ollama client for multimodal inference
                response = client.chat(
                    model=self.model,
                    messages=[
                        {
                            'role': 'user',
                            'content': self.prompt,
                            'images': [frame_b64]
                        }
                    ]
                )
                
                text = response['message']['content']
                qwen_time = int((time.perf_counter() - start_time) * 1000)
                
                with self._lock:
                    self.result = {
                        "text": text,
                        "updated_at": time.time(),
                        "processing": False,
                        "qwen_time": qwen_time
                    }
                    
            except Exception as e:
                with self._lock:
                    self.result = {
                        "text": f"Qwen error: {str(e)}",
                        "updated_at": time.time(),
                        "processing": False,
                        "qwen_time": None
                    }


def draw_fps_and_time(img: np.ndarray, fps: float, yolo_time: int, qwen_time: Optional[int]):
    """Draw FPS and inference times in top-right corner"""
    text = f"FPS: {fps:.1f} | YOLO: {yolo_time}ms"
    if qwen_time:
        text += f" | Qwen: {qwen_time}ms"
    
    # Position in top-right
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Position from top-right
    x = img.shape[1] - text_width - 20
    y = text_height + 20
    
    # Background rectangle
    cv2.rectangle(img, (x-10, y-text_height-10), (x+text_width+10, y+10), (0, 0, 0), -1)
    cv2.rectangle(img, (x-10, y-text_height-10), (x+text_width+10, y+10), (255, 255, 255), 2)
    
    # Text
    cv2.putText(img, text, (x, y), font, font_scale, (255, 255, 255), thickness)


def draw_boxes_and_labels(img: np.ndarray, results):
    """Draw bounding boxes, segmentation masks, and labels"""
    if not hasattr(results, 'boxes') or results.boxes is None:
        return
    
    names = results.names if hasattr(results, 'names') else {}
    
    for i, box in enumerate(results.boxes):
        # Get box coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0].cpu().numpy())
        cls_id = int(box.cls[0].cpu().numpy())
        cls_name = names.get(cls_id, f"class_{cls_id}")
        
        # Draw bounding box
        color = (0, 255, 0)  # Green
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw label with confidence
        label = f"{cls_name} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Background rectangle for label
        cv2.rectangle(img, (x1, y1-text_height-10), (x1+text_width+10, y1), (0, 0, 0), -1)
        
        # Label text
        cv2.putText(img, label, (x1+5, y1-5), font, font_scale, (255, 255, 255), thickness)


def draw_qwen_decision(img: np.ndarray, qwen_result: dict):
    """Draw Qwen decision text with background color that changes on update"""
    text = qwen_result.get("text", "Waiting...")
    
    # Background color changes based on update time
    time_since_update = time.time() - qwen_result.get("updated_at", 0)
    if time_since_update < 2.0:  # Flash for 2 seconds after update
        bg_color = (0, 100, 0)  # Dark green
    else:
        bg_color = (50, 50, 50)  # Dark gray
    
    # Position in bottom-left
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    # Split text into lines if too long
    lines = []
    max_line_length = 60
    for i in range(0, len(text), max_line_length):
        lines.append(text[i:i+max_line_length])
    
    # Calculate total height needed
    line_height = 25
    total_height = len(lines) * line_height + 20
    
    # Position
    x = 20
    y = img.shape[0] - 20
    
    # Draw background
    cv2.rectangle(img, (x-10, y-total_height-10), (x+600, y+10), bg_color, -1)
    cv2.rectangle(img, (x-10, y-total_height-10), (x+600, y+10), (255, 255, 255), 2)
    
    # Draw text lines
    for i, line in enumerate(lines):
        y_pos = y - total_height + (i * line_height) + 20
        cv2.putText(img, line, (x, y_pos), font, font_scale, (255, 255, 255), thickness)


def draw_debug_info(img: np.ndarray, frame_info: dict):
    """Draw debug information on debug image"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    # Debug info in top-left corner
    debug_lines = [
        "DEBUG: Image sent to Qwen",
        f"Size: {img.shape[1]}x{img.shape[0]}",
        f"Timestamp: {frame_info.get('timestamp', 'N/A')}",
        f"Processing: {frame_info.get('processing', False)}",
        f"Last Update: {frame_info.get('last_update', 'N/A')}"
    ]
    
    y_offset = 30
    for i, line in enumerate(debug_lines):
        y_pos = y_offset + (i * 25)
        # Background for text
        (text_width, text_height), baseline = cv2.getTextSize(line, font, font_scale, thickness)
        cv2.rectangle(img, (10, y_pos-text_height-5), (10+text_width+10, y_pos+5), (0, 0, 0), -1)
        cv2.rectangle(img, (10, y_pos-text_height-5), (10+text_width+10, y_pos+5), (255, 255, 0), 2)
        # Text
        cv2.putText(img, line, (15, y_pos), font, font_scale, (255, 255, 255), thickness)


def open_source(args):
    """Open video source based on arguments"""
    source = args.source.lower()
    
    if source == 'image':
        img = cv2.imread(args.path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {args.path}")
        return img, True
    
    elif source == 'video':
        cap = cv2.VideoCapture(args.path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Video not found: {args.path}")
        return cap, False
    
    elif source == 'webcam':
        cap = cv2.VideoCapture(args.cam_index)
        if not cap.isOpened():
            raise RuntimeError(f"Webcam not available at index {args.cam_index}")
        return cap, False
    
    elif source == 'droidcam':
        url = f"http://{args.droidcam_ip}:{args.droidcam_port}/video"
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            raise RuntimeError(f"DroidCam not available at {url}")
        return cap, False
    
    else:
        raise ValueError("Source must be: image, video, webcam, or droidcam")


def main():
    parser = argparse.ArgumentParser(description="YOLO11-seg + Qwen2.5VL inference")
    parser.add_argument('--source', required=True, choices=['image', 'video', 'webcam', 'droidcam'],
                       help='Input source type')
    parser.add_argument('--path', default='', help='Path to image/video file')
    parser.add_argument('--cam-index', type=int, default=0, help='Webcam index')
    parser.add_argument('--droidcam-ip', default='192.168.0.66', help='DroidCam IP address')
    parser.add_argument('--droidcam-port', type=int, default=4747, help='DroidCam port')
    parser.add_argument('--model', default='yolo11l-seg.pt', help='YOLO model path')
    parser.add_argument('--ollama-host', default='http://192.168.0.87:11434', help='Ollama host URL')
    parser.add_argument('--qwen-model', default='qwen2.5vl:7b', help='Qwen model name')
    
    args = parser.parse_args()
    
    # Load YOLO model
    print("Loading YOLO model...")
    model = YOLO(args.model)
    
    # Open source
    print(f"Opening {args.source}...")
    source, is_single = open_source(args)
    
    # Start Qwen worker
    prompt = """Analyze the cardboard in the GREEN bounding box and return JSON with:
{
    "Warp": true/false,
    "OverallQuality": "good"/"medium"/"bad"
}
Be concise and accurate."""
    
    qwen_worker = QwenWorker(args.ollama_host, args.qwen_model, prompt)
    qwen_worker.start()
    
    # FPS tracking
    fps = 0.0
    last_time = time.perf_counter()
    
    def process_frame(frame):
        nonlocal fps, last_time
        
        # YOLO inference
        start_time = time.perf_counter()
        results = model(frame, conf=0.25, iou=0.20, imgsz=640, verbose=False)[0]
        yolo_time = int((time.perf_counter() - start_time) * 1000)
        
        # Create visualization
        vis_img = frame.copy()
        
        # Draw segmentation masks and boxes
        if hasattr(results, 'masks') and results.masks is not None:
            vis_img = results.plot()  # This includes masks and boxes
        
        # Draw our custom boxes and labels
        draw_boxes_and_labels(vis_img, results)
        
        # Calculate FPS
        current_time = time.perf_counter()
        dt = current_time - last_time
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
        last_time = current_time
        
        # Draw FPS and timing info
        qwen_time = qwen_worker.result.get("qwen_time")
        draw_fps_and_time(vis_img, fps, yolo_time, qwen_time)
        
        # Draw Qwen decision
        draw_qwen_decision(vis_img, qwen_worker.result)
        
        # Send frame to Qwen (with YOLO annotations) and show debug window
        qwen_debug_img = vis_img.copy()
        
        # Add debug information to debug image
        frame_info = {
            'timestamp': time.strftime('%H:%M:%S'),
            'processing': qwen_worker.result.get('processing', False),
            'last_update': time.strftime('%H:%M:%S', time.localtime(qwen_worker.result.get('updated_at', 0)))
        }
        draw_debug_info(qwen_debug_img, frame_info)
        
        # Encode and send to Qwen
        _, buffer = cv2.imencode('.jpg', qwen_debug_img)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        qwen_worker.submit_frame(frame_b64)
        
        # Show debug window with what's sent to Qwen
        cv2.imshow('DEBUG: Image sent to Qwen', qwen_debug_img)
        
        return vis_img
    
    if is_single:
        # Single image
        print("Processing single image...")
        result_img = process_frame(source)
        cv2.imshow('YOLO11-seg + Qwen2.5VL', result_img)
        print("Showing main result and debug window...")
        print("Press any key to exit...")
        cv2.waitKey(0)
    else:
        # Video/webcam stream
        print("Starting video stream...")
        print("Press 'q' to quit, 'd' to toggle debug window")
        cap = source
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            result_img = process_frame(frame)
            cv2.imshow('YOLO11-seg + Qwen2.5VL', result_img)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                # Toggle debug window visibility
                debug_visible = cv2.getWindowProperty('DEBUG: Image sent to Qwen', cv2.WND_PROP_VISIBLE)
                if debug_visible > 0:
                    cv2.destroyWindow('DEBUG: Image sent to Qwen')
                else:
                    cv2.imshow('DEBUG: Image sent to Qwen', result_img)
        
        cap.release()
    
    cv2.destroyAllWindows()
    print("Done!")


if __name__ == '__main__':
    main()
