import cv2
import numpy as np
import time
import threading
import asyncio
import ollama
import base64
from ultralytics import YOLO
from typing import Optional, Tuple
import argparse


class YOLOQwenApp:
    def __init__(self):
        import torch
        
        # Load YOLO model with explicit CPU device
        self.model = YOLO('yolo11m-seg.pt')
        self.model.model.model[-1].iou = 0.1  # Set IoU threshold to 10%
        self.model.model.model[-1].conf = 0.6  # Set confidence threshold to 60%
        
        # Force all YOLO operations to CPU
        self.model.to('cpu')
        
        # Verify CPU usage
        print(f"YOLO device: {next(self.model.model.parameters()).device}")
        print("YOLO forced to CPU, GPU available for Qwen")
        
        # Check GPU memory usage - specify RTX GPU (device 1)
        if torch.cuda.is_available():
            torch.cuda.set_device(1)  # Use RTX GPU (device 1)
            print(f"Using GPU device 1 (RTX)")
            print(f"GPU memory before: {torch.cuda.memory_allocated(1)/1024**2:.1f} MB")
            torch.cuda.empty_cache()  # Clear any residual GPU memory
        
        # Set default device for YOLO predictions
        self.yolo_device = 'cpu'
        
        # Configure Ollama to use RTX GPU (device 1)
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Only show RTX GPU to Ollama
        self.ollama_client = ollama.Client()
        self.qwen_response = ""
        self.qwen_updated = False
        self.qwen_processing = False
        self.qwen_request_count = 0
        self.qwen_stats = {"time": 0, "status": "Ready"}
        
        # Set GPU parameters and warm up Qwen model on startup
        self.set_qwen_gpu_params()
        self.warm_up_qwen()
        
        self.fps = 0
        self.inference_time = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Performance optimization settings
        self.skip_frames = 2  # Process every 3rd frame for YOLO
        self.frame_counter = 0
        self.qwen_interval = 30  # Process with Qwen every 30 frames (1-2 seconds)
        self.qwen_counter = 0
        self.last_results = None  # Cache last YOLO results
        
    def get_input_source(self, source_type: str, source_path: str = None) -> cv2.VideoCapture:
        """Initialize video capture based on source type"""
        if source_type == "webcam":
            print("Initializing external webcam...")
            # Try different backends for faster startup
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
            
            for backend in backends:
                print(f"Trying backend: {backend}")
                cap = cv2.VideoCapture(1, backend)
                
                # Quick test to see if camera responds
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Start with lower resolution
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Test if camera is working
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"Camera initialized successfully with backend {backend}")
                    return cap
                else:
                    cap.release()
                    
            # If all backends fail, try camera index 0 (built-in)
            print("External camera failed, trying built-in camera...")
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        elif source_type == "droidcam":
            # DroidCam IP webcam URL format
            droidcam_url = f"http://192.168.0.66:4747/video"
            cap = cv2.VideoCapture(droidcam_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        elif source_type == "video":
            cap = cv2.VideoCapture(source_path)
            # Try different backends if the default fails (helpful for .mov files)
            if not cap.isOpened():
                print(f"Failed with default backend, trying alternatives for {source_path}")
                backends = [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER, cv2.CAP_ANY]
                for backend in backends:
                    print(f"Trying backend: {backend}")
                    cap = cv2.VideoCapture(source_path, backend)
                    if cap.isOpened():
                        print(f"Successfully opened with backend {backend}")
                        break
                    else:
                        cap.release()
            return cap
        elif source_type == "image":
            # For single image, we'll handle this differently
            return None
        else:
            raise ValueError(f"Unknown source type: {source_type}")
    
    def resize_frame(self, frame: np.ndarray, target_size: int = 640) -> np.ndarray:
        """Resize frame to target size while maintaining aspect ratio"""
        h, w = frame.shape[:2]
        scale = target_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(frame, (new_w, new_h))
    
    def draw_bounding_boxes_only(self, frame: np.ndarray, results) -> np.ndarray:
        """Draw only bounding boxes and labels (for Qwen analysis)"""
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            names = results[0].names
            
            for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                x1, y1, x2, y2 = box.astype(int)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw confidence and class label
                label = f"{names[int(cls)]}: {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return frame
    
    def draw_detections(self, frame: np.ndarray, results) -> np.ndarray:
        """Draw bounding boxes, masks, confidence, and class labels"""
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            names = results[0].names
            
            for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                x1, y1, x2, y2 = box.astype(int)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw confidence and class label
                label = f"{names[int(cls)]}: {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Draw segmentation masks
        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            for mask in masks:
                # Resize mask to frame size
                mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                mask_colored = np.zeros_like(frame)
                mask_colored[:, :, 1] = mask_resized * 255  # Green channel
                frame = cv2.addWeighted(frame, 1.0, mask_colored, 0.3, 0)
        
        return frame
    
    def draw_ui_elements(self, frame: np.ndarray) -> np.ndarray:
        """Draw FPS, inference time, and Qwen response"""
        h, w = frame.shape[:2]
        
        # Draw FPS and inference time (top-right)
        fps_text = f"FPS: {self.fps:.1f}"
        inference_text = f"YOLO: {self.inference_time:.0f}ms"
        
        cv2.putText(frame, fps_text, (w - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, inference_text, (w - 150, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw Qwen stats
        qwen_stats_text = f"Qwen: {self.qwen_stats['time']:.0f}ms ({self.qwen_stats['status']})"
        cv2.putText(frame, qwen_stats_text, (w - 200, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw Qwen response with enhanced styling
        if self.qwen_response:
            # Parse JSON response if possible
            try:
                import json
                response_data = json.loads(self.qwen_response.strip('```json').strip('```').strip())
                formatted_lines = [
                    f"CARDBOARD ANALYSIS",
                    f"Warp: {'YES' if response_data.get('Warp', False) else 'NO'}",
                    f"Quality: {response_data.get('Overall quality', 'Unknown').upper()}"
                ]
            except:
                # Fallback to raw response
                formatted_lines = [
                    "🔍 QWEN ANALYSIS",
                    *self.qwen_response.split('\n')[:2]
                ]
            
            # Enhanced styling
            panel_width = 300
            panel_height = len(formatted_lines) * 35 + 20
            panel_x = 10
            panel_y = h - panel_height - 10
            
            # Background colors based on update status
            if self.qwen_updated:
                bg_color = (50, 200, 50)  # Bright green when updated
                border_color = (0, 255, 0)
                text_color = (255, 255, 255)
            else:
                bg_color = (60, 60, 60)   # Dark gray when stable
                border_color = (120, 120, 120)
                text_color = (220, 220, 220)
            
            # Draw main panel with rounded corners effect
            cv2.rectangle(frame, (panel_x, panel_y), 
                         (panel_x + panel_width, panel_y + panel_height), bg_color, -1)
            
            # Draw border
            cv2.rectangle(frame, (panel_x, panel_y), 
                         (panel_x + panel_width, panel_y + panel_height), border_color, 2)
            
            # Draw decorative top bar
            cv2.rectangle(frame, (panel_x, panel_y), 
                         (panel_x + panel_width, panel_y + 8), border_color, -1)
            
            # Draw text with better spacing and fonts
            for i, line in enumerate(formatted_lines):
                y_pos = panel_y + 25 + i * 35
                font_scale = 0.8 if i == 0 else 0.6
                thickness = 2 if i == 0 else 1
                
                # Add shadow effect
                cv2.putText(frame, line, (panel_x + 12, y_pos + 1), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1)
                
                # Main text
                cv2.putText(frame, line, (panel_x + 10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
            
            # Add processing indicator
            if self.qwen_processing:
                # Animated processing dots
                import time
                dots = "..." if int(time.time() * 2) % 2 == 0 else "   "
                processing_text = f"Processing{dots}"
                cv2.putText(frame, processing_text, (panel_x + 10, panel_y + panel_height + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Reset update flag after displaying
        if self.qwen_updated:
            threading.Timer(1.0, lambda: setattr(self, 'qwen_updated', False)).start()
        
        return frame
    
    def process_with_qwen(self, frame_with_boxes: np.ndarray):
        """Process frame with Qwen asynchronously"""
        if self.qwen_processing:
            return
            
        self.qwen_processing = True
        self.qwen_stats["status"] = "Processing..."
        
        def qwen_worker():
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    start_time = time.time()
                    print(f"[TIMING] Qwen request started at {start_time:.3f}")
                    
                    # Increment request counter
                    self.qwen_request_count += 1
                    
                    # First, test if Ollama is responding
                    test_time = time.time()
                    try:
                        models = self.ollama_client.list()
                        print(f"[TIMING] Ollama connection test: {(time.time() - test_time)*1000:.1f}ms")
                    except Exception:
                        raise ConnectionError("Ollama server not responding")
                    
                    # Resize frame for faster processing (maintain aspect ratio)
                    resize_time = time.time()
                    h, w = frame_with_boxes.shape[:2]
                    target_size = 512  # Smaller than 640px for faster processing
                    if max(h, w) > target_size:
                        scale = target_size / max(h, w)
                        new_w, new_h = int(w * scale), int(h * scale)
                        frame_resized = cv2.resize(frame_with_boxes, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        print(f"[TIMING] Frame resized from {w}x{h} to {new_w}x{new_h}: {(time.time() - resize_time)*1000:.1f}ms")
                    else:
                        frame_resized = frame_with_boxes
                    
                    # Encode frame to base64 with optimized quality
                    encode_time = time.time()
                    _, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 75])  # Lower quality for speed
                    img_base64 = base64.b64encode(buffer).decode('utf-8')
                    print(f"[TIMING] Image encoding: {(time.time() - encode_time)*1000:.1f}ms, size: {len(img_base64)/1024:.1f}KB")
                    
                    request_time = time.time()
                    response = self.ollama_client.generate(
                        model="qwen2.5vl:3b",
                        prompt="Describe the cardboard in green bounding box, return only in JSON. \nWarp: True/False\nOverall quality: Good(completely flat, no gap), medium(Slightly gap), bad",
                        images=[img_base64],
                        stream=False,
                        options={
                            "num_gpu": 999,  # Use maximum GPU layers
                            "num_ctx": 2048,
                            "temperature": 0.1,
                            "top_p": 0.9
                        }
                    )
                    print(f"[TIMING] Qwen API request: {(time.time() - request_time)*1000:.1f}ms")
                    
                    self.qwen_response = response['response']
                    self.qwen_updated = True
                    self.qwen_stats["status"] = "Complete"
                    total_time = (time.time() - start_time) * 1000
                    self.qwen_stats["time"] = total_time
                    print(f"[TIMING] Total Qwen processing: {total_time:.1f}ms")
                    
                    # Clear GPU memory after successful inference
                    try:
                        import torch
                        if torch.cuda.is_available():
                            gpu_clear_time = time.time()
                            torch.cuda.empty_cache()  # Clear RTX GPU memory
                            print(f"[TIMING] GPU memory cleared: {(time.time() - gpu_clear_time)*1000:.1f}ms")
                    except ImportError:
                        pass
                        
                    break  # Success, exit retry loop
                        
                except (ConnectionError, TimeoutError, ollama.ConnectError, ollama.RequestError) as e:
                    retry_count += 1
                    self.qwen_stats["status"] = f"Retry {retry_count}/{max_retries}"
                    if retry_count >= max_retries:
                        self.qwen_response = f"Connection failed after {max_retries} retries. Is Ollama running? (ollama serve)"
                        self.qwen_stats["status"] = "Connection Error"
                        self.qwen_stats["time"] = 0
                    else:
                        time.sleep(2)  # Wait before retry
                        
                except Exception as e:
                    self.qwen_response = f"Error: {str(e)}"
                    self.qwen_stats["status"] = "Error"
                    self.qwen_stats["time"] = 0
                    break
                    
            self.qwen_processing = False
        
        threading.Thread(target=qwen_worker, daemon=True).start()
    
    def set_qwen_gpu_params(self):
        """Set Qwen GPU parameters via API using options"""
        print("🔧 Setting Qwen GPU parameters...")
        try:
            # Test the GPU parameters by making a simple request with options
            response = self.ollama_client.generate(
                model="qwen2.5vl:3b",
                prompt="test gpu config",
                stream=False,
                options={
                    "num_gpu": 999,  # Use maximum GPU layers
                    "num_ctx": 2048
                }
            )
            
            print("✅ GPU parameters set successfully")
                
        except Exception as e:
            print(f"⚠️ GPU params error: {str(e)}")
    
    def warm_up_qwen(self):
        """Warm up Qwen model on startup to avoid cold start delays"""
        print("🔥 Warming up Qwen model...")
        try:
            # Create a small test image
            test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.rectangle(test_frame, (10, 10), (90, 90), (0, 255, 0), 2)
            
            # Send warm-up request
            _, buffer = cv2.imencode('.jpg', test_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            start_time = time.time()
            response = self.ollama_client.generate(
                model="qwen2.5vl:3b",
                prompt="warm up",
                images=[img_base64],
                stream=False,
                options={
                    "num_gpu": 999,  # Use maximum GPU layers
                    "num_ctx": 2048,
                    "temperature": 0.1
                }
            )
            
            warm_up_time = (time.time() - start_time) * 1000
            print(f"✅ Qwen model warmed up successfully in {warm_up_time:.1f}ms")
            
            # Clear GPU memory after warm-up
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # Clear RTX GPU memory
            except ImportError:
                pass
                
        except Exception as e:
            print(f"⚠️ Qwen warm-up error: {str(e)}")
    
    def update_fps(self):
        """Calculate and update FPS"""
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= 1.0:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()
    
    def process_single_image(self, image_path: str):
        """Process a single image"""
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not load image {image_path}")
            return
        
        # Resize frame
        frame = self.resize_frame(frame)
        
        # YOLO inference on CPU
        start_time = time.time()
        results = self.model(frame, device=self.yolo_device, verbose=False)
        self.inference_time = (time.time() - start_time) * 1000
        
        # Draw detections (with masks for display)
        frame_with_boxes = self.draw_detections(frame.copy(), results)
        
        # Create frame with only bounding boxes for Qwen
        frame_for_qwen = self.draw_bounding_boxes_only(frame.copy(), results)
        
        # Process with Qwen
        self.process_with_qwen(frame_for_qwen)
        
        # Draw UI elements
        frame_display = self.draw_ui_elements(frame_with_boxes)
        
        # Display
        cv2.imshow('YOLO + Qwen Analysis', frame_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def run_video_stream(self, source_type: str, source_path: str = None):
        """Run video stream processing"""
        cap = self.get_input_source(source_type, source_path)
        
        if cap is None:
            print("Error: Could not initialize video source")
            return
        
        print(f"Starting {source_type} stream. Press 'q' to quit.")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    if source_type == "video":
                        break
                    else:
                        continue
                
                # Resize frame to 640px
                frame = self.resize_frame(frame)
                
                # Frame skipping optimization - only process YOLO every N frames
                self.frame_counter += 1
                if self.frame_counter % (self.skip_frames + 1) == 0:
                    # YOLO inference on CPU
                    start_time = time.time()
                    results = self.model(frame, device=self.yolo_device, verbose=False)
                    self.inference_time = (time.time() - start_time) * 1000
                    self.last_results = results  # Cache results
                else:
                    # Use cached results for skipped frames
                    results = self.last_results if self.last_results else None
                
                # Draw detections (use cached or new results)
                if results:
                    frame_with_boxes = self.draw_detections(frame.copy(), results)
                else:
                    frame_with_boxes = frame.copy()
                
                # Smart Qwen processing - only every N frames and when objects detected
                self.qwen_counter += 1
                if (results and results[0].boxes is not None and 
                    len(results[0].boxes) > 0 and 
                    self.qwen_counter % self.qwen_interval == 0 and 
                    not self.qwen_processing):
                    # Create frame with only bounding boxes for Qwen
                    frame_for_qwen = self.draw_bounding_boxes_only(frame.copy(), results)
                    self.process_with_qwen(frame_for_qwen)
                
                # Update FPS
                self.update_fps()
                
                # Draw UI elements
                frame_display = self.draw_ui_elements(frame_with_boxes)
                
                # Display
                cv2.imshow('YOLO + Qwen Analysis', frame_display)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='YOLO + Qwen Analysis Tool')
    parser.add_argument('--source', choices=['webcam', 'droidcam', 'video', 'image'], 
                       default='webcam', help='Input source type')
    parser.add_argument('--path', type=str, help='Path to video file or image')
    
    args = parser.parse_args()
    
    app = YOLOQwenApp()
    
    if args.source == 'image':
        if not args.path:
            print("Error: --path required for image source")
            return
        app.process_single_image(args.path)
    else:
        app.run_video_stream(args.source, args.path)


if __name__ == "__main__":
    main()