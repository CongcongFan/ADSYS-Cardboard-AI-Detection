import cv2
import numpy as np
import time
import threading
import asyncio
import openai
import base64
import json
import random
from datetime import datetime
from ultralytics import YOLO
from typing import Optional, Tuple, List, Dict
import argparse
import os


class CardboardRealtimeYOLO:
    def __init__(self, video_source: str = "0", yolo_device: str = "cpu"):
        import torch
        
        # Store parameters
        self.video_source = video_source
        self.yolo_device = yolo_device
        
        # Generate random barcode at initialization
        self.barcode = self.generate_random_barcode()
        
        # Initialize OpenAI client
        openai.api_key = "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"
        self.openai_client = openai.OpenAI(api_key=openai.api_key)
        
        # Load YOLO model - check for available models
        yolo_model_path = self._find_yolo_model()
        self.model = YOLO(yolo_model_path)
        self.model.model.model[-1].iou = 0.1  # Set IoU threshold to 10%
        self.model.model.model[-1].conf = 0.4  # Set confidence threshold to 40%
        
        # Configure YOLO device
        if yolo_device.lower() == "gpu" and torch.cuda.is_available():
            self.model.to('cuda:0')  # Use RTX GPU
            print(f"YOLO device: GPU (cuda:0)")
        else:
            self.model.to('cpu')
            print(f"YOLO device: CPU")
            
        print(f"Using YOLO model: {yolo_model_path}")
        print(f"Video source: {video_source}")
        
        # Check GPU memory usage - specify RTX GPU (device 0)
        if torch.cuda.is_available():
            torch.cuda.set_device(0)  # Use RTX GPU (device 0)
            print(f"Using GPU device 0 (RTX)")
            print(f"GPU memory before: {torch.cuda.memory_allocated(0)/1024**2:.1f} MB")
            torch.cuda.empty_cache()  # Clear any residual GPU memory
        
        # Set device for YOLO predictions
        self.yolo_prediction_device = 'cuda:0' if yolo_device.lower() == "gpu" and torch.cuda.is_available() else 'cpu'
        
        # Real-time processing settings
        self.fps_counter = 0
        self.fps_time = time.time()
        self.current_fps = 0.0
        self.processing_time = 0.0
        
        # OpenAI analysis settings
        self.openai_analysis_interval = 5.0  # Analyze with OpenAI every 5 seconds
        self.last_openai_analysis_time = 0
        self.latest_openai_result = None
        self.openai_analysis_in_progress = False
        
        # Test OpenAI API connection
        self.test_openai_connection()
        
        print(f"\n🤖 Real-time YOLO Analysis Initialized")
        print(f"🏷️  Generated barcode: {self.barcode}")
        print(f"📸 Real-time processing with side-by-side display")
        print(f"🧠 OpenAI analysis every {self.openai_analysis_interval}s")
        print(f"💡 Press 'G' to generate new barcode, 'Q' to quit")

    def _find_yolo_model(self) -> str:
        """
        Smart YOLO model detection - searches for model files in multiple locations
        """
        model_names = ['yolo11l-seg.pt', 'yolo11m-seg.pt', 'yolo11s-seg.pt']
        
        # Search locations in priority order
        search_paths = [
            # 1. Script directory
            os.path.dirname(os.path.abspath(__file__)),
            # 2. Current working directory
            os.getcwd(),
            # 3. Parent directory of script
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ]
        
        print(f"🔍 Searching for YOLO model...")
        for model_name in model_names:
            for path in search_paths:
                model_path = os.path.join(path, model_name)
                print(f"   Checking: {model_path}")
                if os.path.exists(model_path):
                    print(f"✅ Found YOLO model: {model_path}")
                    return model_path
        
        # If not found, return default (will be downloaded by ultralytics)
        default_model = 'yolo11m-seg.pt'
        print(f"📝 Model not found locally, will use default: {default_model}")
        return default_model

    def test_openai_connection(self):
        """Test OpenAI API connection"""
        print("🔍 Testing OpenAI API connection...")
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Test connection"}],
                max_tokens=10
            )
            print("✅ OpenAI API connection successful")
        except Exception as e:
            print(f"❌ OpenAI API connection failed: {str(e)}")

    def get_input_source(self, source_type: str = None, source_path: str = None) -> cv2.VideoCapture:
        """Initialize video capture based on source type or video_source parameter"""
        # Use video_source parameter if no source_type specified
        if source_type is None:
            if self.video_source.isdigit():
                # Webcam index
                webcam_index = int(self.video_source)
                print(f"Initializing webcam index {webcam_index}...")
                cap = cv2.VideoCapture(webcam_index)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap
            else:
                # Video file path
                print(f"Loading video file: {self.video_source}")
                backends = [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER, cv2.CAP_ANY]
                for backend in backends:
                    cap = cv2.VideoCapture(self.video_source, backend)
                    if cap.isOpened():
                        print(f"Video loaded with backend {backend}")
                        return cap
                    cap.release()
                print(f"Failed to load video file: {self.video_source}")
                return None
        elif source_type == "webcam":
            print("Initializing external webcam...")
            # Try different backends for faster startup
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
            
            for backend in backends:
                print(f"Trying backend: {backend}")
                cap = cv2.VideoCapture(1, backend)
                
                # Quick test to see if camera responds
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
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
        else:
            raise ValueError(f"Real-time mode only supports webcam input")
    
    def resize_frame(self, frame: np.ndarray, target_size: int = 640) -> np.ndarray:
        """Resize frame to target size while maintaining aspect ratio"""
        h, w = frame.shape[:2]
        scale = target_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(frame, (new_w, new_h))
    
    def draw_yolo_results(self, frame: np.ndarray, results) -> np.ndarray:
        """Draw YOLO bounding boxes, labels, and segmentation masks"""
        annotated_frame = frame.copy()
        
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            names = results[0].names
            
            for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                x1, y1, x2, y2 = box.astype(int)
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw confidence and class label
                label = f"{names[int(cls)]}: {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Draw segmentation masks if available
        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            for mask in masks:
                # Resize mask to frame size
                mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                # Create colored mask
                colored_mask = np.zeros_like(frame)
                colored_mask[:, :, 1] = (mask_resized * 255).astype(np.uint8)  # Green channel
                # Blend with original frame
                annotated_frame = cv2.addWeighted(annotated_frame, 0.8, colored_mask, 0.2, 0)
        
        return annotated_frame
    
    def analyze_with_openai_async(self, frame_with_boxes: np.ndarray) -> None:
        """Analyze frame with OpenAI GPT-4o in a separate thread"""
        if self.openai_analysis_in_progress:
            return  # Skip if analysis is already in progress
            
        self.openai_analysis_in_progress = True
        
        def analysis_thread():
            try:
                result = self.analyze_with_openai(frame_with_boxes)
                self.latest_openai_result = result
                self.last_openai_analysis_time = time.time()
            finally:
                self.openai_analysis_in_progress = False
        
        thread = threading.Thread(target=analysis_thread, daemon=True)
        thread.start()
    
    def analyze_with_openai(self, frame_with_boxes: np.ndarray) -> Dict:
        """Analyze frame with OpenAI GPT-4o and return structured result"""
        print(f"\n🔍 Analyzing frame with GPT-4o...")
        
        start_time = time.time()
        
        try:
            # Resize frame for faster processing
            h, w = frame_with_boxes.shape[:2]
            target_size = 512
            if max(h, w) > target_size:
                scale = target_size / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                frame_resized = cv2.resize(frame_with_boxes, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                frame_resized = frame_with_boxes
            
            # Encode frame to base64
            _, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # OpenAI GPT-4o analysis
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """You are an expert cardboard quality inspector. Analyze the cardboard bundle within the green bounding boxes in this image.

Focus on:
1. WARP: Look for major bending, curving, or deformation on the edge of the cardboard bundle, slight bending is acceptable.

Return ONLY this exact JSON format with no additional text:
{"warp": true}

warp: true if ANY major bending/curving, false if completely flat or slight bending

If cardboard is not visible, return "unknown" in warp."""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_base64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=100,
                temperature=0.1
            )
            
            # Parse JSON response
            response_text = response.choices[0].message.content.strip()
            try:
                # Try to extract JSON from response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    json_text = response_text[json_start:json_end]
                    result = json.loads(json_text)
                else:
                    # Fallback parsing
                    warp = "true" in response_text.lower() or "warp" in response_text.lower()
                    if "good" in response_text.lower():
                        quality = "good"
                    elif "bad" in response_text.lower():
                        quality = "bad"
                    else:
                        quality = "medium"
                    result = {"warp": warp, "quality": quality}
            except:
                # Emergency fallback
                result = {"warp": False, "quality": "medium"}
            
            analysis_time = (time.time() - start_time) * 1000
            print(f"✅ Frame analyzed in {analysis_time:.1f}ms")
            print(f"📊 Result: Warp={result.get('warp', False)}, Quality={result.get('quality', 'medium')}")
            
            return {
                "warp": result.get("warp", False),
                "quality": result.get("quality", "medium"),
                "timestamp": datetime.now().isoformat(),
                "analysis_time_ms": analysis_time
            }
            
        except Exception as e:
            print(f"❌ Analysis error: {str(e)}")
            return {
                "warp": False,
                "quality": "unknown",
                "timestamp": datetime.now().isoformat(),
                "analysis_time_ms": 0,
                "error": str(e)
            }
    
    def draw_realtime_ui(self, frame: np.ndarray) -> np.ndarray:
        """Draw real-time UI elements"""
        h, w = frame.shape[:2]
        
        # Draw barcode and performance info
        barcode_text = f"Barcode: {self.barcode}"
        fps_text = f"FPS: {self.current_fps:.1f}"
        processing_text = f"Processing: {self.processing_time:.1f}ms"
        instruction_text = "G=New Barcode | Q=Quit"
        
        # OpenAI analysis info
        current_time = time.time()
        time_since_last_analysis = current_time - self.last_openai_analysis_time
        next_analysis_in = max(0, self.openai_analysis_interval - time_since_last_analysis)
        
        if self.openai_analysis_in_progress:
            openai_status = "OpenAI: Analyzing..."
            status_color = (0, 255, 255)  # Yellow
        elif self.latest_openai_result:
            warp = self.latest_openai_result.get('warp', False)
            quality = self.latest_openai_result.get('quality', 'unknown')
            openai_status = f"OpenAI: Warp={warp}, Quality={quality}"
            status_color = (0, 255, 0) if not warp else (0, 165, 255)  # Green if good, orange if warp
        else:
            openai_status = f"OpenAI: Next in {next_analysis_in:.1f}s"
            status_color = (200, 200, 200)  # Gray
        
        # Background panel - increased height for OpenAI info
        panel_height = 150
        cv2.rectangle(frame, (10, 10), (w - 10, panel_height), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, 10), (w - 10, panel_height), (0, 255, 255), 2)
        
        # Text
        cv2.putText(frame, barcode_text, (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, fps_text, (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, processing_text, (20, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, openai_status, (20, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(frame, instruction_text, (20, 135), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return frame
    
    def create_side_by_side_display(self, original_frame: np.ndarray, yolo_frame: np.ndarray) -> np.ndarray:
        """Create side-by-side display of original and YOLO processed frames"""
        # Ensure both frames have the same height
        h1, w1 = original_frame.shape[:2]
        h2, w2 = yolo_frame.shape[:2]
        
        if h1 != h2:
            target_height = min(h1, h2)
            if h1 > target_height:
                original_frame = cv2.resize(original_frame, (int(w1 * target_height / h1), target_height))
            if h2 > target_height:
                yolo_frame = cv2.resize(yolo_frame, (int(w2 * target_height / h2), target_height))
        
        # Add labels
        original_labeled = original_frame.copy()
        yolo_labeled = yolo_frame.copy()
        
        # Label original frame
        cv2.putText(original_labeled, "ORIGINAL", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.rectangle(original_labeled, (5, 5), (150, 40), (0, 0, 0), 2)
        
        # Label YOLO frame
        cv2.putText(yolo_labeled, "YOLO INFERENCE", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.rectangle(yolo_labeled, (5, 5), (250, 40), (0, 0, 0), 2)
        
        # Combine frames horizontally
        combined_frame = np.hstack([original_labeled, yolo_labeled])
        
        return combined_frame
    
    def update_fps_counter(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_time)
            self.fps_counter = 0
            self.fps_time = current_time
    
    def generate_random_barcode(self) -> str:
        """Generate a random 10-digit barcode"""
        barcode = ''.join([str(random.randint(0, 9)) for _ in range(10)])
        print(f"🎲 Generated random barcode: {barcode}")
        return barcode
    
    def run_realtime_analysis(self):
        """Run the real-time analysis workflow"""
        cap = self.get_input_source("webcam")
        
        if cap is None:
            print("❌ Error: Could not initialize camera")
            return
        
        print(f"\n🤖 Starting Real-time YOLO + OpenAI Analysis")
        print(f"🏷️  Barcode: {self.barcode}")
        print(f"📸 Side-by-side display: Original | YOLO Inference")
        print(f"🧠 OpenAI analysis every {self.openai_analysis_interval}s when cardboard detected")
        print(f"⌨️  Press 'G' for new barcode, 'Q' to quit\n")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Start processing time measurement
                process_start = time.time()
                
                # Resize frame
                frame_resized = self.resize_frame(frame)
                
                # YOLO inference
                results = self.model(frame_resized, device=self.yolo_prediction_device, verbose=False)
                
                # Draw YOLO results
                yolo_frame = self.draw_yolo_results(frame_resized.copy(), results)
                
                # Check if it's time for OpenAI analysis
                current_time = time.time()
                time_since_last_analysis = current_time - self.last_openai_analysis_time
                
                if (time_since_last_analysis >= self.openai_analysis_interval and 
                    not self.openai_analysis_in_progress and
                    results[0].boxes is not None and len(results[0].boxes) > 0):
                    
                    # Create frame with bounding boxes for OpenAI analysis
                    frame_for_openai = self.draw_yolo_results(frame_resized.copy(), results)
                    self.analyze_with_openai_async(frame_for_openai)
                
                # Add UI to original frame
                original_with_ui = self.draw_realtime_ui(frame_resized.copy())
                
                # Create side-by-side display
                display_frame = self.create_side_by_side_display(original_with_ui, yolo_frame)
                
                # Calculate processing time
                self.processing_time = (time.time() - process_start) * 1000
                
                # Update FPS counter
                self.update_fps_counter()
                
                # Show combined display
                cv2.imshow('Cardboard Real-time YOLO + OpenAI Analysis', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('g') or key == ord('G'):
                    # Generate new random barcode
                    self.barcode = self.generate_random_barcode()
                    print(f"🔄 New barcode generated: {self.barcode}")
                
                elif key == ord('q') or key == ord('Q'):
                    print("❌ Real-time analysis stopped by user")
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("🏁 Real-time YOLO + OpenAI analysis session ended")


def main():
    parser = argparse.ArgumentParser(description='Cardboard Real-time YOLO + OpenAI Analysis System')
    parser.add_argument('--video-source', type=str, default='0',
                       help='Video source: webcam index (0,1,2...) or video file path (default: 0)')
    parser.add_argument('--yolo-device', type=str, default='cpu', choices=['cpu', 'gpu'],
                       help='Device for YOLO processing: cpu or gpu (default: cpu)')
    
    args = parser.parse_args()
    
    app = CardboardRealtimeYOLO(
        video_source=args.video_source,
        yolo_device=args.yolo_device
    )
    app.run_realtime_analysis()


if __name__ == "__main__":
    main()