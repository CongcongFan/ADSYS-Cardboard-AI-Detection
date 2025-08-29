import cv2
import numpy as np
import base64
import json
import requests
import time
import threading
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import argparse

class VisionSystem:
    def __init__(self):
        # Initialize YOLO model
        self.yolo_model = YOLO('yolo11l-seg.pt')
        
        # Qwen API configuration
        self.qwen_url = "http://localhost:11434/api/generate"
        self.qwen_headers = {"Content-Type": "application/json"}
        
        # State variables - Improved decision flow
        self.current_decision = ""
        self.decision_updated = False
        self.qwen_processing = False
        self.qwen_processing_start_time = 0
        self.last_successful_qwen_time = 0
        self.qwen_request_interval = 5.0  # Only send new request every 5 seconds AFTER completion
        self.qwen_timeout = 20.0  # Maximum seconds for Qwen processing
        
        # Queue management for requests
        self.pending_qwen_request = None  # Store the next request to send
        self.request_lock = threading.Lock()  # Thread safety
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.inference_time = 0
        self.yolo_skip_counter = 0
        
        # Colors for visualization
        self.colors = {
            'bbox': (0, 255, 0),
            'mask': (255, 0, 0),
            'text_bg': (0, 0, 0),
            'text': (255, 255, 255),
            'decision_bg_normal': (50, 50, 50),
            'decision_bg_updated': (0, 100, 200),
            'qwen_processing': (255, 165, 0)
        }

    def test_qwen_connection(self):
        """Test Qwen connection and model availability"""
        print("🔍 Testing Qwen connection...")
        try:
            test_payload = {
                "model": "qwen2.5vl:7b",
                "prompt": "Say 'Hello' in JSON format: {\"message\": \"Hello\"}",
                "stream": False
            }
            
            print("📡 Testing simple text request...")
            response = requests.post(self.qwen_url, headers=self.qwen_headers, 
                                   data=json.dumps(test_payload), timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Text test successful: {result.get('response', 'No response')}")
                return True
            else:
                print(f"❌ Text test failed: HTTP {response.status_code}")
                print(f"❌ Response: {response.text}")
                return False
                
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to Ollama server at http://localhost:11434")
            print("💡 Make sure Ollama is running: 'ollama serve'")
            return False
        except requests.exceptions.Timeout:
            print("❌ Ollama server is not responding (timeout)")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False

    def encode_image_base64(self, image):
        """Convert OpenCV image to base64 string"""
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')

    def draw_bounding_box_on_original(self, image, bbox):
        """Draw bounding box on original image for Qwen"""
        img_copy = image.copy()
        x1, y1, x2, y2 = bbox.astype(int)
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 3)
        return img_copy

    def queue_qwen_request(self, image_with_bbox, detection_info):
        """Queue a new Qwen request (only if not currently processing)"""
        with self.request_lock:
            if not self.qwen_processing:
                # Check if enough time has passed since last successful completion
                current_time = time.time()
                if current_time - self.last_successful_qwen_time >= self.qwen_request_interval:
                    print(f"🎯 Queuing new Qwen request for {detection_info}")
                    self.pending_qwen_request = {
                        'image': image_with_bbox,
                        'info': detection_info,
                        'timestamp': current_time
                    }
                    self._start_qwen_processing()
                else:
                    remaining_time = self.qwen_request_interval - (current_time - self.last_successful_qwen_time)
                    print(f"⏰ Qwen request queued, waiting {remaining_time:.1f}s for next interval")
            else:
                # Update pending request with latest detection (overwrite previous pending)
                self.pending_qwen_request = {
                    'image': image_with_bbox,
                    'info': detection_info,
                    'timestamp': current_time
                }
                print(f"🔄 Updated pending Qwen request with latest detection: {detection_info}")

    def _start_qwen_processing(self):
        """Start processing the queued Qwen request"""
        def qwen_thread():
            with self.request_lock:
                if not self.pending_qwen_request:
                    print("⚠️ No pending request to process")
                    return
                
                # Get the request data
                request_data = self.pending_qwen_request.copy()
                self.pending_qwen_request = None  # Clear the queue
                self.qwen_processing = True
                self.qwen_processing_start_time = time.time()
            
            print(f"🚀 Starting Qwen processing for: {request_data['info']}")
            
            try:
                img_base64 = self.encode_image_base64(request_data['image'])
                
                payload = {
                    "model": "qwen2.5vl:7b",
                    "prompt": "Describe the cardboard in green bounding box, return only in JSON.\n"
                             "Estimated Warping rate: #%,\n"
                             "Warp: True/False,\n"
                             "Overall quality: Good(completely flat, no gap), Medium(slightly gap), Bad",
                    "stream": False,
                    "images": [img_base64]
                }
                
                print("📡 Sending request to Qwen...")
                response = requests.post(self.qwen_url, headers=self.qwen_headers, 
                                       data=json.dumps(payload), timeout=15)
                
                processing_time = time.time() - self.qwen_processing_start_time
                print(f"⏱️ Qwen processing completed in {processing_time:.2f} seconds")
                
                if response.status_code == 200:
                    result = response.json()
                    new_decision = result.get('response', 'No response')
                    
                    # Update decision atomically
                    with self.request_lock:
                        self.current_decision = new_decision
                        self.decision_updated = True
                        self.last_successful_qwen_time = time.time()
                        
                        # Reset update timer
                        if hasattr(self, 'decision_update_time'):
                            delattr(self, 'decision_update_time')
                    
                    print(f"✅ Decision updated: {new_decision[:100]}...")
                    
                else:
                    error_msg = f"Error: HTTP {response.status_code} - {response.text[:100]}"
                    print(f"❌ HTTP Error: {response.status_code}")
                    
                    with self.request_lock:
                        self.current_decision = error_msg
                        self.decision_updated = True
                        if hasattr(self, 'decision_update_time'):
                            delattr(self, 'decision_update_time')
                    
            except requests.exceptions.Timeout:
                error_msg = "Error: Qwen request timed out (>15s)"
                print("⏰ Timeout: Qwen took too long to respond")
                
                with self.request_lock:
                    self.current_decision = error_msg
                    self.decision_updated = True
                    if hasattr(self, 'decision_update_time'):
                        delattr(self, 'decision_update_time')
                    
            except requests.exceptions.ConnectionError:
                error_msg = "Error: Cannot connect to Qwen (check if Ollama is running)"
                print("🔌 Connection Error: Is Ollama running?")
                
                with self.request_lock:
                    self.current_decision = error_msg
                    self.decision_updated = True
                    if hasattr(self, 'decision_update_time'):
                        delattr(self, 'decision_update_time')
                    
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                print(f"💥 Exception: {str(e)}")
                
                with self.request_lock:
                    self.current_decision = error_msg
                    self.decision_updated = True
                    if hasattr(self, 'decision_update_time'):
                        delattr(self, 'decision_update_time')
                        
            finally:
                with self.request_lock:
                    self.qwen_processing = False
                    
                    # Check if there's a pending request to process
                    if self.pending_qwen_request:
                        print("🔄 Found pending request, starting next processing cycle...")
                        threading.Thread(target=self._start_qwen_processing, daemon=True).start()
                
                total_time = time.time() - self.qwen_processing_start_time
                print(f"🏁 Qwen processing cycle finished (total: {total_time:.2f}s)")
        
        # Start the processing thread
        thread = threading.Thread(target=qwen_thread, daemon=True)
        thread.start()

    def draw_fps_and_inference_time(self, frame):
        """Draw FPS and inference time on top-right corner"""
        height, width = frame.shape[:2]
        
        # FPS text
        fps_text = f"FPS: {self.current_fps:.1f}"
        inference_text = f"Inference: {self.inference_time:.1f}ms"
        
        # Calculate text size and position
        font_scale = 0.6
        thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # FPS
        (fps_w, fps_h), _ = cv2.getTextSize(fps_text, font, font_scale, thickness)
        fps_x = width - fps_w - 10
        fps_y = 30
        
        # Inference time
        (inf_w, inf_h), _ = cv2.getTextSize(inference_text, font, font_scale, thickness)
        inf_x = width - inf_w - 10
        inf_y = fps_y + fps_h + 10
        
        # Draw background rectangles
        cv2.rectangle(frame, (fps_x - 5, fps_y - fps_h - 5), (fps_x + fps_w + 5, fps_y + 5), 
                     self.colors['text_bg'], -1)
        cv2.rectangle(frame, (inf_x - 5, inf_y - inf_h - 5), (inf_x + inf_w + 5, inf_y + 5), 
                     self.colors['text_bg'], -1)
        
        # Draw text
        cv2.putText(frame, fps_text, (fps_x, fps_y), font, font_scale, self.colors['text'], thickness)
        cv2.putText(frame, inference_text, (inf_x, inf_y), font, font_scale, self.colors['text'], thickness)

    def draw_decision_text(self, frame):
        """Draw Qwen decision text with status indication"""
        height, width = frame.shape[:2]
        current_time = time.time()
        
        # Prepare decision text and background color
        with self.request_lock:  # Thread-safe access to state
            if self.qwen_processing:
                processing_duration = current_time - self.qwen_processing_start_time
                
                # Check for timeout
                if processing_duration > self.qwen_timeout:
                    print(f"⚠️ Qwen processing timeout after {processing_duration:.1f}s - forcing reset")
                    self.qwen_processing = False
                    self.current_decision = f"Error: Qwen timed out after {self.qwen_timeout}s"
                    self.decision_updated = True
                    if hasattr(self, 'decision_update_time'):
                        delattr(self, 'decision_update_time')
                    display_text = self.current_decision
                    bg_color = self.colors['decision_bg_normal']
                else:
                    display_text = f"Qwen is analyzing... ({processing_duration:.1f}s)"
                    bg_color = self.colors['qwen_processing']
                    
            else:
                display_text = self.current_decision if self.current_decision else "Waiting for detection..."
                
                # Check if we should show updated background
                if self.decision_updated:
                    if not hasattr(self, 'decision_update_time'):
                        self.decision_update_time = current_time
                    
                    # Show updated background for 3 seconds
                    time_since_update = current_time - self.decision_update_time
                    if time_since_update <= 3.0:
                        bg_color = self.colors['decision_bg_updated']
                    else:
                        bg_color = self.colors['decision_bg_normal']
                        self.decision_updated = False
                        delattr(self, 'decision_update_time')
                else:
                    bg_color = self.colors['decision_bg_normal']
        
        # Add status information
        next_request_time = self.last_successful_qwen_time + self.qwen_request_interval
        time_to_next = max(0, next_request_time - current_time)
        
        if time_to_next > 0 and not self.qwen_processing:
            status_text = f"Next analysis in: {time_to_next:.1f}s"
        elif self.qwen_processing:
            status_text = "Analyzing..."
        else:
            status_text = "Ready for analysis"
        
        display_text = f"{display_text}\n[{status_text}]"
        
        # Wrap text for long decisions
        max_width = width - 40
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        # Split text into lines
        lines = display_text.split('\n')
        wrapped_lines = []
        
        for line in lines:
            words = line.split(' ')
            current_line = ""
            
            for word in words:
                test_line = current_line + " " + word if current_line else word
                (test_w, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
                
                if test_w <= max_width:
                    current_line = test_line
                else:
                    if current_line:
                        wrapped_lines.append(current_line)
                    current_line = word
            
            if current_line:
                wrapped_lines.append(current_line)
        
        # Calculate total text area
        line_height = 25
        total_height = len(wrapped_lines) * line_height + 20
        
        # Draw background
        cv2.rectangle(frame, (10, height - total_height - 10), 
                     (width - 10, height - 10), bg_color, -1)
        
        # Draw text lines
        for i, line in enumerate(wrapped_lines):
            y_pos = height - total_height + (i + 1) * line_height
            cv2.putText(frame, line, (20, y_pos), font, font_scale, 
                       self.colors['text'], thickness)

    def process_frame(self, frame, skip_yolo=False):
        """Process a single frame with YOLO and optionally queue Qwen request"""
        start_time = time.time()
        
        # Resize frame to 720p for display
        height, width = frame.shape[:2]
        if height != 720:
            aspect_ratio = width / height
            new_width = int(720 * aspect_ratio)
            frame = cv2.resize(frame, (new_width, 720))
        
        # YOLO inference
        annotated_frame = frame.copy()
        
        if not skip_yolo:
            results = self.yolo_model(frame, iou=0.9)
            self.inference_time = (time.time() - start_time) * 1000
            self.last_yolo_results = results
        else:
            if hasattr(self, 'last_yolo_results'):
                results = self.last_yolo_results
                self.inference_time = 0
            else:
                results = self.yolo_model(frame, iou=0.9)
                self.inference_time = (time.time() - start_time) * 1000
                self.last_yolo_results = results
        
        # Process YOLO results
        best_detection = None
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            masks = results[0].masks
            
            for i in range(len(boxes)):
                # Get detection info
                bbox = boxes.xyxy[i].cpu().numpy()
                confidence = boxes.conf[i].cpu().numpy()
                class_id = int(boxes.cls[i].cpu().numpy())
                class_name = self.yolo_model.names[class_id]
                
                x1, y1, x2, y2 = bbox.astype(int)
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), self.colors['bbox'], 2)
                
                # Draw segmentation mask if available
                if masks is not None:
                    mask = masks.xy[i]
                    if len(mask) > 0:
                        mask_points = mask.astype(np.int32)
                        cv2.polylines(annotated_frame, [mask_points], True, self.colors['mask'], 2)
                        
                        # Fill mask with transparency
                        overlay = annotated_frame.copy()
                        cv2.fillPoly(overlay, [mask_points], self.colors['mask'])
                        annotated_frame = cv2.addWeighted(annotated_frame, 0.8, overlay, 0.2, 0)
                
                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                cv2.rectangle(annotated_frame, (x1, y1 - label_h - 10), 
                             (x1 + label_w, y1), self.colors['text_bg'], -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
                
                # Keep track of best detection (highest confidence)
                if best_detection is None or confidence > best_detection['confidence']:
                    best_detection = {
                        'bbox': bbox,
                        'confidence': confidence,
                        'class_name': class_name,
                        'frame': frame
                    }
        
        # Queue Qwen request for best detection (if any)
        if best_detection is not None and not skip_yolo:
            detection_info = f"{best_detection['class_name']} ({best_detection['confidence']:.2f})"
            img_with_bbox = self.draw_bounding_box_on_original(
                best_detection['frame'], best_detection['bbox']
            )
            self.queue_qwen_request(img_with_bbox, detection_info)
        
        # Draw overlays
        self.draw_fps_and_inference_time(annotated_frame)
        self.draw_decision_text(annotated_frame)
        
        # Update FPS
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
        
        return annotated_frame

    def run_webcam(self, camera_id=0):
        """Run inference on webcam"""
        print(f"Initializing camera {camera_id}...")
        
        # Try different backends for better camera support
        backends_to_try = [cv2.CAP_DSHOW, cv2.CAP_V4L2, cv2.CAP_ANY]
        
        cap = None
        for backend in backends_to_try:
            try:
                cap = cv2.VideoCapture(camera_id, backend)
                if cap.isOpened():
                    print(f"Successfully opened camera with backend: {backend}")
                    break
                cap.release()
            except:
                continue
        
        if cap is None or not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        # Optimize camera settings
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        
        print("Camera configuration:")
        print(f"  Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print(f"  FPS: {cap.get(cv2.CAP_PROP_FPS)}")
        print("Press 'q' to quit")
        
        # Warm up camera
        for i in range(5):
            ret, _ = cap.read()
            if not ret:
                break
        
        print("Starting inference with sequential Qwen processing...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            processed_frame = self.process_frame(frame)
            cv2.imshow('YOLO + Qwen Vision System (Sequential)', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

    def run_droidcam(self, ip="192.168.0.66", port="4747"):
        """Run inference on DroidCam"""
        droidcam_url = f"http://{ip}:{port}/video"
        cap = cv2.VideoCapture(droidcam_url)
        
        print(f"Connecting to DroidCam at {ip}:{port}")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to get frame from DroidCam")
                time.sleep(1)
                continue
            
            processed_frame = self.process_frame(frame)
            cv2.imshow('YOLO + Qwen Vision System (DroidCam)', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='YOLO11 + Qwen2.5VL Vision System (Sequential Processing)')
    parser.add_argument('--mode', choices=['webcam', 'droidcam', 'test'], 
                       required=True, help='Input mode')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID for webcam mode')
    parser.add_argument('--droidcam-ip', type=str, default='192.168.0.66', 
                       help='DroidCam IP address')
    parser.add_argument('--droidcam-port', type=str, default='4747', 
                       help='DroidCam port')
    parser.add_argument('--qwen-interval', type=float, default=5.0,
                       help='Seconds between Qwen analysis requests (default: 5.0)')
    
    args = parser.parse_args()
    
    # Initialize vision system
    vision_system = VisionSystem()
    vision_system.qwen_request_interval = args.qwen_interval
    
    # Test mode
    if args.mode == 'test':
        print("🧪 Running Qwen connection test...")
        if vision_system.test_qwen_connection():
            print("✅ All tests passed! System ready for sequential processing.")
        else:
            print("❌ Tests failed. Fix connection issues before running.")
        return
    
    try:
        if args.mode == 'webcam':
            vision_system.run_webcam(args.camera)
        elif args.mode == 'droidcam':
            vision_system.run_droidcam(args.droidcam_ip, args.droidcam_port)
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Example usage:
    # python vision_system.py --mode test
    # python vision_system.py --mode webcam --camera 1
    # python vision_system.py --mode webcam --camera 1 --qwen-interval 3.0
    # python vision_system.py --mode droidcam
    
    main()