import cv2
import numpy as np
import time
import threading
import asyncio
import openai
import base64
import pandas as pd
import json
from datetime import datetime
from ultralytics import YOLO
from typing import Optional, Tuple, List, Dict
import argparse
import os
import random


class CardboardProductionOpenAI:
    def __init__(self, barcode: str = "1234567890", video_source: str = "0", yolo_device: str = "cpu"):
        import torch
        
        # Store parameters
        self.video_source = video_source
        self.yolo_device = yolo_device
        
        # Initialize OpenAI client
        openai.api_key = "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"
        self.openai_client = openai.OpenAI(api_key=openai.api_key)
        
        # Load YOLO model
        self.model = YOLO('yolo11m-seg.pt')
        self.model.model.model[-1].iou = 0.1  # Set IoU threshold to 10%
        self.model.model.model[-1].conf = 0.4  # Set confidence threshold to 60%
        
        # Configure YOLO device
        if yolo_device.lower() == "gpu" and torch.cuda.is_available():
            self.model.to('cuda:0')  # Use RTX GPU
            print(f"YOLO device: GPU (cuda:0)")
        else:
            self.model.to('cpu')
            print(f"YOLO device: CPU")
            
        print(f"Using OpenAI GPT-4o model for vision analysis")
        print(f"Video source: {video_source}")
        
        # Check GPU memory usage - specify RTX GPU (device 0)
        if torch.cuda.is_available():
            torch.cuda.set_device(0)  # Use RTX GPU (device 0)
            print(f"Using GPU device 0 (RTX)")
            print(f"GPU memory before: {torch.cuda.memory_allocated(0)/1024**2:.1f} MB")
            torch.cuda.empty_cache()  # Clear any residual GPU memory
        
        # Set device for YOLO predictions
        self.yolo_prediction_device = 'cuda:0' if yolo_device.lower() == "gpu" and torch.cuda.is_available() else 'cpu'
        
        # Production settings
        self.barcode = barcode
        self.csv_path = self._find_csv_file("cardboard_database.csv")
        print(f"🏷️  Production barcode: {self.barcode}")
        print(f"💡 Press 'G' during video stream to generate random barcode")
        self.analysis_results = []  # Store 4 analysis results
        self.current_picture = 0
        self.total_pictures = 4
        
        # Test OpenAI API connection
        self.test_openai_connection()
        
        print(f"\n🏭 Production Mode Initialized")
        print(f"📦 Processing Barcode: {self.barcode}")
        print(f"📸 Will capture {self.total_pictures} pictures")
        print(f"💾 Database: {self.csv_path}")

    def _find_csv_file(self, filename):
        """
        Smart file path detection - searches for CSV file in multiple locations
        """
        import os
        
        # Search locations in priority order
        search_paths = [
            # 1. Current working directory
            os.path.join(os.getcwd(), filename),
            # 2. Script directory
            os.path.join(os.path.dirname(os.path.abspath(__file__)), filename),
            # 3. Parent directory of script
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), filename),
            # 4. Current directory with Claude-Code-App subfolder
            os.path.join(os.getcwd(), "Claude-Code-App", filename),
            # 5. Just the filename (current directory)
            filename
        ]
        
        print(f"🔍 Searching for {filename}...")
        for path in search_paths:
            print(f"   Checking: {path}")
            if os.path.exists(path):
                print(f"✅ Found: {path}")
                return path
        
        # If not found, create in current working directory
        fallback_path = os.path.join(os.getcwd(), filename)
        print(f"📝 File not found, will create: {fallback_path}")
        return fallback_path

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
                
                # Try different backends for webcam
                backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
                for backend in backends:
                    try:
                        cap = cv2.VideoCapture(webcam_index, backend)
                        if cap.isOpened():
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                            print(f"Webcam initialized with backend {backend}")
                            return cap
                        cap.release()
                    except:
                        continue
                
                # Fallback to default backend
                cap = cv2.VideoCapture(webcam_index)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    print("Webcam initialized with default backend")
                    return cap
                
                print(f"Failed to initialize webcam index {webcam_index}")
                return None
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
            print("Initializing webcam...")
            
            # Try different camera indices and backends
            camera_indices = [0, 1, 2]  # Try common camera indices
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
            
            for camera_index in camera_indices:
                for backend in backends:
                    try:
                        print(f"Trying camera index {camera_index} with backend {backend}")
                        cap = cv2.VideoCapture(camera_index, backend)
                        
                        if cap.isOpened():
                            # Configure camera
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                            cap.set(cv2.CAP_PROP_FPS, 30)
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            
                            # Test if camera is working
                            ret, frame = cap.read()
                            if ret and frame is not None:
                                print(f"✅ Camera {camera_index} initialized successfully with backend {backend}")
                                return cap
                            else:
                                cap.release()
                    except Exception as e:
                        print(f"❌ Error with camera {camera_index}, backend {backend}: {e}")
                        continue
            
            print("❌ Could not initialize any camera")
            return None
        else:
            raise ValueError(f"Production mode only supports webcam input")
    
    def resize_frame(self, frame: np.ndarray, target_size: int = 640) -> np.ndarray:
        """Resize frame to target size while maintaining aspect ratio"""
        h, w = frame.shape[:2]
        scale = target_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(frame, (new_w, new_h))
    
    def draw_bounding_boxes_only(self, frame: np.ndarray, results) -> np.ndarray:
        """Draw only bounding boxes and labels (for OpenAI analysis)"""
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
    
    def draw_production_ui(self, frame: np.ndarray) -> np.ndarray:
        """Draw production UI elements"""
        h, w = frame.shape[:2]
        
        # Draw barcode and progress info
        barcode_text = f"Barcode: {self.barcode}"
        progress_text = f"Picture: {self.current_picture}/{self.total_pictures}"
        instruction_text = "ENTER=Capture | G=Random Barcode | Q=Quit"
        
        # Background panel
        panel_height = 100
        cv2.rectangle(frame, (10, 10), (w - 10, panel_height), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, 10), (w - 10, panel_height), (0, 255, 0), 2)
        
        # Text
        cv2.putText(frame, barcode_text, (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, progress_text, (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, instruction_text, (20, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Show recent analysis results
        if self.analysis_results:
            results_y = h - 150
            cv2.rectangle(frame, (10, results_y), (w - 10, h - 10), (40, 40, 40), -1)
            cv2.putText(frame, "Recent Analysis Results:", (20, results_y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            for i, result in enumerate(self.analysis_results[-3:]):  # Show last 3
                y_pos = results_y + 50 + i * 25
                result_text = f"Pic {result['picture']}: Warp={result['warp']}, Quality={result['quality']}"
                cv2.putText(frame, result_text, (30, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def analyze_with_openai(self, frame_with_boxes: np.ndarray, picture_number: int) -> Dict:
        """Analyze frame with OpenAI GPT-4o and return structured result"""
        print(f"\n🔍 Analyzing Picture {picture_number} with GPT-4o...")
        
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
            print(f"✅ Picture {picture_number} analyzed in {analysis_time:.1f}ms")
            print(f"📊 Result: Warp={result['warp']}, Quality={result['quality']}")
            
            return {
                "picture": picture_number,
                "warp": result["warp"],
                "quality": result["quality"],
                "timestamp": datetime.now().isoformat(),
                "analysis_time_ms": analysis_time
            }
            
        except Exception as e:
            print(f"❌ Analysis error for picture {picture_number}: {str(e)}")
            return {
                "picture": picture_number,
                "warp": False,
                "quality": "unknown",
                "timestamp": datetime.now().isoformat(),
                "analysis_time_ms": 0,
                "error": str(e)
            }
    
    def calculate_final_results(self) -> Tuple[bool, str]:
        """Calculate final averaged results from 4 pictures"""
        if len(self.analysis_results) < 4:
            return False, "unknown"
        
        # Count warp occurrences
        warp_count = sum(1 for r in self.analysis_results if r.get("warp", False))
        warp_final = warp_count >= 2  # Majority vote
        
        # Quality scoring and averaging
        quality_scores = []
        for result in self.analysis_results:
            quality = result.get("quality", "medium")
            if quality == "good":
                quality_scores.append(3)
            elif quality == "medium":
                quality_scores.append(2)
            elif quality == "bad":
                quality_scores.append(1)
            else:
                quality_scores.append(2)  # Default to medium
        
        avg_quality_score = sum(quality_scores) / len(quality_scores)
        
        if avg_quality_score >= 2.5:
            quality_final = "good"
        elif avg_quality_score >= 1.5:
            quality_final = "medium"
        else:
            quality_final = "bad"
        
        print(f"\n📈 Final Results Calculation:")
        print(f"   Warp votes: {warp_count}/4 → Final: {warp_final}")
        print(f"   Quality avg: {avg_quality_score:.2f} → Final: {quality_final}")
        
        return warp_final, quality_final
    
    def update_database(self, final_warp: bool, final_quality: str):
        """Update CSV database with results - creates new barcode entries if not found"""
        try:
            # Validate barcode format (digits only, reasonable length)
            if not (self.barcode.isdigit() and 6 <= len(self.barcode) <= 12):
                print(f"❌ Invalid barcode format: {self.barcode} (must be 6-12 digits)")
                return False
            
            # Read existing CSV
            df = pd.read_csv(self.csv_path)
            
            # Find the barcode row
            barcode_int = int(self.barcode)
            barcode_row = df[df['barcode'] == barcode_int]
            
            if barcode_row.empty:
                # Create new row for barcode not found in database
                print(f"🆕 Barcode {self.barcode} not found, creating new entry...")
                
                # Create new row data
                new_row = {
                    'barcode': barcode_int,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'completed'
                }
                
                # Add picture results
                for i, result in enumerate(self.analysis_results, 1):
                    new_row[f'picture_{i}_warp'] = result['warp']
                    new_row[f'picture_{i}_quality'] = result['quality']
                
                # Add final results
                new_row['final_warp'] = final_warp
                new_row['final_quality'] = final_quality
                
                # Add new row to dataframe
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                print(f"✅ New barcode {self.barcode} added to database")
                
            else:
                # Update existing barcode entry
                row_index = barcode_row.index[0]
                print(f"🔄 Updating existing barcode {self.barcode}...")
                
                # Update individual picture results
                for i, result in enumerate(self.analysis_results, 1):
                    df.loc[row_index, f'picture_{i}_warp'] = result['warp']
                    df.loc[row_index, f'picture_{i}_quality'] = result['quality']
                
                # Update final results
                df.loc[row_index, 'final_warp'] = final_warp
                df.loc[row_index, 'final_quality'] = final_quality
                df.loc[row_index, 'status'] = 'completed'
                df.loc[row_index, 'timestamp'] = datetime.now().isoformat()
                print(f"✅ Existing barcode {self.barcode} updated")
            
            # Save back to CSV
            df.to_csv(self.csv_path, index=False)
            
            print(f"💾 Database saved successfully for barcode {self.barcode}")
            return True
            
        except Exception as e:
            print(f"❌ Database update error: {str(e)}")
            return False
    
    def generate_random_barcode(self) -> str:
        """Generate a random 10-digit barcode"""
        barcode = ''.join([str(random.randint(0, 9)) for _ in range(10)])
        print(f"🎲 Generated random barcode: {barcode}")
        return barcode
    
    def run_production_analysis(self):
        """Run the production analysis workflow"""
        cap = self.get_input_source("webcam")
        
        if cap is None:
            print("❌ Error: Could not initialize camera")
            return
        
        print(f"\n🏭 Starting Production Analysis with OpenAI GPT-4o")
        print(f"📦 Barcode: {self.barcode}")
        print(f"📸 Ready to capture {self.total_pictures} pictures")
        print(f"⌨️  Press ENTER to capture each picture, 'q' to quit\n")
        
        try:
            while self.current_picture < self.total_pictures:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Resize frame
                frame = self.resize_frame(frame)
                
                # YOLO inference
                results = self.model(frame, device=self.yolo_prediction_device, verbose=False)
                
                # Draw production UI
                frame_display = self.draw_production_ui(frame.copy())
                
                # Try to show live feed (handle OpenCV GUI issues)
                try:
                    cv2.imshow('Cardboard Production Analysis - OpenAI', frame_display)
                    # Wait for key press
                    key = cv2.waitKey(1) & 0xFF
                except cv2.error:
                    # GUI not available, use keyboard input instead
                    print("⚠️ GUI not available, using console input mode")
                    print("Press ENTER to capture picture, 'g' for new barcode, 'q' to quit")
                    try:
                        user_input = input().strip().lower()
                        
                        if user_input == '':
                            key = ord('\r')
                        elif user_input == 'g':
                            key = ord('g')
                        elif user_input == 'q':
                            key = ord('q')
                        else:
                            key = 0
                    except EOFError:
                        # Handle case where input is not available (like in some environments)
                        print("Input not available, continuing without user input")
                        key = 0
                
                if key == ord('\r') or key == 13:  # Enter key
                    if results[0].boxes is not None and len(results[0].boxes) > 0:
                        # Create frame with bounding boxes for analysis
                        frame_for_analysis = self.draw_bounding_boxes_only(frame.copy(), results)
                        
                        # Create capture directory if it doesn't exist
                        os.makedirs("./capture", exist_ok=True)
                        
                        # Capture and save the analysis frame
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"capture_openai_{self.barcode}_pic{self.current_picture + 1}_{timestamp}.jpg"
                        cv2.imwrite("./capture/"+filename, frame_for_analysis)
                        print(f"📸 Captured: {filename}")
                        
                        # Analyze with OpenAI GPT-4o
                        analysis_result = self.analyze_with_openai(frame_for_analysis, self.current_picture + 1)
                        self.analysis_results.append(analysis_result)
                        self.current_picture += 1
                        
                        print(f"✅ Picture {self.current_picture}/{self.total_pictures} completed")
                        
                        if self.current_picture >= self.total_pictures:
                            print(f"\n🎉 All {self.total_pictures} pictures captured!")
                            break
                    else:
                        print("⚠️ No cardboard detected! Please ensure cardboard is visible and try again.")
                
                elif key == ord('g') or key == ord('G'):
                    # Generate random barcode and restart analysis
                    self.barcode = self.generate_random_barcode()
                    self.analysis_results = []  # Reset analysis results
                    self.current_picture = 0    # Reset picture counter
                    print(f"🔄 Analysis reset with new barcode: {self.barcode}")
                    print("📸 Ready to capture 4 pictures with new barcode...")
                
                elif key == ord('q'):
                    print("❌ Production analysis cancelled by user")
                    return
            
            # Calculate final results and update database
            if len(self.analysis_results) == self.total_pictures:
                final_warp, final_quality = self.calculate_final_results()
                success = self.update_database(final_warp, final_quality)
                
                if success:
                    print(f"\n🎊 Production Analysis Complete!")
                    print(f"📦 Barcode: {self.barcode}")
                    print(f"📊 Final Result: Warp={final_warp}, Quality={final_quality}")
                    print(f"💾 Database updated successfully")
                else:
                    print(f"\n⚠️ Analysis complete but database update failed")
        
        finally:
            cap.release()
            try:
                cv2.destroyAllWindows()
            except:
                pass  # Ignore GUI errors on cleanup


def main():
    parser = argparse.ArgumentParser(description='Cardboard Production Analysis System - OpenAI Version')
    parser.add_argument('--barcode', type=str, default='1234567890', 
                       help='Barcode for the pallet (default: 1234567890)')
    parser.add_argument('--video-source', type=str, default='0',
                       help='Video source: webcam index (0,1,2...) or video file path (default: 0)')
    parser.add_argument('--yolo-device', type=str, default='gpu', choices=['cpu', 'gpu'],
                       help='Device for YOLO processing: cpu or gpu (default: cpu)')
    
    args = parser.parse_args()
    
    app = CardboardProductionOpenAI(
        barcode=args.barcode, 
        video_source=args.video_source,
        yolo_device=args.yolo_device
    )
    app.run_production_analysis()


if __name__ == "__main__":
    main()