import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading
import os
import datetime
import time
import GPUtil
import psutil
import openai
import base64
import json
import asyncio
from typing import Dict, List

class YOLOGUIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Detection GUI")
        self.root.geometry("1200x800")
        
        # Variables
        default_model = "C:/Users/76135/Desktop/ADSYS-Cardboard-AI-Detection/Versions/V2 Nanobanana Sythetic cardboard bundle dataset/models/weights/model_- 6 september 2025 11_35.pt"
        self.model_path = tk.StringVar(value=default_model)
        self.image_path = tk.StringVar()
        self.video_path = tk.StringVar()
        self.save_path = tk.StringVar(value="C:/Users/76135/Desktop/ADSYS-Cardboard-AI-Detection/trash")
        self.webcam_index = tk.IntVar(value=0)
        self.confidence_threshold = tk.DoubleVar(value=0.25)
        self.iou_threshold = tk.DoubleVar(value=0.20)
        self.device = tk.StringVar(value="cuda")
        self.source_type = tk.StringVar(value="image")
        
        # Notification system
        self.notification_id = None
        
        self.model = None
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.last_update_time = 0
        
        # FPS and performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.frame_times = []
        
        # Performance optimization settings
        self.frame_skip = 1  # Process every N frames for real-time
        self.frame_count = 0
        self.batch_size = 1  # Can be increased for batch processing
        self.last_inference_result = None
        
        # Video recording settings
        self.video_writer = None
        self.is_recording = False
        self.recording_filename = None
        self.auto_save_video = tk.BooleanVar(value=False)
        
        # OpenAI Vision LLM settings
        self.openai_enabled = tk.BooleanVar(value=True)
        self.openai_api_key = "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"
        self.openai_client = None
        self.analysis_results = []
        self.analysis_interval = 30  # Analyze every 30 frames
        self.frame_count_for_analysis = 0
        self.last_analysis_time = 0
        
        # Initialize OpenAI client
        self.init_openai_client()
        
        self.setup_ui()
        self.auto_load_model()
        self.test_openai_connection()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Left panel - Controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Model selection
        ttk.Label(control_frame, text="Model Path:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(control_frame, textvariable=self.model_path, width=40).grid(row=0, column=1, pady=2)
        ttk.Button(control_frame, text="Browse", command=self.browse_model).grid(row=0, column=2, pady=2)
        
        # Source selection
        ttk.Label(control_frame, text="Source Type:").grid(row=1, column=0, sticky=tk.W, pady=2)
        source_combo = ttk.Combobox(control_frame, textvariable=self.source_type, 
                                   values=["image", "video", "webcam"], state="readonly")
        source_combo.grid(row=1, column=1, pady=2)
        source_combo.bind("<<ComboboxSelected>>", self.on_source_combo_change)
        
        # Image path
        self.image_frame = ttk.Frame(control_frame)
        self.image_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=2)
        ttk.Label(self.image_frame, text="Image Path:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(self.image_frame, textvariable=self.image_path, width=40).grid(row=0, column=1)
        ttk.Button(self.image_frame, text="Browse", command=self.browse_image).grid(row=0, column=2)
        
        # Video path
        self.video_frame = ttk.Frame(control_frame)
        self.video_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=2)
        ttk.Label(self.video_frame, text="Video Path:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(self.video_frame, textvariable=self.video_path, width=40).grid(row=0, column=1)
        ttk.Button(self.video_frame, text="Browse", command=self.browse_video).grid(row=0, column=2)
        
        # Auto-save video checkbox
        self.auto_save_checkbox = ttk.Checkbutton(self.video_frame, text="Auto-save processed video", 
                                                 variable=self.auto_save_video, command=self.on_auto_save_change)
        self.auto_save_checkbox.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=2)
        
        # Webcam index
        self.webcam_frame = ttk.Frame(control_frame)
        self.webcam_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=2)
        ttk.Label(self.webcam_frame, text="Webcam Index:").grid(row=0, column=0, sticky=tk.W)
        webcam_spin = ttk.Spinbox(self.webcam_frame, from_=0, to=10, textvariable=self.webcam_index, 
                                 width=10, command=self.on_webcam_change)
        webcam_spin.grid(row=0, column=1)
        
        # Parameters
        ttk.Label(control_frame, text="Confidence Threshold:").grid(row=5, column=0, sticky=tk.W, pady=2)
        conf_scale = ttk.Scale(control_frame, from_=0.0, to=1.0, variable=self.confidence_threshold, 
                              orient=tk.HORIZONTAL, command=self.on_conf_change)
        conf_scale.grid(row=5, column=1, pady=2)
        ttk.Label(control_frame, textvariable=self.confidence_threshold).grid(row=5, column=2, pady=2)
        
        ttk.Label(control_frame, text="IoU Threshold:").grid(row=6, column=0, sticky=tk.W, pady=2)
        iou_scale = ttk.Scale(control_frame, from_=0.0, to=1.0, variable=self.iou_threshold, 
                             orient=tk.HORIZONTAL, command=self.on_iou_change)
        iou_scale.grid(row=6, column=1, pady=2)
        ttk.Label(control_frame, textvariable=self.iou_threshold).grid(row=6, column=2, pady=2)
        
        ttk.Label(control_frame, text="Device:").grid(row=7, column=0, sticky=tk.W, pady=2)
        device_combo = ttk.Combobox(control_frame, textvariable=self.device, values=["cpu", "cuda"], 
                                   state="readonly")
        device_combo.grid(row=7, column=1, pady=2)
        device_combo.bind("<<ComboboxSelected>>", self.on_device_change)
        
        # Performance settings
        perf_frame = ttk.LabelFrame(control_frame, text="Performance", padding="5")
        perf_frame.grid(row=8, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(perf_frame, text="Frame Skip:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.frame_skip_var = tk.IntVar(value=2)
        skip_spin = ttk.Spinbox(perf_frame, from_=1, to=5, textvariable=self.frame_skip_var, 
                               width=10, command=self.on_frame_skip_change)
        skip_spin.grid(row=0, column=1, pady=2)
        ttk.Label(perf_frame, text="(Process every Nth frame)").grid(row=0, column=2, sticky=tk.W, pady=2)
        
        # Save path
        ttk.Label(control_frame, text="Save Path:").grid(row=9, column=0, sticky=tk.W, pady=2)
        ttk.Entry(control_frame, textvariable=self.save_path, width=40).grid(row=9, column=1, pady=2)
        ttk.Button(control_frame, text="Browse", command=self.browse_save_path).grid(row=9, column=2, pady=2)
        
        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=10, column=0, columnspan=3, pady=10)
        
        ttk.Button(button_frame, text="Load Model", command=self.load_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Start", command=self.start_detection).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Stop", command=self.stop_detection).pack(side=tk.LEFT, padx=5)
        
        # Recording buttons
        self.record_button = ttk.Button(button_frame, text="Start Recording", command=self.toggle_recording)
        self.record_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Save Result", command=self.save_result).pack(side=tk.LEFT, padx=5)
        
        # OpenAI Vision LLM settings
        openai_frame = ttk.LabelFrame(control_frame, text="Vision LLM (OpenAI GPT-4o)", padding="5")
        openai_frame.grid(row=11, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # OpenAI enable checkbox
        ttk.Checkbutton(openai_frame, text="Enable cardboard quality analysis", 
                       variable=self.openai_enabled, command=self.on_openai_toggle).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=2)
        
        # Analysis interval setting
        ttk.Label(openai_frame, text="Analysis Interval:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.analysis_interval_var = tk.IntVar(value=30)
        interval_spin = ttk.Spinbox(openai_frame, from_=10, to=120, textvariable=self.analysis_interval_var, 
                                   width=10, command=self.on_analysis_interval_change)
        interval_spin.grid(row=1, column=1, pady=2)
        ttk.Label(openai_frame, text="frames").grid(row=1, column=2, sticky=tk.W, pady=2)
        
        # Quality results display area
        results_frame = ttk.LabelFrame(main_frame, text="Cardboard Quality Analysis Results", padding="10")
        results_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
        # Results text area with scrollbar
        self.results_text = scrolledtext.ScrolledText(results_frame, height=12, width=80, wrap=tk.WORD,
                                                     font=("Courier", 10))
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Initial message
        self.results_text.insert(tk.END, "Vision LLM Analysis Results will appear here...\n")
        self.results_text.insert(tk.END, "Enable OpenAI analysis and start detection to begin cardboard quality assessment.\n\n")
        self.results_text.config(state=tk.DISABLED)
        
        # Right panel - Display
        display_frame = ttk.LabelFrame(main_frame, text="Display", padding="10")
        display_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.canvas = tk.Canvas(display_frame, width=800, height=600, bg="black")
        self.canvas.pack()
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Notification label (initially hidden)
        self.notification_label = ttk.Label(self.root, text="", background="lightgreen", 
                                           foreground="black", font=("Arial", 10, "bold"))
        self.notification_label.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        self.notification_label.grid_remove()
        
        # Initialize source visibility
        self.on_source_change()
    
    def on_source_change(self, event=None):
        source = self.source_type.get()
        self.image_frame.grid_remove()
        self.video_frame.grid_remove()
        self.webcam_frame.grid_remove()
        
        if source == "image":
            self.image_frame.grid()
        elif source == "video":
            self.video_frame.grid()
        elif source == "webcam":
            self.webcam_frame.grid()
    
    def on_source_combo_change(self, event=None):
        self.on_source_change(event)
        self.show_notification("Source type changed")
    
    def on_conf_change(self, value=None):
        self.show_notification("Confidence threshold updated")
    
    def on_iou_change(self, value=None):
        self.show_notification("IoU threshold updated")
    
    def on_device_change(self, event=None):
        self.show_notification("Device changed")
    
    def on_webcam_change(self):
        self.show_notification("Webcam index updated")
    
    def on_frame_skip_change(self):
        self.frame_skip = self.frame_skip_var.get()
        self.show_notification(f"Frame skip set to {self.frame_skip}")
    
    def on_auto_save_change(self):
        if self.auto_save_video.get():
            self.show_notification("Auto-save video enabled")
        else:
            self.show_notification("Auto-save video disabled")
    
    def on_openai_toggle(self):
        if self.openai_enabled.get():
            self.show_notification("OpenAI cardboard analysis enabled")
        else:
            self.show_notification("OpenAI cardboard analysis disabled")
    
    def on_analysis_interval_change(self):
        self.analysis_interval = self.analysis_interval_var.get()
        self.show_notification(f"Analysis interval set to {self.analysis_interval} frames")
    
    def show_notification(self, message):
        if self.notification_id:
            self.root.after_cancel(self.notification_id)
        
        self.notification_label.config(text=message)
        self.notification_label.grid()
        
        self.notification_id = self.root.after(1500, self.hide_notification)
    
    def hide_notification(self):
        self.notification_label.grid_remove()
        self.notification_id = None
    
    def init_openai_client(self):
        """Initialize OpenAI client"""
        try:
            self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
            print("OpenAI client initialized successfully")
        except Exception as e:
            print(f"Failed to initialize OpenAI client: {e}")
            self.openai_client = None
    
    def test_openai_connection(self):
        """Test OpenAI API connection"""
        if not self.openai_client:
            return
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Test connection"}],
                max_tokens=10
            )
            print("✅ OpenAI API connection successful")
            self.update_results_text("OpenAI GPT-4o connection: ✅ Connected\n")
        except Exception as e:
            print(f"❌ OpenAI API connection failed: {str(e)}")
            self.update_results_text(f"OpenAI GPT-4o connection: ❌ Failed - {str(e)}\n")
    
    def auto_load_model(self):
        if os.path.exists(self.model_path.get()):
            try:
                self.model = YOLO(self.model_path.get())
                
                # Optimize model for GPU performance
                if self.device.get() == "cuda":
                    # Warm up the model with a dummy tensor
                    import torch
                    dummy_input = torch.randn(1, 3, 640, 640).cuda()
                    self.model.model.half()  # Use FP16 for faster inference
                
                self.status_var.set(f"Model auto-loaded: {os.path.basename(self.model_path.get())}")
                self.show_notification("Model auto-loaded successfully")
            except Exception as e:
                self.status_var.set(f"Auto-load failed: {str(e)}")
        else:
            self.status_var.set("Default model path not found")
    
    def browse_model(self):
        path = filedialog.askopenfilename(
            title="Select YOLO Model",
            initialdir="C:/Users/76135/Desktop/ADSYS-Cardboard-AI-Detection/Versions/",
            filetypes=[("PyTorch Model", "*.pt"), ("All files", "*.*")]
        )
        if path:
            self.model_path.set(path)
            self.show_notification("Model path updated")
    
    def browse_image(self):
        path = filedialog.askopenfilename(
            title="Select Image",
            initialdir="C:/Users/76135/Desktop/ADSYS-Cardboard-AI-Detection/Versions/",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if path:
            self.image_path.set(path)
            self.show_notification("Image path updated")
    
    def browse_video(self):
        path = filedialog.askopenfilename(
            title="Select Video",
            initialdir="C:/Users/76135/Desktop/ADSYS-Cardboard-AI-Detection/Versions/",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if path:
            self.video_path.set(path)
            self.show_notification("Video path updated")
    
    def browse_save_path(self):
        path = filedialog.askdirectory(
            title="Select Save Directory",
            initialdir="C:/Users/76135/Desktop/ADSYS-Cardboard-AI-Detection/trash"
        )
        if path:
            self.save_path.set(path)
            self.show_notification("Save path updated")
    
    def load_model(self):
        if not self.model_path.get():
            messagebox.showerror("Error", "Please select a model file")
            return
        
        try:
            self.model = YOLO(self.model_path.get())
            
            # Optimize model for GPU performance
            if self.device.get() == "cuda":
                import torch
                # Enable half precision for faster inference
                self.model.model.half()
                # Warm up the model
                dummy_input = torch.randn(1, 3, 640, 640).cuda()
                
            self.status_var.set(f"Model loaded: {os.path.basename(self.model_path.get())}")
            messagebox.showinfo("Success", "Model loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def start_detection(self):
        if self.model is None:
            messagebox.showerror("Error", "Please load a model first")
            return
        
        source = self.source_type.get()
        
        if source == "image" and not self.image_path.get():
            messagebox.showerror("Error", "Please select an image file")
            return
        elif source == "video" and not self.video_path.get():
            messagebox.showerror("Error", "Please select a video file")
            return
        
        self.is_running = True
        
        if source == "image":
            thread = threading.Thread(target=self.process_image)
        elif source == "video":
            thread = threading.Thread(target=self.process_video)
        elif source == "webcam":
            thread = threading.Thread(target=self.process_webcam)
        
        thread.daemon = True
        thread.start()
    
    def stop_detection(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.stop_recording()
        self.status_var.set("Detection stopped")
    
    def process_image(self):
        try:
            image = cv2.imread(self.image_path.get())
            if image is None:
                self.status_var.set("Error: Could not load image")
                return
            
            results = self.model(
                image,
                conf=self.confidence_threshold.get(),
                iou=self.iou_threshold.get(),
                device=self.device.get(),
                verbose=False
            )
            
            result_image = self.draw_detections(image, results[0])
            self.display_image(result_image)
            self.current_frame = result_image
            
            detections = len(results[0].boxes) if results[0].boxes else 0
            self.status_var.set(f"Image processed: {detections} detection(s)")
            
            # Run OpenAI analysis for single image if enabled and detections found
            if self.openai_enabled.get() and detections > 0:
                def run_image_analysis():
                    try:
                        analysis_result = self.analyze_with_openai(result_image.copy(), detections)
                        if analysis_result:
                            self.analysis_results.append(analysis_result)
                            formatted_result = self.format_analysis_result(analysis_result)
                            self.update_results_text(formatted_result)
                    except Exception as e:
                        print(f"Image analysis thread error: {e}")
                
                threading.Thread(target=run_image_analysis, daemon=True).start()
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
    
    def process_video(self):
        try:
            self.cap = cv2.VideoCapture(self.video_path.get())
            if not self.cap.isOpened():
                self.status_var.set("Error: Could not open video")
                return
            
            # Get video FPS for frame timing
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            
            # Use user-defined frame skip or calculate based on video FPS
            if not hasattr(self, 'frame_skip_var'):
                self.frame_skip = max(1, int(fps / 30))  # Target 30 FPS max
            else:
                self.frame_skip = self.frame_skip_var.get()
            
            # Auto-start recording if checkbox is enabled
            auto_recording_started = False
            if self.auto_save_video.get() and not self.is_recording:
                # Wait for first frame to get dimensions, then start recording
                ret, first_frame = self.cap.read()
                if ret:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
                    # Set current frame temporarily to start recording
                    temp_frame = first_frame.copy()
                    self.current_frame = temp_frame
                    self.start_recording()
                    auto_recording_started = True
            
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                
                # Skip frames for better performance
                if self.frame_count % self.frame_skip == 0:
                    # Resize frame for faster processing if too large
                    height, width = frame.shape[:2]
                    if width > 1280:
                        scale = 1280 / width
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        frame = cv2.resize(frame, (new_width, new_height))
                    
                    # Run inference with optimized settings
                    results = self.model(
                        frame,
                        conf=self.confidence_threshold.get(),
                        iou=self.iou_threshold.get(),
                        device=self.device.get(),
                        verbose=False,
                        imgsz=640,  # Fixed input size for consistency
                        half=True if self.device.get() == "cuda" else False  # Use FP16 on GPU
                    )
                    
                    self.last_inference_result = results[0]
                
                # Always draw detections (using last result if frame was skipped)
                if self.last_inference_result is not None:
                    result_frame = self.draw_detections(frame, self.last_inference_result)
                    
                    # OpenAI Vision LLM Analysis for video
                    self.frame_count_for_analysis += 1
                    current_time = time.time()
                    
                    # Check if it's time for OpenAI analysis
                    if (self.openai_enabled.get() and 
                        self.frame_count_for_analysis % self.analysis_interval == 0 and
                        current_time - self.last_analysis_time > 5.0):  # Minimum 5 seconds between analyses
                        
                        detection_count = len(self.last_inference_result.boxes) if self.last_inference_result.boxes else 0
                        
                        if detection_count > 0:
                            # Run analysis in separate thread to avoid blocking
                            def run_analysis():
                                try:
                                    analysis_result = self.analyze_with_openai(result_frame.copy(), detection_count)
                                    if analysis_result:
                                        self.analysis_results.append(analysis_result)
                                        formatted_result = self.format_analysis_result(analysis_result)
                                        self.update_results_text(formatted_result)
                                except Exception as e:
                                    print(f"Analysis thread error: {e}")
                            
                            threading.Thread(target=run_analysis, daemon=True).start()
                            self.last_analysis_time = current_time
                else:
                    result_frame = self.draw_detections(frame, None)
                
                # Use after_idle to update display in main thread
                self.root.after_idle(lambda f=result_frame: self.display_image(f))
                self.current_frame = result_frame
                
                # Write frame to video if recording
                self.write_frame_to_video(result_frame)
            
            self.cap.release()
            
            # Stop auto-recording if it was started
            if auto_recording_started and self.is_recording:
                self.stop_recording()
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
        finally:
            if self.cap:
                self.cap.release()
    
    def process_webcam(self):
        try:
            self.cap = cv2.VideoCapture(self.webcam_index.get())
            if not self.cap.isOpened():
                self.status_var.set("Error: Could not open webcam")
                return
            
            # Optimize webcam settings for performance
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Use user-defined frame skip
            self.frame_skip = self.frame_skip_var.get() if hasattr(self, 'frame_skip_var') else 2
            
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                
                # Skip frames for better performance
                if self.frame_count % self.frame_skip == 0:
                    # Resize frame for faster processing
                    height, width = frame.shape[:2]
                    if width > 1024:
                        scale = 1024 / width
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        frame = cv2.resize(frame, (new_width, new_height))
                    
                    # Run inference with optimized settings
                    results = self.model(
                        frame,
                        conf=self.confidence_threshold.get(),
                        iou=self.iou_threshold.get(),
                        device=self.device.get(),
                        verbose=False,
                        imgsz=640,  # Fixed input size for consistency
                        half=True if self.device.get() == "cuda" else False,  # Use FP16 on GPU
                        max_det=100  # Limit max detections for speed
                    )
                    
                    self.last_inference_result = results[0]
                
                # Always draw detections (using last result if frame was skipped)
                if self.last_inference_result is not None:
                    result_frame = self.draw_detections(frame, self.last_inference_result)
                    
                    # OpenAI Vision LLM Analysis for webcam
                    self.frame_count_for_analysis += 1
                    current_time = time.time()
                    
                    # Check if it's time for OpenAI analysis
                    if (self.openai_enabled.get() and 
                        self.frame_count_for_analysis % self.analysis_interval == 0 and
                        current_time - self.last_analysis_time > 5.0):  # Minimum 5 seconds between analyses
                        
                        detection_count = len(self.last_inference_result.boxes) if self.last_inference_result.boxes else 0
                        
                        if detection_count > 0:
                            # Run analysis in separate thread to avoid blocking
                            def run_analysis():
                                try:
                                    analysis_result = self.analyze_with_openai(result_frame.copy(), detection_count)
                                    if analysis_result:
                                        self.analysis_results.append(analysis_result)
                                        formatted_result = self.format_analysis_result(analysis_result)
                                        self.update_results_text(formatted_result)
                                except Exception as e:
                                    print(f"Analysis thread error: {e}")
                            
                            threading.Thread(target=run_analysis, daemon=True).start()
                            self.last_analysis_time = current_time
                else:
                    result_frame = self.draw_detections(frame, None)
                
                # Use after_idle to update display in main thread
                self.root.after_idle(lambda f=result_frame: self.display_image(f))
                self.current_frame = result_frame
                
                # Write frame to video if recording
                self.write_frame_to_video(result_frame)
            
            self.cap.release()
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
        finally:
            if self.cap:
                self.cap.release()
    
    def get_gpu_utilization(self):
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                return f"{gpus[0].load * 100:.1f}%"
            return "N/A"
        except:
            return "N/A"
    
    def update_fps(self):
        current_time = time.time()
        self.frame_times.append(current_time)
        
        # Keep only last 30 frame times for rolling average
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        
        # Calculate FPS from frame times
        if len(self.frame_times) > 1:
            time_diff = self.frame_times[-1] - self.frame_times[0]
            if time_diff > 0:
                self.current_fps = (len(self.frame_times) - 1) / time_diff
    
    def draw_detections(self, image, result):
        result_image = image.copy()
        
        # Update FPS
        self.update_fps()
        
        # Add inference info overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        detection_count = len(result.boxes) if result and result.boxes else 0
        info_text = f"Detections: {detection_count}"
        cv2.putText(result_image, info_text, (10, 30), font, 0.7, (255, 255, 255), 2)
        
        # Add FPS and GPU utilization in top right (green text)
        height, width = result_image.shape[:2]
        fps_text = f"FPS: {self.current_fps:.1f}"
        gpu_util = self.get_gpu_utilization()
        gpu_text = f"GPU: {gpu_util}"
        
        # Calculate text sizes for positioning
        (fps_w, fps_h), _ = cv2.getTextSize(fps_text, font, 0.6, 2)
        (gpu_w, gpu_h), _ = cv2.getTextSize(gpu_text, font, 0.6, 2)
        
        # Draw FPS in top right corner
        cv2.putText(result_image, fps_text, (width - fps_w - 10, 30), font, 0.6, (0, 255, 0), 2)
        
        # Draw GPU utilization below FPS
        cv2.putText(result_image, gpu_text, (width - gpu_w - 10, 60), font, 0.6, (0, 255, 0), 2)
        
        # Add recording indicator if recording
        if self.is_recording:
            rec_text = "● REC"
            cv2.putText(result_image, rec_text, (width - 80, 90), font, 0.6, (0, 0, 255), 2)
        
        if result and result.boxes:
            # Create color palette for different classes
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
            
            for i, box in enumerate(result.boxes):
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf.item()
                cls = int(box.cls.item())
                class_name = self.model.names[cls] if hasattr(self.model, 'names') and self.model.names else f"class_{cls}"
                
                # Choose color based on class
                color = colors[cls % len(colors)]
                thickness = 2
                
                # Draw rectangle
                cv2.rectangle(result_image, (x1, y1), (x2, y2), color, thickness)
                
                # Draw label with confidence and class ID
                label = f"{class_name}: {conf:.2f}"
                font_scale = 0.6
                label_thickness = 1
                
                # Get text size
                (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, label_thickness)
                
                # Draw background rectangle for text
                cv2.rectangle(result_image, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
                
                # Draw text
                cv2.putText(result_image, label, (x1, y1 - 5), font, font_scale, (0, 0, 0), label_thickness)
        
        # Add timestamp and confidence threshold info
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conf_text = f"Conf: {self.confidence_threshold.get():.2f} | {timestamp}"
        cv2.putText(result_image, conf_text, (10, result_image.shape[0] - 10), font, 0.5, (255, 255, 255), 1)
        
        return result_image
    
    def display_image(self, image):
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image to fit canvas
        height, width = image_rgb.shape[:2]
        max_size = (800, 600)
        
        if width > max_size[0] or height > max_size[1]:
            scale = min(max_size[0] / width, max_size[1] / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image_rgb = cv2.resize(image_rgb, (new_width, new_height))
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(Image.fromarray(image_rgb))
        
        # Update canvas without clearing (prevents flashing)
        if hasattr(self.canvas, 'image_item'):
            # Update existing image
            self.canvas.itemconfig(self.canvas.image_item, image=photo)
        else:
            # Create initial image
            self.canvas.image_item = self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        
        self.canvas.image = photo  # Keep reference
    
    def save_result(self):
        if self.current_frame is None:
            messagebox.showerror("Error", "No result to save")
            return
        
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detection_result_{timestamp}.jpg"
            save_path = os.path.join(self.save_path.get(), filename)
            
            cv2.imwrite(save_path, self.current_frame)
            self.status_var.set(f"Result saved to: {save_path}")
            messagebox.showinfo("Success", f"Result saved to:\n{save_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save result: {str(e)}")
    
    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        if self.current_frame is None:
            messagebox.showerror("Error", "No video stream to record")
            return
        
        try:
            # Generate filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.recording_filename = os.path.join(self.save_path.get(), f"recorded_video_{timestamp}.mp4")
            
            # Get frame dimensions
            height, width = self.current_frame.shape[:2]
            
            # Set up video writer with MP4 codec
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30  # Fixed FPS for recording
            
            self.video_writer = cv2.VideoWriter(
                self.recording_filename,
                fourcc,
                fps,
                (width, height)
            )
            
            if not self.video_writer.isOpened():
                raise Exception("Failed to open video writer")
            
            self.is_recording = True
            self.record_button.config(text="Stop Recording", style="Accent.TButton")
            self.status_var.set(f"Recording to: {self.recording_filename}")
            self.show_notification("Recording started")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start recording: {str(e)}")
    
    def stop_recording(self):
        if self.is_recording and self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            self.is_recording = False
            self.record_button.config(text="Start Recording", style="")
            
            if self.recording_filename:
                self.status_var.set(f"Recording saved: {self.recording_filename}")
                self.show_notification("Recording stopped and saved")
                messagebox.showinfo("Recording Complete", f"Video saved to:\n{self.recording_filename}")
            
            self.recording_filename = None
    
    def write_frame_to_video(self, frame):
        if self.is_recording and self.video_writer:
            try:
                self.video_writer.write(frame)
            except Exception as e:
                print(f"Error writing frame to video: {e}")
                self.stop_recording()
    
    def analyze_with_openai(self, frame_with_boxes: np.ndarray, detection_count: int) -> Dict:
        """Analyze frame with OpenAI GPT-4o for cardboard quality assessment"""
        if not self.openai_client or not self.openai_enabled.get():
            return None
        
        try:
            start_time = time.time()
            
            # Resize frame for better quality analysis (higher resolution for defect detection)
            h, w = frame_with_boxes.shape[:2]
            cv2.imwrite("temp_frame.jpg", frame_with_boxes)  # Temporary save for debugging
            target_size = 1024  # Increased from 512 for better defect visibility
            if max(h, w) > target_size:
                scale = target_size / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                frame_resized = cv2.resize(frame_with_boxes, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            else:
                frame_resized = frame_with_boxes
            
            # Encode frame to base64 with higher quality for better defect detection
            _, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
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
                                "text": f"""You are a practical cardboard quality inspector for industrial customers. Analyze the {detection_count} cardboard bundle(s) within the green bounding boxes in this image.

INSPECTION CRITERIA:
1. WARP: Look for SIGNIFICANT bending, curving, or deformation that affects usability
2. EDGE CONDITION: Check for major damage, severe fraying, or crushed edges
3. STACKING: Bundles should be reasonably aligned - minor offsets are acceptable
4. SURFACE DAMAGE: Look for major dents, water damage, or structural compromise
5. STRUCTURAL INTEGRITY: Check for severe compression damage or sagging

BALANCED QUALITY STANDARDS:
- "good": Bundles are flat and well-aligned with minimal defects, suitable for customer use
- "medium": Some minor imperfections, slight misalignments, or small edge wear that don't affect functionality
- "bad": SIGNIFICANT defects like major warping, severe damage, or structural issues that compromise usability

Return ONLY this exact JSON format with no additional text:
{{"bundles_analyzed": {detection_count}, "overall_warp": true/false, "overall_quality": "good/medium/bad", "details": "specific defects observed"}}

overall_warp: true if bundles show SIGNIFICANT bending/curving that affects quality, not minor variations
overall_quality: Rate based on practical usability - minor imperfections are acceptable if bundles are functional
details: Focus on defects that actually impact customer satisfaction

Remember: Industrial customers expect FUNCTIONAL quality, not perfection. Rate based on practical usability."""
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
                max_tokens=200,  # Increased for more detailed defect descriptions
                temperature=0.0  # More deterministic for consistent quality assessment
            )
            
            # Parse JSON response
            response_text = response.choices[0].message.content.strip()
            try:
                # Extract JSON from response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    json_text = response_text[json_start:json_end]
                    result = json.loads(json_text)
                else:
                    # Fallback parsing
                    result = {
                        "bundles_analyzed": detection_count,
                        "overall_warp": "warp" in response_text.lower() or "bend" in response_text.lower(),
                        "overall_quality": "medium",
                        "details": "Fallback analysis"
                    }
            except:
                # Emergency fallback
                result = {
                    "bundles_analyzed": detection_count,
                    "overall_warp": False,
                    "overall_quality": "medium",
                    "details": "Analysis completed with defaults"
                }
            
            analysis_time = (time.time() - start_time) * 1000
            
            # Add metadata
            result["timestamp"] = datetime.datetime.now().strftime("%H:%M:%S")
            result["analysis_time_ms"] = int(analysis_time)
            
            return result
            
        except Exception as e:
            print(f"❌ OpenAI analysis error: {str(e)}")
            return {
                "bundles_analyzed": detection_count,
                "overall_warp": False,
                "overall_quality": "unknown",
                "details": f"Error: {str(e)}",
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                "analysis_time_ms": 0
            }
    
    def update_results_text(self, text: str):
        """Update the results text area in a thread-safe way"""
        def update_text():
            self.results_text.config(state=tk.NORMAL)
            self.results_text.insert(tk.END, text)
            self.results_text.see(tk.END)
            self.results_text.config(state=tk.DISABLED)
        
        # Schedule update in main thread
        self.root.after_idle(update_text)
    
    def format_analysis_result(self, result: Dict) -> str:
        """Format analysis result for display"""
        if not result:
            return ""
        
        warp_status = "⚠️ WARPED" if result.get("overall_warp", False) else "✅ FLAT"
        quality = result.get("overall_quality", "unknown").upper()
        
        if quality == "GOOD":
            quality_icon = "✅"
        elif quality == "MEDIUM":
            quality_icon = "⚠️"
        elif quality == "BAD":
            quality_icon = "❌"
        else:
            quality_icon = "❓"
        
        formatted_text = (
            f"[{result.get('timestamp', 'N/A')}] Analysis Complete ({result.get('analysis_time_ms', 0)}ms)\n"
            f"  Bundles Detected: {result.get('bundles_analyzed', 0)}\n"
            f"  Warp Status: {warp_status}\n"
            f"  Quality Rating: {quality_icon} {quality}\n"
            f"  Details: {result.get('details', 'No details')}\n"
            f"  {'='*50}\n\n"
        )
        
        return formatted_text

def main():
    root = tk.Tk()
    app = YOLOGUIApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()