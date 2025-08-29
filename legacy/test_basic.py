#!/usr/bin/env python3
"""
Simple test script to verify YOLO11-seg works with test image
"""

import cv2
import numpy as np
from ultralytics import YOLO

def test_yolo_basic():
    print("Testing basic YOLO11-seg functionality...")
    
    # Load model
    model = YOLO('yolo11l-seg.pt')
    
    # Load test image
    img_path = './test_img/IMG_5497.JPG'
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Error: Could not load image {img_path}")
        return False
    
    print(f"Image loaded: {img.shape}")
    
    # Run inference
    print("Running YOLO inference...")
    results = model(img, conf=0.25, iou=0.20, imgsz=640, verbose=False)[0]
    
    print(f"Detection results: {len(results.boxes) if results.boxes else 0} objects")
    
    # Check if we have masks
    if hasattr(results, 'masks') and results.masks is not None:
        print(f"Segmentation masks: {len(results.masks)}")
    else:
        print("No segmentation masks found")
    
    # Create visualization
    vis_img = results.plot()
    
    # Show result
    cv2.imshow('Test Result', vis_img)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("Basic test completed successfully!")
    return True

if __name__ == '__main__':
    test_yolo_basic()
