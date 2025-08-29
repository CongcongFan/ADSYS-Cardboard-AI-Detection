#!/usr/bin/env python3
"""
Simple test script for Ollama Qwen2.5-VL with cardboard images
"""

import requests
import json
import base64
from pathlib import Path

def encode_image(image_path):
    """Encode image to base64 for Ollama"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_ollama_vision(image_path, model_name="qwen2.5vl:7b", prompt="Analyze this cardboard bundle image. Is the bundle flat or warped? Provide your assessment."):
    """Test Ollama vision model with an image"""
    
    # Encode the image
    print(f"Loading image: {image_path}")
    encoded_image = encode_image(image_path)
    
    # Prepare the request
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": model_name,
        "prompt": prompt,
        "images": [encoded_image],
        "stream": False
    }
    
    print(f"Sending request to Ollama...")
    print(f"Model: {model_name}")
    print(f"Prompt: {prompt}")
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print("\n" + "="*60)
            print("OLLAMA RESPONSE:")
            print("="*60)
            print(result['response'])
            print("="*60)
            return result['response']
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to Ollama. Make sure Ollama is running.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    # Test with a sample cardboard image
    test_images = [
        r"C:\Users\76135\Desktop\ADSYS-Cardboard-AI-Detection\test_img\IMG_5497.JPG"
    ]
    
    # Also check for images in the overlay directory
    overlay_dir = Path(r"C:\Users\76135\Desktop\ADSYS-Cardboard-AI-Detection\Yolo DS To Qwen DS\out\overlays_overlay")
    if overlay_dir.exists():
        overlay_images = list(overlay_dir.glob("*.jpg"))[:3]  # Test first 3
        test_images.extend([str(img) for img in overlay_images])
    
    print("Testing Ollama Qwen2.5-VL with Cardboard Images")
    print("=" * 60)
    
    for image_path in test_images:
        if Path(image_path).exists():
            print(f"\nTesting with: {Path(image_path).name}")
            response = test_ollama_vision(image_path)
            
            if response:
                print("\nTest successful!")
            else:
                print("\nTest failed!")
            
            print("\n" + "-" * 60)
        else:
            print(f"Image not found: {image_path}")

if __name__ == "__main__":
    main()