#!/usr/bin/env python3
"""
Test script that applies LoRA-style prompting to existing Ollama model
This simulates the LoRA behavior by using optimized prompts based on your training data
"""

import requests
import json
import base64
from pathlib import Path

def encode_image(image_path):
    """Encode image to base64 for Ollama"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def create_lora_style_prompt():
    """Create a prompt that mimics the LoRA training behavior"""
    return """You are an expert cardboard quality inspector trained to detect warping and deformation in cardboard bundles.

Analyze the cardboard pieces in this image carefully. Focus on:
1. WARP: Look for major bending, curving, or deformation from a flat surface. Slight bending is acceptable.
2. Overall flatness and structural integrity
3. Any visible deformation or irregularities

Based on your analysis, determine if the bundle should PASS or FAIL quality control.

Provide your assessment in this exact format:
- Status: [PASS/FAIL]  
- Reason: [Brief explanation of your assessment]

Be precise and focus only on significant warping that would affect product quality."""

def test_cardboard_qc(image_path, model_name="qwen2.5vl:7b"):
    """Test cardboard QC using LoRA-style prompting"""
    
    print(f"Testing cardboard QC with: {Path(image_path).name}")
    
    # Encode the image
    encoded_image = encode_image(image_path)
    
    # Create LoRA-style prompt
    prompt = create_lora_style_prompt()
    
    # Prepare the request
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": model_name,
        "prompt": prompt,
        "images": [encoded_image],
        "stream": False,
        "options": {
            "temperature": 0.1,  # Low temperature for consistent results
            "top_p": 0.9,
            "repeat_penalty": 1.1
        }
    }
    
    try:
        print("Sending request to Ollama...")
        response = requests.post(url, json=payload, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            print("\n" + "="*60)
            print("CARDBOARD QC ANALYSIS:")
            print("="*60)
            print(result['response'])
            print("="*60)
            
            # Try to extract Pass/Fail status
            response_text = result['response'].upper()
            if "PASS" in response_text and "FAIL" not in response_text:
                status = "PASS"
            elif "FAIL" in response_text:
                status = "FAIL"
            else:
                status = "UNCERTAIN"
            
            print(f"EXTRACTED STATUS: {status}")
            return result['response'], status
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None, None
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to Ollama. Make sure Ollama is running.")
        return None, None
    except Exception as e:
        print(f"Error: {e}")
        return None, None

def main():
    """Test cardboard QC with sample images"""
    
    # Test images
    test_images = []
    
    # Add test images from test_img directory
    test_dir = Path(r"C:\Users\76135\Desktop\ADSYS-Cardboard-AI-Detection\test_img")
    if test_dir.exists():
        for img in test_dir.glob("*.JPG"):
            test_images.append(str(img))
    
    # Add some images from the training overlay directory
    overlay_dir = Path(r"C:\Users\76135\Desktop\ADSYS-Cardboard-AI-Detection\Yolo DS To Qwen DS\out\overlays_overlay")
    if overlay_dir.exists():
        overlay_images = list(overlay_dir.glob("*.jpg"))[:3]  # Test first 3
        test_images.extend([str(img) for img in overlay_images])
    
    print("Testing Cardboard QC with LoRA-style Prompting")
    print("=" * 60)
    print(f"Found {len(test_images)} test images")
    
    results = []
    
    for i, image_path in enumerate(test_images):
        if Path(image_path).exists():
            print(f"\n[{i+1}/{len(test_images)}] Processing: {Path(image_path).name}")
            response, status = test_cardboard_qc(image_path)
            
            if response and status:
                results.append({
                    'image': Path(image_path).name,
                    'status': status,
                    'response': response
                })
                print(f"Result: {status}")
            else:
                print("Failed to get result")
            
            print("-" * 60)
        else:
            print(f"Image not found: {image_path}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY RESULTS:")
    print("="*60)
    
    pass_count = sum(1 for r in results if r['status'] == 'PASS')
    fail_count = sum(1 for r in results if r['status'] == 'FAIL')
    uncertain_count = sum(1 for r in results if r['status'] == 'UNCERTAIN')
    
    print(f"Total processed: {len(results)}")
    print(f"PASS: {pass_count}")
    print(f"FAIL: {fail_count}")  
    print(f"UNCERTAIN: {uncertain_count}")
    
    for result in results:
        print(f"- {result['image']}: {result['status']}")

if __name__ == "__main__":
    main()