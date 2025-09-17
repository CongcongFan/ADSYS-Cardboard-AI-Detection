"""
Test script for synthetic cardboard generator - generates 1 image for testing
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from synthetic_cardboard_generator import CardboardSyntheticGenerator

def test_single_generation():
    # Configuration - test different URL formats
    ROBOFLOW_API_KEY = "yiXU9DZrZTOcO2taUh0g"
    PROJECT_URL = "https://universe.roboflow.com/boxes-jdugd/synthetic_finished_cardboard_b-tflkf"  # Try full URL
    DOWNLOAD_DIR = "generate image folder"
    
    print("Testing Synthetic Cardboard Generator...")
    print("=" * 50)
    
    # Initialize generator
    generator = CardboardSyntheticGenerator(ROBOFLOW_API_KEY, PROJECT_URL, DOWNLOAD_DIR)
    
    # Generate one test image
    print("Generating test image...")
    prompt = generator.generate_prompt()
    print(f"Using prompt: {prompt}")
    
    result = generator.generate_image(prompt, 1)
    
    if result:
        print("SUCCESS: Test generation successful!")
        print(f"Full API response: {result}")
        if 'image_url' in result:
            print(f"Roboflow URL: {result['image_url']}")
        if 'local_path' in result:
            print(f"Local file: {result['local_path']}")
        if 'download_failed' in result:
            print("WARNING: Image generated but download failed")
    else:
        print("FAILED: Test generation failed")

if __name__ == "__main__":
    test_single_generation()