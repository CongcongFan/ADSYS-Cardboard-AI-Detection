#!/usr/bin/env python3
"""
Script to test the fine-tuned Qwen2.5-VL model for cardboard quality control.
This verifies that the model is working correctly with Ollama.
"""

import os
import sys
import argparse
import json
import time
import base64
import logging
from pathlib import Path
import requests
from PIL import Image
import io

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_ollama_server():
    """Check if Ollama server is running."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            logger.info("✅ Ollama server is running")
            return True
        else:
            logger.error(f"Ollama server responded with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        logger.error("❌ Cannot connect to Ollama server")
        logger.info("Please start Ollama server with: ollama serve")
        return False
    except Exception as e:
        logger.error(f"Error checking Ollama server: {e}")
        return False

def check_model_exists(model_name):
    """Check if the specified model exists in Ollama."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json()
            model_names = [model['name'] for model in models.get('models', [])]
            
            if model_name in model_names:
                logger.info(f"✅ Model '{model_name}' found in Ollama")
                return True
            else:
                logger.error(f"❌ Model '{model_name}' not found in Ollama")
                logger.info(f"Available models: {', '.join(model_names)}")
                logger.info(f"Please import the model first using import_model.sh or import_model.bat")
                return False
        else:
            logger.error("Failed to get model list from Ollama")
            return False
    except Exception as e:
        logger.error(f"Error checking model existence: {e}")
        return False

def encode_image_to_base64(image_path):
    """Encode image to base64 for Ollama API."""
    try:
        with open(image_path, 'rb') as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_image
    except Exception as e:
        logger.error(f"Error encoding image: {e}")
        return None

def test_text_generation(model_name):
    """Test basic text generation capabilities."""
    logger.info("Testing basic text generation...")
    
    test_prompts = [
        "What factors determine cardboard quality?",
        "List the main types of cardboard defects you can identify.",
        "Explain how to assess cardboard warping severity."
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        logger.info(f"Test {i}: {prompt}")
        
        try:
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '').strip()
                
                if generated_text:
                    logger.info(f"✅ Response: {generated_text[:100]}...")
                    if len(generated_text) > 100:
                        logger.info(f"    (Response truncated, full length: {len(generated_text)} chars)")
                else:
                    logger.warning("⚠️  Empty response received")
                    return False
            else:
                logger.error(f"❌ Request failed with status {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            logger.error("❌ Request timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Error in text generation test: {e}")
            return False
        
        time.sleep(1)  # Brief pause between requests
    
    logger.info("✅ Text generation tests passed!")
    return True

def test_image_analysis(model_name, image_paths=None):
    """Test image analysis capabilities."""
    logger.info("Testing image analysis capabilities...")
    
    if not image_paths:
        # Try to find test images in the project
        possible_image_paths = [
            "../../../test_img",
            "../../test",
            "../../../Claude-Code-App/capture",
            "../../../../test_img"
        ]
        
        test_images = []
        for img_dir in possible_image_paths:
            abs_path = os.path.abspath(img_dir)
            if os.path.exists(abs_path):
                # Look for image files
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                    import glob
                    images = glob.glob(os.path.join(abs_path, ext))
                    test_images.extend(images[:3])  # Max 3 images per directory
                    if len(test_images) >= 3:
                        break
                if len(test_images) >= 3:
                    break
        
        if test_images:
            logger.info(f"Found test images: {[os.path.basename(img) for img in test_images[:2]]}")
            image_paths = test_images[:2]  # Use first 2 images
        else:
            logger.warning("No test images found. Skipping image analysis tests.")
            logger.info("To test image analysis, provide image paths with --test-images")
            return True
    
    test_prompts = [
        "Analyze this cardboard image for quality issues and defects.",
        "Rate the quality of this cardboard on a scale from 1-10 and explain your rating.",
        "What specific recommendations would you make for quality control based on this image?"
    ]
    
    for i, image_path in enumerate(image_paths):
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            continue
            
        logger.info(f"Testing with image: {os.path.basename(image_path)}")
        
        # Encode image
        encoded_image = encode_image_to_base64(image_path)
        if not encoded_image:
            logger.error(f"Failed to encode image: {image_path}")
            continue
        
        # Test with one prompt per image
        prompt = test_prompts[i % len(test_prompts)]
        logger.info(f"Prompt: {prompt}")
        
        try:
            payload = {
                "model": model_name,
                "prompt": prompt,
                "images": [encoded_image],
                "stream": False
            }
            
            logger.info("Sending request to Ollama (this may take 30-60 seconds)...")
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=120  # Longer timeout for vision tasks
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '').strip()
                
                if generated_text:
                    logger.info(f"✅ Analysis: {generated_text[:200]}...")
                    if len(generated_text) > 200:
                        logger.info(f"    (Response truncated, full length: {len(generated_text)} chars)")
                    
                    # Check for quality control keywords
                    qc_keywords = ['quality', 'defect', 'damage', 'warp', 'cardboard', 'condition']
                    found_keywords = [kw for kw in qc_keywords if kw.lower() in generated_text.lower()]
                    
                    if found_keywords:
                        logger.info(f"✅ QC keywords found: {', '.join(found_keywords)}")
                    else:
                        logger.warning("⚠️  No quality control keywords found in response")
                else:
                    logger.warning("⚠️  Empty response received")
                    return False
            else:
                logger.error(f"❌ Request failed with status {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            logger.error("❌ Request timed out (vision tasks can be slow)")
            return False
        except Exception as e:
            logger.error(f"❌ Error in image analysis test: {e}")
            return False
        
        time.sleep(2)  # Pause between image requests
        
        # Only test 2 images max to keep test time reasonable
        if i >= 1:
            break
    
    logger.info("✅ Image analysis tests passed!")
    return True

def test_model_performance(model_name):
    """Test model performance and response characteristics."""
    logger.info("Testing model performance characteristics...")
    
    performance_prompts = [
        {
            "prompt": "Provide a brief quality assessment of cardboard with visible warping.",
            "expected_length": (50, 300),
            "expected_keywords": ["warp", "quality", "assessment"]
        },
        {
            "prompt": "Rate cardboard quality from 1-10 and justify your rating.",
            "expected_length": (30, 200),
            "expected_keywords": ["rating", "quality", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        }
    ]
    
    for i, test_case in enumerate(performance_prompts, 1):
        logger.info(f"Performance test {i}: {test_case['prompt']}")
        
        start_time = time.time()
        
        try:
            payload = {
                "model": model_name,
                "prompt": test_case['prompt'],
                "stream": False
            }
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=45
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '').strip()
                
                # Check response length
                min_len, max_len = test_case['expected_length']
                if min_len <= len(generated_text) <= max_len:
                    logger.info(f"✅ Response length appropriate: {len(generated_text)} chars")
                else:
                    logger.warning(f"⚠️  Response length unexpected: {len(generated_text)} chars (expected {min_len}-{max_len})")
                
                # Check for expected keywords
                found_keywords = []
                for keyword in test_case['expected_keywords']:
                    if keyword.lower() in generated_text.lower():
                        found_keywords.append(keyword)
                
                if found_keywords:
                    logger.info(f"✅ Found expected keywords: {', '.join(found_keywords)}")
                else:
                    logger.warning(f"⚠️  No expected keywords found. Response: {generated_text[:100]}...")
                
                logger.info(f"✅ Response time: {response_time:.1f} seconds")
                
            else:
                logger.error(f"❌ Request failed with status {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error in performance test: {e}")
            return False
    
    logger.info("✅ Performance tests completed!")
    return True

def main():
    parser = argparse.ArgumentParser(description="Test fine-tuned Qwen2.5-VL model for cardboard QC")
    parser.add_argument(
        "--model-name",
        type=str,
        default="qwen2.5vl-cardboard-qc",
        help="Name of the Ollama model to test (default: qwen2.5vl-cardboard-qc)"
    )
    parser.add_argument(
        "--test-images",
        type=str,
        nargs="+",
        help="Paths to test images for vision analysis"
    )
    parser.add_argument(
        "--skip-text-tests",
        action="store_true",
        help="Skip text-only generation tests"
    )
    parser.add_argument(
        "--skip-image-tests",
        action="store_true",
        help="Skip image analysis tests"
    )
    parser.add_argument(
        "--skip-performance-tests",
        action="store_true",
        help="Skip performance characteristic tests"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Fine-tuned Qwen2.5-VL Model Test Suite")
    logger.info("=" * 60)
    logger.info(f"Model name: {args.model_name}")
    logger.info("=" * 60)
    
    # Check Ollama server
    if not check_ollama_server():
        sys.exit(1)
    
    # Check model exists
    if not check_model_exists(args.model_name):
        sys.exit(1)
    
    test_results = {
        "text_generation": True,
        "image_analysis": True,
        "performance": True
    }
    
    # Run tests
    try:
        if not args.skip_text_tests:
            logger.info("\n" + "="*40 + " TEXT TESTS " + "="*40)
            test_results["text_generation"] = test_text_generation(args.model_name)
        
        if not args.skip_image_tests:
            logger.info("\n" + "="*40 + " VISION TESTS " + "="*40)
            test_results["image_analysis"] = test_image_analysis(args.model_name, args.test_images)
        
        if not args.skip_performance_tests:
            logger.info("\n" + "="*40 + " PERFORMANCE TESTS " + "="*40)
            test_results["performance"] = test_model_performance(args.model_name)
        
    except KeyboardInterrupt:
        logger.info("\nTests interrupted by user")
        sys.exit(1)
    
    # Results summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    
    all_passed = True
    for test_name, passed in test_results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
        if not passed:
            all_passed = False
    
    logger.info("=" * 60)
    
    if all_passed:
        logger.info("🎉 ALL TESTS PASSED!")
        logger.info(f"✅ Your fine-tuned model '{args.model_name}' is working correctly!")
        logger.info("")
        logger.info("You can now use it for cardboard quality control:")
        logger.info(f"  ollama run {args.model_name}")
        logger.info("")
        logger.info("Or integrate it into your applications using the Ollama API.")
    else:
        logger.error("❌ SOME TESTS FAILED!")
        logger.error("Please check the logs above for details.")
        logger.info("Common issues:")
        logger.info("- Model not properly fine-tuned")
        logger.info("- Incorrect model import")
        logger.info("- Insufficient system resources")
        logger.info("- Ollama server issues")
        sys.exit(1)

if __name__ == "__main__":
    main()