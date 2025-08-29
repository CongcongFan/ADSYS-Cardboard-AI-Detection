"""
Synthetic Cardboard Bundle Generator for YOLO Training
Generates realistic cardboard bundle images using Roboflow + DALL-E 3
"""

import requests
import json
import os
import time
import random
from typing import List, Dict, Optional
from datetime import datetime
from urllib.parse import urlparse
from pathlib import Path

class CardboardSyntheticGenerator:
    def __init__(self, roboflow_api_key: str, project_url: str, download_dir: str = "generate image folder"):
        self.api_key = roboflow_api_key
        self.project_url = project_url
        self.api_endpoint = f"https://api.roboflow.com/synthetic-image?api_key={self.api_key}"
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        # Prompt components for variation
        self.lighting_variations = [
            "bright fluorescent warehouse lighting",
            "natural daylight through warehouse skylights", 
            "overhead LED lighting with shadows",
            "mixed fluorescent and natural lighting",
            "warm industrial lighting"
        ]
        
        self.angles = ["5 degree", "6 degree", "7 degree", "8 degree", "9 degree", "10 degree"]
        
        self.bundle_configs = [
            "3 distinct cardboard bundle stacks",
            "4 separate cardboard bundle groups", 
            "multiple rows of cardboard bundles",
            "5 clearly separated bundle stacks",
            "densely arranged cardboard inventory with visible gaps"
        ]
        
        self.differentiation_features = [
            "white shipping labels on some bundles",
            "varying cardboard shades from light to dark brown",
            "some bundles wrapped in plastic, others exposed",
            "bundles of different heights creating visual separation",
            "mix of labeled and unlabeled cardboard stacks"
        ]
        
        self.industrial_context = [
            "forklift visible in background",
            "warehouse shelving in background", 
            "industrial floor markings",
            "concrete warehouse floor with tire marks",
            "clean industrial environment"
        ]

    def generate_prompt(self) -> str:
        """Generate a varied prompt for cardboard bundle synthesis"""
        lighting = random.choice(self.lighting_variations)
        angle = random.choice(self.angles)
        bundles = random.choice(self.bundle_configs)
        features = random.choice(self.differentiation_features)
        context = random.choice(self.industrial_context)
        
        prompt = f"""Industrial warehouse scene showing corrugated cardboard bundles stacked on wooden pallets, 
        viewed from a side angle at {angle} tilt, showing {bundles} with clear visual separation between each group, 
        {features}, {lighting}, {context}, professional warehouse photography style, high quality"""
        
        # Clean up formatting
        prompt = " ".join(prompt.split())
        return prompt

    def download_image(self, image_url: str, filename: str) -> bool:
        """Download image from URL to local directory"""
        try:
            response = requests.get(image_url, stream=True)
            if response.status_code == 200:
                file_path = self.download_dir / filename
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"✓ Downloaded: {filename}")
                return True
            else:
                print(f"✗ Failed to download image: {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Download error: {str(e)}")
            return False
    
    def generate_image(self, prompt: str, image_index: int = 0) -> Optional[Dict]:
        """Generate a single synthetic image and download it locally"""
        payload = {
            "project_url": self.project_url,
            "prompt": prompt
        }
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        try:
            response = requests.post(self.api_endpoint, headers=headers, data=json.dumps(payload))
            
            if response.status_code == 200:
                result = response.json()
                
                # Download the image if URL is provided
                if 'image_url' in result and result['image_url']:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"cardboard_bundle_{timestamp}_{image_index:03d}.jpg"
                    
                    if self.download_image(result['image_url'], filename):
                        result['local_path'] = str(self.download_dir / filename)
                    else:
                        result['download_failed'] = True
                
                return result
            else:
                print(f"Error {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            print(f"Request failed: {str(e)}")
            return None

    def batch_generate(self, num_images: int, output_file: str = "generation_log.txt") -> List[Dict]:
        """Generate multiple images with logging and local download"""
        results = []
        successful_generations = 0
        successful_downloads = 0
        
        print(f"Starting batch generation of {num_images} cardboard bundle images...")
        print(f"Images will be saved to: {self.download_dir.absolute()}")
        
        log_path = self.download_dir / output_file
        with open(log_path, 'w') as log_file:
            log_file.write(f"Cardboard Bundle Synthetic Generation Log - {datetime.now()}\n")
            log_file.write("="*60 + "\n\n")
            
            for i in range(num_images):
                print(f"Generating image {i+1}/{num_images}...")
                
                prompt = self.generate_prompt()
                log_file.write(f"Image {i+1}:\n")
                log_file.write(f"Prompt: {prompt}\n")
                
                result = self.generate_image(prompt, i+1)
                
                if result:
                    successful_generations += 1
                    results.append({
                        'index': i+1,
                        'prompt': prompt,
                        'result': result
                    })
                    
                    if 'image_url' in result:
                        log_file.write(f"Status: SUCCESS - URL: {result['image_url']}\n")
                        if 'local_path' in result:
                            log_file.write(f"Local Path: {result['local_path']}\n")
                            successful_downloads += 1
                            print(f"✓ Generated and downloaded: {result['local_path']}")
                        else:
                            print(f"✓ Generated but download failed: {result['image_url']}")
                    else:
                        log_file.write(f"Status: SUCCESS - No URL returned\n")
                        print("✓ Generated successfully (no URL)")
                        
                else:
                    log_file.write(f"Status: FAILED\n")
                    print("✗ Generation failed")
                
                log_file.write("-" * 40 + "\n")
                
                # Rate limiting - be respectful to the API
                if i < num_images - 1:  # Don't sleep after the last request
                    time.sleep(2)
        
        print(f"\nGeneration complete: {successful_generations}/{num_images} successful")
        print(f"Downloads complete: {successful_downloads}/{successful_generations} downloaded")
        return results

    def create_specialized_prompts(self) -> List[str]:
        """Create specialized prompts for specific training scenarios"""
        specialized_prompts = [
            # High contrast scenarios
            "Industrial warehouse with corrugated cardboard bundles on pallets, side view at 7 degrees, bundles with stark white shipping labels against dark brown cardboard, bright LED lighting creating clear shadows, concrete floor",
            
            # Dense packing
            "Densely packed cardboard bundle warehouse, side angle at 6 degrees, multiple stacks tightly arranged but clearly separated, varying bundle heights, natural warehouse lighting, industrial setting",
            
            # Minimal scenario
            "Clean warehouse with 3 cardboard bundle stacks on wooden pallets, side view at 8 degrees, minimal background, bright even lighting, smooth concrete floor, professional product photography",
            
            # Real-world messy
            "Working warehouse environment, cardboard bundles on pallets with some plastic wrapping, side view at 5 degrees, forklift tracks on floor, mixed lighting conditions, industrial reality",
            
            # Label focus
            "Cardboard bundle storage with prominent barcode shipping labels, side perspective at 9 degrees, labels clearly visible on multiple bundles, warehouse shelving background, good lighting for label reading",
            
            # Texture emphasis
            "Corrugated cardboard bundles showing detailed cardboard texture, side view at 10 degrees, varying cardboard colors and textures, some wear and handling marks, realistic warehouse conditions"
        ]
        return specialized_prompts

    def generate_specialized_set(self) -> List[Dict]:
        """Generate the specialized prompt set with local download"""
        prompts = self.create_specialized_prompts()
        results = []
        successful_downloads = 0
        
        print(f"Generating {len(prompts)} specialized cardboard images...")
        print(f"Images will be saved to: {self.download_dir.absolute()}")
        
        for i, prompt in enumerate(prompts):
            print(f"Generating specialized image {i+1}/{len(prompts)}...")
            result = self.generate_image(prompt, f"specialized_{i+1}")
            
            if result:
                results.append({
                    'index': i+1,
                    'prompt': prompt,
                    'result': result,
                    'type': 'specialized'
                })
                if 'local_path' in result:
                    successful_downloads += 1
                    print(f"✓ Specialized image generated and downloaded")
                else:
                    print(f"✓ Specialized image generated (download failed)")
            else:
                print(f"✗ Specialized generation failed")
            
            time.sleep(2)
        
        print(f"Specialized downloads: {successful_downloads}/{len(results)} downloaded")
        return results

def main():
    # Configuration - Fixed project URL format
    ROBOFLOW_API_KEY = "yiXU9DZrZTOcO2taUh0g"
    PROJECT_URL = "boxes-jdugd/synthetic_finished_cardboard_b-tflkf"  # Fixed: Just the project identifier
    DOWNLOAD_DIR = "generate image folder"  # Local directory for downloaded images
    
    if not ROBOFLOW_API_KEY:
        print("Error: Please set ROBOFLOW_API_KEY environment variable")
        return
    
    if not PROJECT_URL:
        print("Error: Please set ROBOFLOW_PROJECT_URL environment variable")
        return
    
    # Initialize generator with download directory
    generator = CardboardSyntheticGenerator(ROBOFLOW_API_KEY, PROJECT_URL, DOWNLOAD_DIR)
    
    # Generate images
    print("Cardboard Bundle Synthetic Data Generator")
    print("="*50)
    
    choice = input("Choose generation mode:\n1. Batch generate (random variations)\n2. Specialized prompts\n3. Both\nChoice (1/2/3): ")
    
    if choice in ['1', '3']:
        num_images = int(input("Number of random variations to generate: "))
        batch_results = generator.batch_generate(num_images)
        print(f"\nBatch generation complete: {len(batch_results)} images generated")
    
    if choice in ['2', '3']:
        specialized_results = generator.generate_specialized_set()
        print(f"\nSpecialized generation complete: {len(specialized_results)} images generated")
    
    print("\nGeneration complete! Check your Roboflow project for the new synthetic images.")
    print("Remember to:")
    print("1. Review and annotate the generated images")
    print("2. Remove any poor-quality generations")
    print("3. Export your dataset in YOLO format")
    print("4. Train your model with the synthetic + real data mix")

if __name__ == "__main__":
    main()