"""
DALL-E 3 Cardboard Bundle Generator for YOLO Training
Generates realistic cardboard bundle images using OpenAI's DALL-E 3
"""

import requests
import json
import os
import time
import random
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path
import base64

class DalleCardboardGenerator:
    def __init__(self, openai_api_key: str, download_dir: str = "generate image folder"):
        self.api_key = openai_api_key
        self.api_endpoint = "https://api.openai.com/v1/images/generations"
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
        {features}, {lighting}, {context}, professional warehouse photography style, high quality, realistic"""
        
        # Clean up formatting
        prompt = " ".join(prompt.split())
        return prompt

    def download_image(self, image_url: str, filename: str, max_retries: int = 3) -> bool:
        """Download image from URL to local directory with retry logic"""
        if not image_url or not image_url.startswith(('http://', 'https://')):
            print(f"✗ Invalid URL: {image_url}")
            return False
            
        for attempt in range(max_retries):
            try:
                response = requests.get(image_url, stream=True, timeout=30)
                
                if response.status_code == 200:
                    file_path = self.download_dir / filename
                    
                    with open(file_path, 'wb') as f:
                        total_size = 0
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                total_size += len(chunk)
                    
                    if total_size > 0:
                        print(f"✓ Downloaded: {filename} ({total_size:,} bytes)")
                        return True
                    else:
                        print(f"✗ Downloaded file is empty: {filename}")
                        return False
                        
                elif response.status_code == 404:
                    print(f"✗ Image not found (404): {image_url}")
                    return False
                else:
                    print(f"✗ HTTP {response.status_code} on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        
            except requests.exceptions.Timeout:
                print(f"✗ Timeout on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
            except Exception as e:
                print(f"✗ Download error on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        print(f"✗ Failed to download after {max_retries} attempts")
        return False
    
    def generate_image(self, prompt: str, image_index: int = 0) -> Optional[Dict]:
        """Generate a single image using DALL-E 3"""
        payload = {
            "model": "dall-e-3",
            "prompt": prompt,
            "n": 1,
            "size": "1024x1024",
            "quality": "standard",
            "style": "natural"
        }
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        
        try:
            print(f"Generating image with DALL-E 3...")
            response = requests.post(
                self.api_endpoint, 
                headers=headers, 
                data=json.dumps(payload),
                timeout=120  # DALL-E can take longer
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if 'data' in result and len(result['data']) > 0:
                    image_data = result['data'][0]
                    image_url = image_data.get('url')
                    
                    if image_url:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"cardboard_dalle_{timestamp}_{image_index:03d}.png"
                        
                        if self.download_image(image_url, filename):
                            return {
                                'success': True,
                                'image_url': image_url,
                                'local_path': str(self.download_dir / filename),
                                'prompt': prompt,
                                'revised_prompt': image_data.get('revised_prompt', prompt)
                            }
                        else:
                            return {
                                'success': False,
                                'error': 'Download failed',
                                'image_url': image_url,
                                'prompt': prompt
                            }
                    else:
                        return {
                            'success': False,
                            'error': 'No image URL in response',
                            'prompt': prompt
                        }
                else:
                    return {
                        'success': False,
                        'error': 'No image data in response',
                        'prompt': prompt
                    }
            else:
                error_msg = f"API Error {response.status_code}"
                try:
                    error_data = response.json()
                    if 'error' in error_data:
                        error_msg += f": {error_data['error'].get('message', 'Unknown error')}"
                except:
                    error_msg += f": {response.text}"
                
                print(f"✗ {error_msg}")
                return {
                    'success': False,
                    'error': error_msg,
                    'prompt': prompt
                }
                
        except requests.exceptions.Timeout:
            error_msg = "Request timed out"
            print(f"✗ {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'prompt': prompt
            }
        except Exception as e:
            error_msg = f"Request failed: {str(e)}"
            print(f"✗ {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'prompt': prompt
            }

    def batch_generate(self, num_images: int) -> List[Dict]:
        """Generate multiple images with logging"""
        results = []
        successful_generations = 0
        successful_downloads = 0
        
        print(f"Starting DALL-E 3 generation of {num_images} cardboard bundle images...")
        print(f"Images will be saved to: {self.download_dir.absolute()}")
        
        for i in range(num_images):
            print(f"\nGenerating image {i+1}/{num_images}...")
            
            prompt = self.generate_prompt()
            print(f"Prompt: {prompt[:100]}...")
            
            result = self.generate_image(prompt, i+1)
            results.append(result)
            
            if result and result.get('success'):
                successful_generations += 1
                if 'local_path' in result:
                    successful_downloads += 1
                    print(f"✓ Successfully generated and downloaded image {i+1}")
                else:
                    print(f"✓ Generated but download failed for image {i+1}")
            else:
                print(f"✗ Failed to generate image {i+1}: {result.get('error', 'Unknown error')}")
            
            # Rate limiting - DALL-E has usage limits
            if i < num_images - 1:  # Don't sleep after the last request
                print("Waiting 3 seconds (rate limiting)...")
                time.sleep(3)
        
        print(f"\n" + "="*50)
        print(f"Generation Summary:")
        print(f"Successful generations: {successful_generations}/{num_images}")
        print(f"Successful downloads: {successful_downloads}/{successful_generations}")
        print(f"Download directory: {self.download_dir.absolute()}")
        
        return results

    def create_specialized_prompts(self) -> List[str]:
        """Create specialized prompts for specific training scenarios"""
        specialized_prompts = [
            # High contrast scenarios
            "Industrial warehouse with corrugated cardboard bundles on wooden pallets, side view at 7 degrees, bundles with stark white shipping labels against dark brown cardboard, bright LED lighting creating clear shadows, concrete floor, realistic warehouse photography",
            
            # Dense packing
            "Densely packed cardboard bundle warehouse, side angle at 6 degrees, multiple stacks tightly arranged but clearly separated, varying bundle heights, natural warehouse lighting, industrial setting, realistic photography",
            
            # Minimal scenario
            "Clean warehouse with 3 cardboard bundle stacks on wooden pallets, side view at 8 degrees, minimal background, bright even lighting, smooth concrete floor, professional product photography, realistic",
            
            # Real-world messy
            "Working warehouse environment, cardboard bundles on pallets with some plastic wrapping, side view at 5 degrees, forklift tracks on floor, mixed lighting conditions, industrial reality, realistic photography",
            
            # Label focus
            "Cardboard bundle storage with prominent barcode shipping labels, side perspective at 9 degrees, labels clearly visible on multiple bundles, warehouse shelving background, good lighting for label reading, realistic",
            
            # Texture emphasis
            "Corrugated cardboard bundles showing detailed cardboard texture, side view at 10 degrees, varying cardboard colors and textures, some wear and handling marks, realistic warehouse conditions, high detail photography"
        ]
        return specialized_prompts

    def generate_specialized_set(self) -> List[Dict]:
        """Generate the specialized prompt set"""
        prompts = self.create_specialized_prompts()
        results = []
        successful_downloads = 0
        
        print(f"Generating {len(prompts)} specialized cardboard images with DALL-E 3...")
        print(f"Images will be saved to: {self.download_dir.absolute()}")
        
        for i, prompt in enumerate(prompts):
            print(f"\nGenerating specialized image {i+1}/{len(prompts)}...")
            print(f"Prompt: {prompt[:100]}...")
            
            result = self.generate_image(prompt, f"specialized_{i+1}")
            
            if result and result.get('success'):
                results.append({
                    'index': i+1,
                    'prompt': prompt,
                    'result': result,
                    'type': 'specialized'
                })
                if 'local_path' in result:
                    successful_downloads += 1
                    print(f"✓ Specialized image {i+1} generated and downloaded")
                else:
                    print(f"✓ Specialized image {i+1} generated (download failed)")
            else:
                results.append({
                    'index': i+1,
                    'prompt': prompt,
                    'result': result,
                    'type': 'specialized',
                    'failed': True
                })
                print(f"✗ Specialized generation {i+1} failed: {result.get('error', 'Unknown error')}")
            
            # Rate limiting
            if i < len(prompts) - 1:
                print("Waiting 3 seconds (rate limiting)...")
                time.sleep(3)
        
        print(f"\n" + "="*50)
        print(f"Specialized generation complete:")
        print(f"Successful downloads: {successful_downloads}/{len(results)}")
        return results

def main():
    # Configuration
    OPENAI_API_KEY = input("Enter your OpenAI API key: ").strip()
    DOWNLOAD_DIR = "generate image folder"  # Local directory for downloaded images
    
    if not OPENAI_API_KEY:
        print("Error: OpenAI API key is required")
        print("Get your API key from: https://platform.openai.com/api-keys")
        return
    
    # Initialize generator
    generator = DalleCardboardGenerator(OPENAI_API_KEY, DOWNLOAD_DIR)
    
    print("DALL-E 3 Cardboard Bundle Generator")
    print("="*50)
    print("This will generate realistic cardboard bundle images for YOLO training")
    print("Note: DALL-E 3 costs $0.040 per image (1024x1024)")
    
    # Generate images
    choice = input("\nChoose generation mode:\n1. Batch generate (random variations)\n2. Specialized prompts\n3. Both\nChoice (1/2/3): ")
    
    total_cost = 0
    
    if choice in ['1', '3']:
        num_images = int(input("Number of random variations to generate: "))
        estimated_cost = num_images * 0.040
        print(f"Estimated cost: ${estimated_cost:.2f}")
        
        confirm = input("Continue? (y/n): ").lower()
        if confirm.startswith('y'):
            batch_results = generator.batch_generate(num_images)
            successful = sum(1 for r in batch_results if r.get('success'))
            actual_cost = successful * 0.040
            total_cost += actual_cost
            print(f"Actual cost for batch: ${actual_cost:.2f}")
    
    if choice in ['2', '3']:
        specialized_count = 6  # Number of specialized prompts
        estimated_cost = specialized_count * 0.040
        print(f"Specialized set estimated cost: ${estimated_cost:.2f}")
        
        confirm = input("Generate specialized set? (y/n): ").lower()
        if confirm.startswith('y'):
            specialized_results = generator.generate_specialized_set()
            successful = sum(1 for r in specialized_results if r.get('result', {}).get('success'))
            actual_cost = successful * 0.040
            total_cost += actual_cost
            print(f"Actual cost for specialized: ${actual_cost:.2f}")
    
    print(f"\nTotal cost: ${total_cost:.2f}")
    print(f"\nGeneration complete! Images saved to: {Path(DOWNLOAD_DIR).absolute()}")
    print("\nNext steps:")
    print("1. Review the generated images")
    print("2. Upload to your dataset manually")
    print("3. Annotate the cardboard bundles")
    print("4. Export in YOLO format")
    print("5. Train your model")

if __name__ == "__main__":
    main()