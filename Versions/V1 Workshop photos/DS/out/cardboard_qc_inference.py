#!/usr/bin/env python3
"""
Production inference script for cardboard quality control
Uses the fine-tuned Qwen2.5-VL model to assess cardboard bundle quality.
"""

import torch
from unsloth import FastVisionModel
from PIL import Image
import argparse
import os
from pathlib import Path
import json
from datetime import datetime

class CardboardQCModel:
    def __init__(self, model_path="cardboard_qc_lora"):
        """Initialize the cardboard QC model."""
        print(f"🔧 Loading cardboard QC model from {model_path}...")
        
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_name=model_path,
            load_in_4bit=True,
        )
        FastVisionModel.for_inference(self.model)
        
        self.instruction = "Analyze this cardboard bundle image. Is the bundle flat or warped? Provide your assessment."
        print("✅ Model loaded and ready for inference!")
    
    def predict(self, image_path_or_pil, return_raw_response=False):
        """
        Predict quality of a cardboard bundle.
        
        Args:
            image_path_or_pil: Path to image file or PIL Image
            return_raw_response: If True, return full model response
            
        Returns:
            dict: Prediction results with label, confidence, and reasoning
        """
        # Load image
        if isinstance(image_path_or_pil, (str, Path)):
            image = Image.open(image_path_or_pil)
            filename = os.path.basename(image_path_or_pil)
        else:
            image = image_path_or_pil
            filename = "image"
        
        # Prepare input
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": self.instruction}
            ]}
        ]
        
        input_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda")
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                use_cache=True,
                temperature=0.3,  # Lower for more consistent results
                min_p=0.05,
                do_sample=True
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract generated part
        prompt_len = len(self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True))
        generated_text = response[prompt_len:].strip()
        
        # Parse prediction
        result = self._parse_response(generated_text, filename)
        
        if return_raw_response:
            result['raw_response'] = generated_text
            
        return result
    
    def _parse_response(self, response_text, filename):
        """Parse the model response to extract prediction."""
        response_lower = response_text.lower()
        
        # Determine label with confidence
        confidence_score = 0.5  # Default neutral confidence
        
        # Check for clear Pass indicators
        pass_indicators = ['pass', 'flat', 'good', 'acceptable', 'appears flat', 'looks flat']
        fail_indicators = ['fail', 'warp', 'bad', 'unacceptable', 'appears warped', 'looks warped']
        
        pass_count = sum(1 for indicator in pass_indicators if indicator in response_lower)
        fail_count = sum(1 for indicator in fail_indicators if indicator in response_lower)
        
        if pass_count > fail_count:
            label = "Pass"
            confidence_score = min(0.9, 0.5 + (pass_count - fail_count) * 0.1)
        elif fail_count > pass_count:
            label = "Fail"
            confidence_score = min(0.9, 0.5 + (fail_count - pass_count) * 0.1)
        else:
            label = "Uncertain"
            confidence_score = 0.5
        
        # Extract reasoning (try to find the explanation part)
        reasoning = response_text
        if '.' in reasoning:
            # Take the first complete sentence as reasoning
            sentences = reasoning.split('.')
            reasoning = sentences[0].strip() + '.'
        
        return {
            'filename': filename,
            'label': label,
            'confidence': round(confidence_score, 3),
            'reasoning': reasoning,
            'timestamp': datetime.now().isoformat()
        }
    
    def batch_predict(self, image_folder, output_file=None):
        """
        Process all images in a folder.
        
        Args:
            image_folder: Path to folder containing images
            output_file: Optional path to save results JSON
            
        Returns:
            list: Results for all images
        """
        image_folder = Path(image_folder)
        results = []
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in image_folder.glob('*') 
                      if f.suffix.lower() in image_extensions]
        
        print(f"🔍 Found {len(image_files)} images in {image_folder}")
        
        # Process each image
        for image_file in image_files:
            try:
                print(f"📸 Processing: {image_file.name}")
                result = self.predict(image_file)
                results.append(result)
                print(f"   Result: {result['label']} (confidence: {result['confidence']})")
                
            except Exception as e:
                print(f"❌ Error processing {image_file.name}: {e}")
                results.append({
                    'filename': image_file.name,
                    'label': 'Error',
                    'confidence': 0.0,
                    'reasoning': f'Processing error: {str(e)}',
                    'timestamp': datetime.now().isoformat()
                })
        
        # Save results if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"💾 Results saved to {output_file}")
        
        # Print summary
        pass_count = sum(1 for r in results if r['label'] == 'Pass')
        fail_count = sum(1 for r in results if r['label'] == 'Fail')
        print(f"\n📊 Batch processing summary:")
        print(f"   Total images: {len(results)}")
        print(f"   Pass: {pass_count}")
        print(f"   Fail: {fail_count}")
        print(f"   Pass rate: {pass_count/len(results)*100:.1f}%")
        
        return results

def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description="Cardboard Quality Control Inference")
    parser.add_argument("--model_path", default="cardboard_qc_lora",
                        help="Path to the fine-tuned model")
    parser.add_argument("--image", help="Path to single image to analyze")
    parser.add_argument("--folder", help="Path to folder of images to analyze")
    parser.add_argument("--output", help="Output JSON file for results")
    parser.add_argument("--raw", action="store_true", 
                        help="Include raw model response in output")
    
    args = parser.parse_args()
    
    # Initialize model
    try:
        qc_model = CardboardQCModel(args.model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    if args.image:
        # Single image prediction
        print(f"🔍 Analyzing single image: {args.image}")
        result = qc_model.predict(args.image, return_raw_response=args.raw)
        
        print(f"\n📋 Analysis Results:")
        print(f"   File: {result['filename']}")
        print(f"   Label: {result['label']}")
        print(f"   Confidence: {result['confidence']}")
        print(f"   Reasoning: {result['reasoning']}")
        
        if args.raw:
            print(f"   Raw Response: {result['raw_response']}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"💾 Results saved to {args.output}")
            
    elif args.folder:
        # Batch processing
        print(f"📁 Analyzing folder: {args.folder}")
        results = qc_model.batch_predict(args.folder, args.output)
        
    else:
        print("❌ Please specify either --image or --folder")
        print("Example usage:")
        print("  python cardboard_qc_inference.py --image path/to/image.jpg")
        print("  python cardboard_qc_inference.py --folder path/to/images/ --output results.json")

if __name__ == "__main__":
    main()