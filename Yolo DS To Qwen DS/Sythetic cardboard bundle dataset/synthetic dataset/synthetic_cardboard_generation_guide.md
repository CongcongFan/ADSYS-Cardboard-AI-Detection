# Synthetic Cardboard Bundle Generation for YOLO Training

## Overview
This guide teaches you how to generate synthetic corrugated cardboard bundle images using DALL-E 3 via Roboflow's API for YOLO training datasets.

## Prerequisites
- OpenAI API key (for DALL-E 3)
- Roboflow account and API key
- Active Roboflow project

## Setup

### 1. Roboflow Project Setup
```bash
# Create new project at roboflow.com
# Note your project URL: https://universe.roboflow.com/your-workspace/your-project
# Get your API key from account settings
```

### 2. Environment Setup
```bash
pip install requests pillow opencv-python
export ROBOFLOW_API_KEY="your_api_key_here"
```

## Prompt Engineering for Cardboard Bundles

### Core Base Prompt
```
"Industrial warehouse scene showing stacks of corrugated cardboard bundles arranged on wooden pallets, viewed from a side angle at 5-10 degrees tilt, showing multiple distinct bundle stacks with visible separation between each bundle group, natural warehouse lighting, concrete floor, industrial setting"
```

### Variations for Dataset Diversity

#### 1. Lighting Variations
- "Bright fluorescent warehouse lighting"
- "Natural daylight through warehouse skylights" 
- "Overhead LED lighting with shadows"
- "Mixed lighting conditions"

#### 2. Bundle Configuration
- "3-4 separate cardboard bundle stacks"
- "Multiple rows of cardboard bundles"
- "Densely packed cardboard inventory"
- "Sparse arrangement with clear separations"

#### 3. Visual Differentiation
- "Bundles with white shipping labels visible"
- "Different cardboard shades from light brown to dark brown"
- "Some bundles wrapped in plastic, others exposed"
- "Bundles of varying heights creating visual separation"

#### 4. Camera Angle Variations
- "Side view at 5 degree angle"
- "Side view at 8 degree angle" 
- "Side view at 10 degree angle"
- "Slightly elevated side perspective"

#### 5. Industrial Context
- "Forklift visible in background"
- "Warehouse shelving in background"
- "Industrial floor markings"
- "Safety signage visible on walls"

## Complete Prompt Templates

### Template 1: Clean Industrial
```
"Professional warehouse photography showing corrugated cardboard bundles stacked on wooden pallets, side view at 7 degree angle, 4 distinct bundle groups with clear visual separation, bright LED warehouse lighting, smooth concrete floor, industrial setting, high quality product photography style"
```

### Template 2: Realistic Working Environment
```
"Industrial warehouse scene with multiple stacks of brown corrugated cardboard bundles on pallets, viewed from side at 6 degree angle, some bundles with white shipping labels, varying cardboard tones, forklift tracks on concrete floor, natural warehouse lighting with shadows"
```

### Template 3: Density Variation
```
"Dense arrangement of corrugated cardboard inventory in warehouse, side perspective at 8 degrees, multiple bundle stacks with height variations creating natural separation, mixed fluorescent and natural lighting, industrial concrete floor"
```

### Template 4: Label Focus
```
"Cardboard bundle warehouse storage showing distinct stacks with prominent white shipping labels, side angle view at 5 degrees, clear separation between bundle groups, professional warehouse lighting, clean industrial environment"
```

## API Implementation

### Basic Generation Script
```python
import requests
import json
import os

def generate_synthetic_cardboard(prompt, project_url, api_key):
    url = f"https://api.roboflow.com/synthetic-image?api_key={api_key}"
    
    payload = {
        "project_url": project_url,
        "prompt": prompt
    }
    
    headers = {
        'Content-Type': 'application/json'
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

# Usage
project_url = "https://universe.roboflow.com/your-workspace/your-project"
api_key = os.environ.get("ROBOFLOW_API_KEY")

prompt = "Industrial warehouse scene showing stacks of corrugated cardboard bundles arranged on wooden pallets, viewed from a side angle at 7 degrees tilt, showing 4 distinct bundle stacks with visible separation between each bundle group, bright LED warehouse lighting, concrete floor"

result = generate_synthetic_cardboard(prompt, project_url, api_key)
```

## Dataset Generation Strategy

### 1. Create Prompt Variations
Generate 50-100 images using different combinations of:
- 5 lighting conditions
- 4 camera angles (5°, 7°, 8°, 10°)
- 3 bundle densities
- 3 background variations
- 2 label visibility options

### 2. Batch Generation
```python
prompts = [
    # Your template variations here
]

for i, prompt in enumerate(prompts):
    print(f"Generating image {i+1}/{len(prompts)}")
    result = generate_synthetic_cardboard(prompt, project_url, api_key)
    if result:
        print(f"Generated: {result.get('image_url', 'No URL returned')}")
    time.sleep(2)  # Rate limiting
```

### 3. Quality Control
- Review generated images for:
  - Clear bundle separation
  - Appropriate side angle (5-10°)
  - Industrial warehouse setting
  - Realistic cardboard textures
  - Proper lighting and shadows

## Annotation Strategy

### Auto-labeling Setup
1. Upload synthetic images to Roboflow
2. Use Roboflow's auto-labeling features
3. Manual review and correction
4. Export in YOLO format

### Class Definitions
```yaml
classes:
  0: cardboard_bundle
  1: pallet
  2: shipping_label
```

### Bounding Box Guidelines
- Each distinct cardboard stack = separate bounding box
- Include full height and width of each bundle
- Ensure separation between adjacent bundles
- Label pallets separately if visible

## Integration with Existing Dataset

### Mixing Strategy
1. Start with 70% synthetic, 30% real images
2. Gradually shift to 30% synthetic, 70% real
3. Use synthetic data for rare scenarios
4. Tag synthetic images for easy removal

### Quality Validation
```python
# Validate synthetic images match real image characteristics
def validate_synthetic_quality(image_path):
    # Check image dimensions
    # Verify angle range
    # Analyze bundle separation
    # Confirm industrial setting
    pass
```

## Best Practices

### Prompt Engineering
- Be specific about viewing angle (5-10 degrees)
- Always mention "distinct bundle stacks"
- Include "visible separation" in prompts
- Specify industrial warehouse context
- Mention concrete floors and pallets

### Dataset Balance
- Generate multiple views of same scenario
- Vary lighting conditions significantly
- Include edge cases (single bundles, crowded scenes)
- Maintain consistent industrial aesthetic

### Quality Assurance
- Manual review of all generated images
- Remove unrealistic or poor-quality generations
- Ensure proper YOLO annotation accuracy
- Test model performance regularly

## Troubleshooting

### Common Issues
- **Bundles too similar**: Add more texture/color variation prompts
- **Poor separation**: Emphasize "distinct stacks" and "clear separation"
- **Wrong angle**: Be more specific about degree measurements
- **Unrealistic lighting**: Reference actual warehouse lighting conditions

### Prompt Refinements
If results don't match requirements:
1. Add more specific industrial details
2. Reference actual cardboard manufacturing terms
3. Include more precise camera positioning
4. Emphasize the corrugated texture visibility

## Monitoring and Iteration
- Track model performance on synthetic vs. real data
- Continuously refine prompts based on results
- Gradually reduce synthetic data percentage
- Document which prompts produce best results for future use