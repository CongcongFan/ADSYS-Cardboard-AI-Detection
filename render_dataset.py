import bpy
import os
import random
import math
from mathutils import Vector, Euler

def setup_output_directory(base_path="C:/Users/76135/Desktop/ADSYS-Cardboard-AI-Detection/generated_dataset"):
    """Setup output directory for rendered images."""
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    images_dir = os.path.join(base_path, "images")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    
    return images_dir

def randomize_camera_position():
    """Randomize camera position for dataset variety."""
    camera = bpy.context.scene.camera
    
    # Random distance from center
    distance = random.uniform(3, 8)
    
    # Random angles
    angle_h = random.uniform(0, 2 * math.pi)  # Horizontal angle
    angle_v = random.uniform(0.2, 1.2)  # Vertical angle (elevation)
    
    # Calculate position
    x = distance * math.cos(angle_h) * math.cos(angle_v)
    y = distance * math.sin(angle_h) * math.cos(angle_v)
    z = distance * math.sin(angle_v) + random.uniform(0.5, 2.0)
    
    camera.location = (x, y, z)
    
    # Add some camera rotation variation
    camera.rotation_euler = (
        angle_v + random.uniform(-0.1, 0.1),
        0,
        angle_h + random.uniform(-0.1, 0.1)
    )

def randomize_lighting():
    """Randomize lighting conditions for dataset variety."""
    # Find sun light
    sun = None
    for obj in bpy.context.scene.objects:
        if obj.type == 'LIGHT' and obj.data.type == 'SUN':
            sun = obj
            break
    
    if sun:
        # Randomize sun energy and angle
        sun.data.energy = random.uniform(2.0, 5.0)
        sun.rotation_euler = (
            random.uniform(0.3, 1.2),  # Elevation
            0,
            random.uniform(0, 2 * math.pi)  # Azimuth
        )
    
    # Randomize area lights
    for obj in bpy.context.scene.objects:
        if obj.type == 'LIGHT' and obj.data.type == 'AREA':
            obj.data.energy = random.uniform(0.5, 2.0)

def add_random_background():
    """Add random background variations."""
    # Create world material with random HDRI-like environment
    world = bpy.context.scene.world
    if not world.use_nodes:
        world.use_nodes = True
    
    nodes = world.node_tree.nodes
    nodes.clear()
    
    # Create nodes
    output = nodes.new(type='ShaderNodeOutputWorld')
    background = nodes.new(type='ShaderNodeBackground')
    
    # Random background color (concrete/warehouse-like)
    background_colors = [
        (0.4, 0.4, 0.45),  # Concrete gray
        (0.35, 0.35, 0.4),  # Dark concrete
        (0.5, 0.5, 0.55),   # Light concrete
        (0.45, 0.4, 0.35),  # Brownish concrete
    ]
    
    color = random.choice(background_colors)
    background.inputs['Color'].default_value = (*color, 1.0)
    background.inputs['Strength'].default_value = random.uniform(0.3, 0.8)
    
    # Connect nodes
    world.node_tree.links.new(background.outputs['Background'], output.inputs['Surface'])

def randomize_pallet_positions():
    """Randomly adjust pallet positions and rotations."""
    cardboard_objects = [obj for obj in bpy.context.scene.objects 
                        if obj.type == 'MESH' and 'CardboardSheet' in obj.name]
    
    # Group by pallet (assuming sheets are created in order)
    pallets = {}
    for obj in cardboard_objects:
        # Extract pallet number from name (assuming naming pattern)
        base_name = obj.name.split('_')[0] + '_1'  # Get base sheet name
        if base_name not in pallets:
            pallets[base_name] = []
        pallets[base_name].append(obj)
    
    # Randomly adjust each pallet as a group
    for pallet_sheets in pallets.values():
        # Random base rotation for entire pallet
        rotation_z = random.uniform(-0.3, 0.3)
        
        # Random base position offset
        offset_x = random.uniform(-0.5, 0.5)
        offset_y = random.uniform(-0.5, 0.5)
        
        for sheet in pallet_sheets:
            sheet.rotation_euler[2] += rotation_z
            sheet.location[0] += offset_x
            sheet.location[1] += offset_y

def render_single_image(output_path, frame_number):
    """Render a single image with current scene configuration."""
    scene = bpy.context.scene
    
    # Set output path
    scene.render.filepath = os.path.join(output_path, f"cardboard_pallet_{frame_number:04d}.png")
    
    # Render
    bpy.ops.render.render(write_still=True)
    
    print(f"Rendered frame {frame_number}")

def generate_dataset(num_images=100, output_dir=None):
    """Generate a dataset of cardboard pallet images."""
    if output_dir is None:
        output_dir = setup_output_directory()
    
    print(f"Generating {num_images} images for YOLO training dataset...")
    
    for i in range(num_images):
        print(f"Generating image {i+1}/{num_images}")
        
        # Randomize scene
        randomize_camera_position()
        randomize_lighting()
        add_random_background()
        
        if i % 10 == 0:  # Randomize pallet positions every 10 images
            randomize_pallet_positions()
        
        # Render image
        render_single_image(output_dir, i+1)
    
    print(f"Dataset generation complete! Images saved to: {output_dir}")

def create_annotation_info():
    """Create information file for YOLO annotation."""
    info_content = """# Cardboard Pallet Dataset Information

## Classes for YOLO Training:
1. cardboard_pallet - Individual stacked cardboard pallets
2. cardboard_sheet - Individual cardboard sheets within pallets  
3. shipping_label - White shipping labels on pallets

## Dataset Details:
- Images: 1920x1080 resolution
- Format: PNG
- Lighting: Varied warehouse/industrial lighting conditions
- Backgrounds: Concrete/warehouse environments
- Camera angles: Multiple viewing angles and distances

## Recommended YOLO Configuration:
- Model: YOLO11-seg for segmentation
- Input size: 640px
- Confidence threshold: 0.25
- IoU threshold: 0.20

## Annotation Guidelines:
- Annotate each visible cardboard pallet as a single object
- Include partially visible pallets at image edges
- Mark individual sheets only if clearly separable
- Include shipping labels as separate objects

## Generated Variations:
- Pallet heights: 8-16 sheets per pallet
- Sheet orientations: Slight rotational variations
- Stack alignments: Natural stacking imperfections
- Label placements: Random positions on pallets
- Lighting conditions: Multiple angles and intensities
"""
    
    output_path = "C:/Users/76135/Desktop/ADSYS-Cardboard-AI-Detection/generated_dataset/dataset_info.txt"
    with open(output_path, 'w') as f:
        f.write(info_content)
    
    print(f"Dataset info saved to: {output_path}")

if __name__ == "__main__":
    # Generate dataset
    generate_dataset(num_images=50)  # Start with 50 images for testing
    create_annotation_info()