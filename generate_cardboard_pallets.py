import bpy
import bmesh
import random
import math
from mathutils import Vector

def clear_scene():
    """Clear all mesh objects from the scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False, confirm=False)

def create_cardboard_sheet(width=0.6, height=0.4, thickness=0.003, name="CardboardSheet"):
    """Create a single cardboard sheet with realistic proportions."""
    bpy.ops.mesh.primitive_cube_add(size=2)
    sheet = bpy.context.active_object
    sheet.name = name
    
    # Scale to cardboard dimensions
    sheet.scale = (width/2, height/2, thickness/2)
    bpy.ops.object.transform_apply(scale=True)
    
    return sheet

def create_cardboard_material():
    """Create realistic cardboard material."""
    mat = bpy.data.materials.new(name="CardboardMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    
    # Clear default nodes
    nodes.clear()
    
    # Create nodes
    output = nodes.new(type='ShaderNodeOutputMaterial')
    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
    noise = nodes.new(type='ShaderNodeTexNoise')
    colorramp = nodes.new(type='ShaderNodeValToRGB')
    mapping = nodes.new(type='ShaderNodeMapping')
    texcoord = nodes.new(type='ShaderNodeTexCoord')
    
    # Position nodes
    output.location = (400, 0)
    principled.location = (200, 0)
    colorramp.location = (0, 0)
    noise.location = (-200, 0)
    mapping.location = (-400, 0)
    texcoord.location = (-600, 0)
    
    # Configure noise for cardboard texture
    noise.inputs['Scale'].default_value = 15.0
    noise.inputs['Detail'].default_value = 8.0
    noise.inputs['Roughness'].default_value = 0.6
    
    # Configure color ramp for cardboard colors
    colorramp.color_ramp.elements[0].color = (0.6, 0.45, 0.25, 1.0)  # Dark cardboard
    colorramp.color_ramp.elements[1].color = (0.8, 0.65, 0.45, 1.0)  # Light cardboard
    
    # Configure principled shader
    principled.inputs['Roughness'].default_value = 0.8
    principled.inputs['Specular'].default_value = 0.1
    
    # Connect nodes
    mat.node_tree.links.new(texcoord.outputs['Generated'], mapping.inputs['Vector'])
    mat.node_tree.links.new(mapping.outputs['Vector'], noise.inputs['Vector'])
    mat.node_tree.links.new(noise.outputs['Fac'], colorramp.inputs['Fac'])
    mat.node_tree.links.new(colorramp.outputs['Color'], principled.inputs['Base Color'])
    mat.node_tree.links.new(principled.outputs['BSDF'], output.inputs['Surface'])
    
    return mat

def create_label_material():
    """Create white label material."""
    mat = bpy.data.materials.new(name="LabelMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    
    # Clear default nodes
    nodes.clear()
    
    # Create nodes
    output = nodes.new(type='ShaderNodeOutputMaterial')
    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
    
    # Position nodes
    output.location = (200, 0)
    principled.location = (0, 0)
    
    # Configure white material
    principled.inputs['Base Color'].default_value = (0.95, 0.95, 0.95, 1.0)
    principled.inputs['Roughness'].default_value = 0.3
    principled.inputs['Specular'].default_value = 0.2
    
    # Connect nodes
    mat.node_tree.links.new(principled.outputs['BSDF'], output.inputs['Surface'])
    
    return mat

def create_label(width=0.15, height=0.1, thickness=0.001):
    """Create a white shipping label."""
    bpy.ops.mesh.primitive_cube_add(size=2)
    label = bpy.context.active_object
    label.name = "ShippingLabel"
    
    # Scale to label dimensions
    label.scale = (width/2, height/2, thickness/2)
    bpy.ops.object.transform_apply(scale=True)
    
    return label

def add_corrugation_detail(obj):
    """Add subtle corrugation detail to cardboard using displacement."""
    # Enter edit mode
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    
    # Add loop cuts for detail
    bpy.ops.mesh.subdivide(number_cuts=3)
    
    # Add displacement modifier
    bpy.ops.object.mode_set(mode='OBJECT')
    displace_mod = obj.modifiers.new(name="Corrugation", type='DISPLACE')
    displace_mod.strength = 0.002
    displace_mod.mid_level = 0.5
    
    # Create texture for displacement
    texture = bpy.data.textures.new(name="CorrugationTexture", type='NOISE')
    texture.noise_scale = 0.5
    displace_mod.texture = texture

def create_pallet_stack(num_sheets=12, base_pos=(0, 0, 0), variation=True):
    """Create a stack of cardboard sheets forming a pallet."""
    cardboard_mat = create_cardboard_material()
    label_mat = create_label_material()
    
    sheets = []
    current_z = base_pos[2]
    
    for i in range(num_sheets):
        # Add some variation in sheet sizes
        if variation:
            width = random.uniform(0.55, 0.65)
            height = random.uniform(0.35, 0.45)
            thickness = random.uniform(0.002, 0.004)
        else:
            width, height, thickness = 0.6, 0.4, 0.003
        
        # Create sheet
        sheet = create_cardboard_sheet(width, height, thickness, f"CardboardSheet_{i+1}")
        
        # Position sheet
        x_offset = random.uniform(-0.02, 0.02) if variation else 0
        y_offset = random.uniform(-0.02, 0.02) if variation else 0
        rotation_z = random.uniform(-0.1, 0.1) if variation else 0
        
        sheet.location = (base_pos[0] + x_offset, base_pos[1] + y_offset, current_z + thickness/2)
        sheet.rotation_euler[2] = rotation_z
        
        # Apply material
        sheet.data.materials.append(cardboard_mat)
        
        # Add corrugation detail to some sheets
        if random.random() < 0.3:
            add_corrugation_detail(sheet)
        
        sheets.append(sheet)
        current_z += thickness
        
        # Add labels to some sheets (visible ones)
        if i % 4 == 0 and i < num_sheets - 2:  # Add labels to every 4th sheet, not top ones
            label = create_label()
            label_x = base_pos[0] + random.uniform(-0.2, 0.2)
            label_y = base_pos[1] + random.uniform(-0.15, 0.15)
            label.location = (label_x, label_y, current_z + 0.001)
            label.data.materials.append(label_mat)
    
    return sheets

def create_multiple_pallets(num_pallets=4, spacing=1.5):
    """Create multiple pallet stacks with different configurations."""
    pallets = []
    
    # Calculate grid positions
    cols = int(math.ceil(math.sqrt(num_pallets)))
    rows = int(math.ceil(num_pallets / cols))
    
    pallet_idx = 0
    for row in range(rows):
        for col in range(cols):
            if pallet_idx >= num_pallets:
                break
                
            # Calculate position
            x = (col - cols/2) * spacing
            y = (row - rows/2) * spacing
            
            # Vary number of sheets per pallet
            num_sheets = random.randint(8, 16)
            
            # Create pallet
            pallet = create_pallet_stack(
                num_sheets=num_sheets,
                base_pos=(x, y, 0),
                variation=True
            )
            pallets.append(pallet)
            pallet_idx += 1
    
    return pallets

def setup_lighting():
    """Setup realistic lighting for the scene."""
    # Clear existing lights
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if obj.type == 'LIGHT':
            obj.select_set(True)
    bpy.ops.object.delete()
    
    # Add sun light (main lighting)
    bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
    sun = bpy.context.active_object
    sun.data.energy = 3.0
    sun.rotation_euler = (0.785, 0, 0.785)  # 45-degree angle
    
    # Add area lights for fill lighting
    bpy.ops.object.light_add(type='AREA', location=(-3, 3, 5))
    area1 = bpy.context.active_object
    area1.data.energy = 1.5
    area1.data.size = 2.0
    area1.rotation_euler = (0.785, 0, -0.785)
    
    bpy.ops.object.light_add(type='AREA', location=(3, -3, 5))
    area2 = bpy.context.active_object
    area2.data.energy = 1.0
    area2.data.size = 2.0
    area2.rotation_euler = (0.785, 0, 0.785)

def setup_camera():
    """Setup camera for dataset rendering."""
    # Clear existing cameras
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if obj.type == 'CAMERA':
            obj.select_set(True)
    bpy.ops.object.delete()
    
    # Add camera
    bpy.ops.object.camera_add(location=(3, 3, 2))
    camera = bpy.context.active_object
    
    # Point camera at center
    constraint = camera.constraints.new(type='TRACK_TO')
    
    # Create empty for camera to track
    bpy.ops.object.empty_add(location=(0, 0, 0.2))
    target = bpy.context.active_object
    target.name = "CameraTarget"
    
    constraint.target = target
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'
    
    # Set as active camera
    bpy.context.scene.camera = camera

def setup_render_settings():
    """Configure render settings for dataset generation."""
    scene = bpy.context.scene
    
    # Set render engine
    scene.render.engine = 'CYCLES'
    
    # Set resolution
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    
    # Set samples for quality
    scene.cycles.samples = 128
    
    # Enable denoising
    scene.cycles.use_denoising = True

def main():
    """Main function to generate cardboard pallet scene."""
    print("Generating cardboard pallets...")
    
    # Clear scene
    clear_scene()
    
    # Create pallets
    pallets = create_multiple_pallets(num_pallets=6, spacing=1.8)
    
    # Setup lighting
    setup_lighting()
    
    # Setup camera
    setup_camera()
    
    # Setup render settings
    setup_render_settings()
    
    print(f"Generated {len(pallets)} cardboard pallets with realistic details")
    print("Scene is ready for rendering and YOLO training dataset generation")

if __name__ == "__main__":
    main()