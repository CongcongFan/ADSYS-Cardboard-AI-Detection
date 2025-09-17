# Cardboard QC Labeling Guide

## Quick Start

1. **Navigate to the folder**: Open command prompt and navigate to this folder:
   ```cmd
   cd "C:\Users\76135\Desktop\ADSYS-Cardboard-AI-Detection\Yolo DS To Qwen DS"
   ```

2. **Start labeling**: Run the labeling tool:
   ```cmd
   python label_helper.py
   ```

## How to Label

### What you'll see:
- **Green boxes** around detected cardboard bundles
- **Images** showing pallets with cardboard stacks

### Labeling Rules:
- **PASS (✓)** = Cardboard bundle appears **FLAT** and properly stacked
- **FAIL (✗)** = Cardboard bundle appears **WARPED**, bent, or damaged

### GUI Controls:
- **✓ PASS (Flat)** button - Mark as good quality (or press 'P' key)
- **✗ FAIL (Warped)** button - Mark as bad quality (or press 'F' key)
- **← Previous / Next →** - Navigate between images
- **Skip** - Skip current image (or press 'S' key)
- **💾 Save Progress** - Save your work (or press Ctrl+S)

### Keyboard Shortcuts:
- `P` = Pass (flat)
- `F` = Fail (warped)  
- `←` / `→` = Navigate images
- `S` = Skip image
- `Ctrl+S` = Save progress

## Examples

### ✅ PASS Examples:
- Cardboard appears flat and even
- Neat, organized stacking
- No visible bending or warping
- Clean, straight edges

### ❌ FAIL Examples:
- Visible warping or bending
- Uneven cardboard surface
- Damaged or crushed areas
- Irregular stacking pattern

## Important Notes

1. **Focus on the GREEN BOX** - Only label what's inside the detected area
2. **Save frequently** - Use Ctrl+S or the Save button regularly
3. **Take breaks** - Label accuracy is more important than speed
4. **When in doubt** - Skip the image rather than guessing

## Progress Tracking

- Progress bar shows completion status
- Window title shows current image status
- Counter shows: labeled/total images

## Troubleshooting

If GUI doesn't work:
- The program will automatically fall back to text mode
- Follow the text prompts to label images
- Open images manually from the `out/overlays_overlay/` folder

## File Locations

- **Images to label**: `out/overlays_overlay/`
- **Progress saved in**: `out/qc_labels.csv`
- **Backup your CSV file** periodically to avoid data loss

---

**Need help?** Contact the project administrator.