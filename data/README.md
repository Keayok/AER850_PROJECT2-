# Data Directory

This directory contains the dataset for training and testing the aircraft skin defect classifier.

## Directory Structure

```
data/
├── raw/                    # Raw training/test data
│   ├── Crack/             # Images of cracks in aircraft skin
│   ├── Missing_Screw_Head/# Images of missing screw heads
│   └── Paint_Degradation/ # Images of paint degradation/paint-off
└── processed/             # (Optional) Processed data
```

## How to Organize Your Data

1. **Crack**: Place images showing cracks or fractures in aircraft skin
2. **Missing_Screw_Head**: Place images showing missing or damaged screw heads/fasteners
3. **Paint_Degradation**: Place images showing paint wear, chipping, or removal

## Data Requirements

### Minimum Requirements
- **At least 50-100 images per class** (more is better)
- **Balanced classes**: Similar number of images in each category
- **Image formats**: JPG, JPEG, PNG, or BMP
- **Image quality**: Clear, focused images

### Recommended Requirements
- **200+ images per class** for best results
- **Variety**: Different angles, lighting conditions, defect severities
- **Resolution**: At least 224x224 pixels (higher is fine, will be resized)
- **Color images**: RGB format preferred

## Data Splitting

The training script automatically splits data:
- **80% Training**: Used to train the model
- **20% Validation**: Used to tune and evaluate during training

If you have a separate test set, place it in a similar structure for final evaluation.

## Example Data Organization

```
data/raw/Crack/
├── crack_001.jpg
├── crack_002.jpg
├── crack_003.jpg
...

data/raw/Missing_Screw_Head/
├── screw_001.jpg
├── screw_002.jpg
├── screw_003.jpg
...

data/raw/Paint_Degradation/
├── paint_001.jpg
├── paint_002.jpg
├── paint_003.jpg
...
```

## Data Quality Tips

1. **Clear Images**: Avoid blurry or out-of-focus images
2. **Consistent Lighting**: Similar lighting conditions across images
3. **Proper Framing**: Defect should be clearly visible
4. **Remove Duplicates**: Don't include identical or near-identical images
5. **Clean Labels**: Ensure images are in the correct category

## Data Augmentation

The training pipeline includes automatic data augmentation:
- Rotation (±20 degrees)
- Width/height shifting (±20%)
- Shearing (20%)
- Zooming (±20%)
- Horizontal flipping

This helps the model generalize better, especially with limited data.

## Getting Started

1. Run setup to create directory structure:
   ```bash
   python example_workflow.py setup
   ```

2. Add your images to the class folders

3. Verify your data structure:
   ```bash
   python example_workflow.py verify
   ```

4. Start training!

## Data Privacy and Security

- Do not commit sensitive or proprietary images to version control
- The `.gitignore` file excludes `data/raw/` by default
- Keep backups of your original data

## Need Sample Data?

If you don't have aircraft defect images yet:
1. Search for publicly available datasets
2. Use synthetic/simulated defect images
3. Collect images from training materials or documentation
4. Contact instructors for sample datasets

## Questions?

See the main README.md or TUTORIAL.md for more information.
