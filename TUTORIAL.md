# AER850 Project 2 - Quick Start Tutorial

## Introduction

This tutorial will guide you through the complete process of training and using the aircraft skin defect classification system.

## Step 1: Installation

First, install all required dependencies:

```bash
pip install -r requirements.txt
```

## Step 2: Prepare Your Data

### Option A: Using the Setup Script

Run the setup script to create the directory structure:

```bash
python example_workflow.py setup
```

### Option B: Manual Setup

Create the following directory structure:

```
data/raw/
├── Crack/
├── Missing_Screw_Head/
└── Paint_Degradation/
```

### Add Your Images

Place your images in the appropriate class directories:
- Aircraft skin crack images → `data/raw/Crack/`
- Missing screw head images → `data/raw/Missing_Screw_Head/`
- Paint degradation images → `data/raw/Paint_Degradation/`

**Supported formats**: JPG, JPEG, PNG, BMP

## Step 3: Verify Your Data

Check that your data is properly organized:

```bash
python example_workflow.py verify
```

## Step 4: Train the Model

### Basic Training (Custom CNN)

Train with default settings:

```bash
python src/train.py --data_dir data/raw --model_type custom --epochs 50
```

### Transfer Learning (VGG16)

Use transfer learning for potentially better results:

```bash
python src/train.py --data_dir data/raw --model_type transfer --epochs 50
```

### Advanced Training Options

```bash
python src/train.py \
  --data_dir data/raw \
  --model_type custom \
  --epochs 100 \
  --batch_size 16 \
  --learning_rate 0.0001 \
  --img_size 224
```

**Training Parameters:**
- `--epochs`: Number of training iterations (default: 50)
- `--batch_size`: Images per batch (default: 32)
  - Reduce if out of memory
  - Increase if you have more GPU memory
- `--learning_rate`: Optimizer learning rate (default: 0.001)
- `--img_size`: Input image dimensions (default: 224)
- `--no_augmentation`: Disable data augmentation

## Step 5: Monitor Training

During training, you'll see:
- Loss and accuracy metrics per epoch
- Validation performance
- Model checkpointing (saves best model)
- Early stopping (stops if no improvement)

Training outputs are saved to:
- `models/` - Trained model files (.h5)
- `results/` - Training plots and logs

## Step 6: Evaluate the Model

Test your trained model:

```bash
python src/evaluate.py \
  --model_path models/defect_classifier_custom_final.h5 \
  --test_dir data/raw
```

This generates:
- Accuracy, precision, recall, F1-score
- Confusion matrix visualization
- Class distribution plots
- Detailed classification report

## Step 7: Make Predictions

### Predict Single Image

```bash
python src/predict.py \
  --model_path models/defect_classifier_custom_final.h5 \
  --image path/to/defect_image.jpg
```

### Predict Multiple Images

```bash
python src/predict.py \
  --model_path models/defect_classifier_custom_final.h5 \
  --images image1.jpg image2.jpg image3.jpg
```

### Predict All Images in Directory

```bash
python src/predict.py \
  --model_path models/defect_classifier_custom_final.h5 \
  --image path/to/image/folder/
```

Prediction outputs include:
- Class predictions with confidence scores
- Probability distributions for all classes
- Visual results saved to `results/predictions/`

## Understanding the Model

### Custom CNN Architecture

The custom model has:
- **4 Convolutional Blocks**: Extract features at different scales
- **Batch Normalization**: Stabilize training
- **Dropout Layers**: Prevent overfitting
- **Dense Layers**: Final classification
- **~27 million parameters**

### Transfer Learning Architecture

Transfer learning uses:
- **Pre-trained VGG16**: Learned features from ImageNet
- **Custom Classification Head**: Adapted for 3 defect classes
- **Frozen Base Model**: Retains general visual features
- **Fewer trainable parameters**: Faster training

## Tips for Best Results

### 1. Data Quality
- Use high-quality, clear images
- Ensure consistent lighting conditions
- Remove corrupted or unclear images
- Balance dataset (similar number of images per class)

### 2. Data Augmentation
- Enabled by default
- Helps model generalize better
- Includes rotation, shift, zoom, flip

### 3. Training Time
- Custom CNN: May take 1-3 hours (CPU) or 10-30 minutes (GPU)
- Transfer Learning: Usually faster
- Use early stopping to avoid overtraining

### 4. Model Selection
- **Custom CNN**: Good for learning from scratch
- **Transfer Learning**: Better with limited data
- Try both and compare results

### 5. Hyperparameter Tuning
- Start with defaults
- Adjust batch size based on memory
- Lower learning rate if loss oscillates
- Increase epochs if still improving

## Troubleshooting

### Out of Memory Error
```bash
# Reduce batch size
python src/train.py --batch_size 8
```

### Poor Accuracy
- Check data quality and organization
- Try transfer learning
- Increase training epochs
- Ensure balanced dataset

### Slow Training
- Enable GPU if available
- Reduce image size: `--img_size 128`
- Increase batch size if memory allows

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

## Example Workflow

Complete workflow from start to finish:

```bash
# 1. Setup
python example_workflow.py setup

# 2. Add your images to data/raw/

# 3. Verify data
python example_workflow.py verify

# 4. Train model
python src/train.py --data_dir data/raw --model_type custom --epochs 50

# 5. Evaluate model
python src/evaluate.py --model_path models/defect_classifier_custom_final.h5

# 6. Make predictions
python src/predict.py --model_path models/defect_classifier_custom_final.h5 --image test_image.jpg
```

## Understanding Results

### Training Metrics

- **Accuracy**: Percentage of correct predictions
- **Loss**: Error measure (lower is better)
- **Precision**: Of predicted defects, how many were correct
- **Recall**: Of actual defects, how many were detected

### Confusion Matrix

Shows prediction vs actual labels:
- Diagonal: Correct predictions
- Off-diagonal: Misclassifications

### Prediction Confidence

- **High (>90%)**: Model is very confident
- **Medium (70-90%)**: Model is fairly confident
- **Low (<70%)**: Model is uncertain

## Next Steps

1. **Collect More Data**: More images improve accuracy
2. **Fine-tune Model**: Adjust hyperparameters
3. **Try Both Architectures**: Compare custom vs transfer learning
4. **Ensemble Methods**: Combine multiple models
5. **Deploy Model**: Integrate into inspection workflow

## Additional Resources

- TensorFlow Documentation: https://www.tensorflow.org/
- Keras Guide: https://keras.io/guides/
- Image Classification Tutorial: https://www.tensorflow.org/tutorials/images/classification

## Support

For issues or questions:
1. Check this tutorial
2. Review README.md
3. Examine error messages
4. Check GitHub issues

## Summary

This system provides a complete pipeline for aircraft defect classification:
- ✓ Data preparation and augmentation
- ✓ Model training with best practices
- ✓ Comprehensive evaluation
- ✓ Easy prediction interface
- ✓ Visualization and reporting

Follow this tutorial to successfully train and deploy your defect classifier!
