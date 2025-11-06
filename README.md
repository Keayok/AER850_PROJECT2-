# AER850 Project 2 - Aircraft Skin Defect Classification using Deep Convolutional Neural Networks (DCNNs)

## Project Overview

This project was developed as part of **AER850: Introduction to Machine Learning** and focuses on building a **Deep Convolutional Neural Network (DCNN)** capable of classifying aircraft skin defects into three categories:

1. **Crack** - Structural cracks in aircraft skin
2. **Missing Screw Head** - Missing or damaged fasteners
3. **Paint Degradation (Paint-Off)** - Areas with paint wear or removal

The goal is to automate visual inspection tasks in aircraft maintenance, improving the **accuracy** and **efficiency** of defect detection compared to traditional manual inspections.

## Features

- **Custom Deep CNN Architecture**: Built from scratch with multiple convolutional blocks
- **Transfer Learning Support**: Pre-trained VGG16 and ResNet50 models for improved performance
- **Data Augmentation**: Robust data augmentation to improve generalization
- **Comprehensive Evaluation**: Detailed metrics including accuracy, precision, recall, F1-score
- **Visualization Tools**: Training history plots, confusion matrices, and prediction visualizations
- **Class Imbalance Handling**: Automatic class weight calculation for balanced training
- **Model Checkpointing**: Automatic saving of best models during training
- **Easy Inference**: Simple prediction interface for single images or batches

## Project Structure

```
AER850_PROJECT2-/
├── src/
│   ├── model.py           # DCNN model architecture definitions
│   ├── data_loader.py     # Data loading and preprocessing utilities
│   ├── train.py           # Training script
│   ├── evaluate.py        # Model evaluation script
│   └── predict.py         # Prediction/inference script
├── data/
│   ├── raw/               # Raw training/test data
│   │   ├── Crack/
│   │   ├── Missing_Screw_Head/
│   │   └── Paint_Degradation/
│   └── processed/         # Processed data (optional)
├── models/                # Saved trained models
├── results/               # Training/evaluation results and plots
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-enabled GPU for faster training

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Keayok/AER850_PROJECT2-.git
cd AER850_PROJECT2-
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

### Directory Structure

Organize your image data in the following structure:

```
data/raw/
├── Crack/
│   ├── crack_001.jpg
│   ├── crack_002.jpg
│   └── ...
├── Missing_Screw_Head/
│   ├── screw_001.jpg
│   ├── screw_002.jpg
│   └── ...
└── Paint_Degradation/
    ├── paint_001.jpg
    ├── paint_002.jpg
    └── ...
```

### Image Format

- Supported formats: JPG, JPEG, PNG, BMP
- Recommended: RGB images
- Images will be automatically resized to 224x224 pixels (configurable)

## Usage

### 1. Training the Model

Train a custom CNN model:

```bash
python src/train.py --data_dir data/raw --model_type custom --epochs 50 --batch_size 32
```

Train with transfer learning (VGG16):

```bash
python src/train.py --data_dir data/raw --model_type transfer --epochs 50 --batch_size 32
```

#### Training Arguments:

- `--data_dir`: Path to training data directory (default: `data/raw`)
- `--model_type`: Model architecture - `custom` or `transfer` (default: `custom`)
- `--epochs`: Number of training epochs (default: `50`)
- `--batch_size`: Batch size for training (default: `32`)
- `--learning_rate`: Learning rate for optimizer (default: `0.001`)
- `--img_size`: Input image size in pixels (default: `224`)
- `--no_augmentation`: Disable data augmentation

### 2. Evaluating the Model

Evaluate a trained model on test data:

```bash
python src/evaluate.py --model_path models/defect_classifier_custom_final.h5 --test_dir data/raw
```

#### Evaluation Arguments:

- `--model_path`: Path to the trained model (required)
- `--test_dir`: Path to test data directory (default: `data/raw`)
- `--img_size`: Input image size in pixels (default: `224`)
- `--batch_size`: Batch size for evaluation (default: `32`)

### 3. Making Predictions

Predict on a single image:

```bash
python src/predict.py --model_path models/defect_classifier_custom_final.h5 --image path/to/image.jpg
```

Predict on multiple images:

```bash
python src/predict.py --model_path models/defect_classifier_custom_final.h5 --images img1.jpg img2.jpg img3.jpg
```

Predict on all images in a directory:

```bash
python src/predict.py --model_path models/defect_classifier_custom_final.h5 --image path/to/image/directory/
```

#### Prediction Arguments:

- `--model_path`: Path to the trained model (required)
- `--image`: Path to a single image or directory of images
- `--images`: List of image paths
- `--img_size`: Input image size in pixels (default: `224`)
- `--no_visualize`: Disable prediction visualizations
- `--no_save`: Do not save results to file

## Model Architecture

### Custom DCNN

The custom model consists of:
- **4 Convolutional Blocks**: Each with 2 Conv2D layers, BatchNormalization, MaxPooling, and Dropout
- **Progressive Filter Expansion**: 32 → 64 → 128 → 256 filters
- **Dense Layers**: 2 fully connected layers (512, 256 neurons) with dropout
- **Output Layer**: Softmax activation for 3-class classification

### Transfer Learning

Transfer learning models use:
- **Pre-trained Base**: VGG16 or ResNet50 (ImageNet weights)
- **Custom Head**: GlobalAveragePooling + Dense layers (512, 256)
- **Fine-tuning Support**: Base model frozen initially, can be unfrozen for fine-tuning

## Training Features

- **Data Augmentation**: Rotation, shift, shear, zoom, horizontal flip
- **Class Weight Balancing**: Automatic handling of class imbalance
- **Model Checkpointing**: Saves best model based on validation accuracy
- **Early Stopping**: Prevents overfitting with patience of 10 epochs
- **Learning Rate Reduction**: Reduces LR on plateau for better convergence
- **CSV Logging**: Detailed training logs for analysis

## Output Files

### Training Outputs

- `models/defect_classifier_*.h5`: Saved model files
- `results/training_history_*.png`: Training/validation metrics plots
- `results/training_log_*.csv`: Detailed training logs

### Evaluation Outputs

- `results/confusion_matrix.png`: Confusion matrix visualization
- `results/class_distribution.png`: True vs predicted class distributions
- `results/evaluation_metrics.txt`: Detailed evaluation metrics

### Prediction Outputs

- `results/predictions/*_prediction.png`: Individual prediction visualizations
- `results/prediction_results.txt`: Prediction results summary

## Performance Metrics

The model is evaluated using:
- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class and weighted average precision
- **Recall**: Per-class and weighted average recall
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown

## Example Workflow

```bash
# 1. Prepare your data in the correct directory structure
mkdir -p data/raw/{Crack,Missing_Screw_Head,Paint_Degradation}
# Copy your images to respective directories

# 2. Train the model
python src/train.py --data_dir data/raw --model_type custom --epochs 50

# 3. Evaluate the model
python src/evaluate.py --model_path models/defect_classifier_custom_final.h5 --test_dir data/raw

# 4. Make predictions on new images
python src/predict.py --model_path models/defect_classifier_custom_final.h5 --image new_defect.jpg
```

## Requirements

See `requirements.txt` for a complete list of dependencies. Main requirements:

- TensorFlow >= 2.12.0
- NumPy >= 1.23.0
- Matplotlib >= 3.7.0
- Scikit-learn >= 1.2.0
- Pillow >= 9.5.0
- Pandas >= 2.0.0
- Seaborn >= 0.12.0

## Hardware Recommendations

- **CPU Training**: Possible but slow (may take hours)
- **GPU Training**: Recommended (NVIDIA GPU with CUDA support)
- **RAM**: Minimum 8GB, 16GB+ recommended
- **Storage**: Depends on dataset size

## Troubleshooting

### Common Issues

1. **Out of Memory Error**: Reduce batch size (`--batch_size 16` or `--batch_size 8`)
2. **Slow Training**: Enable GPU support or reduce image size
3. **Poor Accuracy**: Try transfer learning, increase epochs, or adjust learning rate
4. **Data Loading Error**: Verify directory structure and image formats

## Future Enhancements

- Support for additional defect types
- Real-time video stream processing
- Web-based user interface
- Mobile deployment (TensorFlow Lite)
- Explainability features (Grad-CAM visualizations)

## License

This project is developed for educational purposes as part of AER850 coursework.

## Acknowledgments

- AER850: Introduction to Machine Learning course
- TensorFlow and Keras communities
- Pre-trained model providers (ImageNet)

## Contact

For questions or issues, please open an issue on the GitHub repository. 
