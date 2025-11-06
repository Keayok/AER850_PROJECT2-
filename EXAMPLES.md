# Usage Examples for Aircraft Skin Defect Classifier

This file contains practical examples for different use cases.

## Example 1: Quick Start with Minimal Data

```bash
# 1. Setup directories
python example_workflow.py setup

# 2. Add a few images to each class folder (minimum 10 per class)
# data/raw/Crack/
# data/raw/Missing_Screw_Head/
# data/raw/Paint_Degradation/

# 3. Train with reduced epochs for quick testing
python src/train.py --data_dir data/raw --epochs 10 --batch_size 16

# 4. Evaluate
python src/evaluate.py --model_path models/defect_classifier_custom_final.h5

# 5. Test prediction
python src/predict.py --model_path models/defect_classifier_custom_final.h5 --image test.jpg
```

## Example 2: Training with Transfer Learning

```bash
# Use VGG16 pre-trained model for better performance with limited data
python src/train.py \
  --data_dir data/raw \
  --model_type transfer \
  --epochs 30 \
  --batch_size 16 \
  --learning_rate 0.0001
```

## Example 3: High-Accuracy Training

```bash
# Train longer with smaller learning rate
python src/train.py \
  --data_dir data/raw \
  --model_type custom \
  --epochs 100 \
  --batch_size 32 \
  --learning_rate 0.0001
```

## Example 4: Training on Limited Memory

```bash
# Reduce batch size and image size
python src/train.py \
  --data_dir data/raw \
  --batch_size 8 \
  --img_size 128
```

## Example 5: Batch Prediction on Multiple Images

```bash
# Predict on all images in a folder
python src/predict.py \
  --model_path models/defect_classifier_custom_final.h5 \
  --image path/to/test/images/

# Predict on specific images
python src/predict.py \
  --model_path models/defect_classifier_custom_final.h5 \
  --images crack1.jpg crack2.jpg screw1.jpg
```

## Example 6: No Data Augmentation

```bash
# Train without augmentation (use if images are limited or already augmented)
python src/train.py \
  --data_dir data/raw \
  --no_augmentation
```

## Example 7: Custom Image Size

```bash
# Use 256x256 images instead of default 224x224
python src/train.py \
  --data_dir data/raw \
  --img_size 256 \
  --batch_size 16
```

## Example 8: Evaluation Only

```bash
# Evaluate pre-trained model
python src/evaluate.py \
  --model_path models/my_model.h5 \
  --test_dir data/test \
  --batch_size 32
```

## Example 9: Prediction Without Saving Results

```bash
# Quick prediction without saving visualizations
python src/predict.py \
  --model_path models/defect_classifier_custom_final.h5 \
  --image test.jpg \
  --no_visualize \
  --no_save
```

## Example 10: Complete Workflow (Custom CNN)

```bash
# Complete workflow from scratch
python example_workflow.py setup
# ... add your images ...
python example_workflow.py verify
python src/train.py --data_dir data/raw --model_type custom --epochs 50
python src/evaluate.py --model_path models/defect_classifier_custom_final.h5
python src/predict.py --model_path models/defect_classifier_custom_final.h5 --image sample.jpg
```

## Example 11: Complete Workflow (Transfer Learning)

```bash
# Using transfer learning for better results
python example_workflow.py setup
# ... add your images ...
python src/train.py --data_dir data/raw --model_type transfer --epochs 30
python src/evaluate.py --model_path models/defect_classifier_transfer_final.h5
python src/predict.py --model_path models/defect_classifier_transfer_final.h5 --image sample.jpg
```

## Example 12: Python API Usage

```python
# Use the classifier in your Python code
from src.model import DefectClassifierDCNN
from src.data_loader import DefectDataLoader
import numpy as np

# Create and build model
classifier = DefectClassifierDCNN(input_shape=(224, 224, 3), num_classes=3)
classifier.build_custom_cnn()
classifier.compile_model()

# Load pre-trained weights
classifier.load_model('models/defect_classifier_custom_final.h5')

# Prepare data
data_loader = DefectDataLoader(data_dir='data/raw', img_size=(224, 224))

# Make prediction
img_array = data_loader.preprocess_single_image('test.jpg')
prediction = classifier.model.predict(img_array)

# Get predicted class
class_names = ['Crack', 'Missing_Screw_Head', 'Paint_Degradation']
predicted_class = np.argmax(prediction[0])
confidence = prediction[0][predicted_class]

print(f"Predicted: {class_names[predicted_class]}")
print(f"Confidence: {confidence:.2%}")
```

## Example 13: Monitoring Training Progress

```bash
# Training outputs are logged to CSV
python src/train.py --data_dir data/raw --epochs 50

# After training, check the logs
cat results/training_log_*.csv

# View training plots
ls results/training_history_*.png
```

## Example 14: Model Comparison

```bash
# Train both models and compare
echo "Training Custom CNN..."
python src/train.py --data_dir data/raw --model_type custom --epochs 50

echo "Training Transfer Learning..."
python src/train.py --data_dir data/raw --model_type transfer --epochs 50

# Evaluate both
echo "Evaluating Custom CNN..."
python src/evaluate.py --model_path models/defect_classifier_custom_final.h5

echo "Evaluating Transfer Learning..."
python src/evaluate.py --model_path models/defect_classifier_transfer_final.h5

# Compare results in results/evaluation_metrics.txt
```

## Troubleshooting Common Issues

### Out of Memory
```bash
# Reduce batch size
python src/train.py --batch_size 8

# Or reduce image size
python src/train.py --img_size 128 --batch_size 16
```

### Slow Training
```bash
# Reduce number of epochs for testing
python src/train.py --epochs 10

# Use smaller image size
python src/train.py --img_size 128
```

### Poor Accuracy
```bash
# Try transfer learning
python src/train.py --model_type transfer

# Train longer
python src/train.py --epochs 100

# Lower learning rate
python src/train.py --learning_rate 0.0001
```

## Tips for Best Results

1. **Data Quality**: Use clear, well-lit images
2. **Data Balance**: Have similar numbers of images per class
3. **Data Quantity**: More data = better results (aim for 100+ per class)
4. **Augmentation**: Keep enabled unless you have lots of data
5. **Model Selection**: Try both custom and transfer learning
6. **Patience**: Let training complete, don't stop early
7. **Validation**: Always evaluate on unseen test data

## Getting Help

- Check README.md for detailed documentation
- See TUTORIAL.md for step-by-step guide
- Run `python example_workflow.py` for quick setup help
- Review training logs in results/ directory
