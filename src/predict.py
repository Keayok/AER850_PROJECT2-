"""
Prediction Script for Aircraft Skin Defect Classifier

This script performs inference on single images or batches of images
to classify aircraft skin defects.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow import keras

from data_loader import DefectDataLoader


def predict_single_image(model, image_path, img_size=(224, 224)):
    """
    Predict defect class for a single image.
    
    Args:
        model: Trained Keras model
        image_path: Path to the image file
        img_size: Input image size (height, width)
        
    Returns:
        Tuple of (predicted_class, confidence, all_probabilities)
    """
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img = img.resize(img_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    return predicted_class, confidence, predictions[0]


def visualize_prediction(image_path, predicted_class, confidence, 
                        all_probs, class_names, save_path=None):
    """
    Visualize the prediction with the image and probability bars.
    
    Args:
        image_path: Path to the image
        predicted_class: Predicted class index
        confidence: Confidence score
        all_probs: All class probabilities
        class_names: List of class names
        save_path: Optional path to save the visualization
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Display image
    img = Image.open(image_path)
    axes[0].imshow(img)
    axes[0].axis('off')
    axes[0].set_title(
        f'Predicted: {class_names[predicted_class]}\nConfidence: {confidence:.2%}',
        fontsize=14, fontweight='bold'
    )
    
    # Display probabilities
    colors = ['green' if i == predicted_class else 'gray' 
              for i in range(len(class_names))]
    axes[1].barh(class_names, all_probs, color=colors, edgecolor='black')
    axes[1].set_xlabel('Probability', fontsize=12)
    axes[1].set_title('Class Probabilities', fontsize=14, fontweight='bold')
    axes[1].set_xlim([0, 1])
    
    # Add percentage labels
    for i, prob in enumerate(all_probs):
        axes[1].text(prob + 0.02, i, f'{prob:.2%}', 
                    va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def predict_images(model_path, image_paths, img_size=(224, 224), 
                   visualize=True, save_results=True):
    """
    Predict defect classes for multiple images.
    
    Args:
        model_path: Path to the trained model
        image_paths: List of image paths or directory path
        img_size: Input image size (height, width)
        visualize: Whether to create visualizations
        save_results: Whether to save results to file
    """
    print("=" * 80)
    print("Aircraft Skin Defect Classifier - Prediction")
    print("=" * 80)
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    model = keras.models.load_model(model_path)
    
    # Define class names
    class_names = ['Crack', 'Missing_Screw_Head', 'Paint_Degradation']
    
    # Handle directory input
    if isinstance(image_paths, str) and os.path.isdir(image_paths):
        image_dir = image_paths
        image_paths = [
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]
        print(f"\nFound {len(image_paths)} images in {image_dir}")
    
    # Create results directory
    if save_results or visualize:
        os.makedirs('results/predictions', exist_ok=True)
    
    # Predict for each image
    results = []
    
    print("\n" + "=" * 80)
    print("Making Predictions...")
    print("=" * 80 + "\n")
    
    for i, image_path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] Processing: {os.path.basename(image_path)}")
        
        try:
            # Make prediction
            pred_class, confidence, all_probs = predict_single_image(
                model, image_path, img_size
            )
            
            # Store result
            result = {
                'image': os.path.basename(image_path),
                'path': image_path,
                'predicted_class': class_names[pred_class],
                'confidence': confidence,
                'probabilities': {
                    class_names[j]: all_probs[j] for j in range(len(class_names))
                }
            }
            results.append(result)
            
            # Print result
            print(f"  Prediction: {class_names[pred_class]}")
            print(f"  Confidence: {confidence:.2%}")
            print(f"  Probabilities: ", end="")
            for j, name in enumerate(class_names):
                print(f"{name}: {all_probs[j]:.2%}", end="  ")
            print("\n")
            
            # Visualize if requested
            if visualize:
                save_path = f'results/predictions/{os.path.splitext(os.path.basename(image_path))[0]}_prediction.png'
                visualize_prediction(
                    image_path, pred_class, confidence, 
                    all_probs, class_names, save_path
                )
        
        except Exception as e:
            print(f"  Error processing image: {e}\n")
            continue
    
    # Save results to file
    if save_results and results:
        results_file = 'results/prediction_results.txt'
        with open(results_file, 'w') as f:
            f.write("Aircraft Skin Defect Classifier - Prediction Results\n")
            f.write("=" * 80 + "\n\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"Image {i}: {result['image']}\n")
                f.write(f"  Predicted Class: {result['predicted_class']}\n")
                f.write(f"  Confidence: {result['confidence']:.2%}\n")
                f.write("  Probabilities:\n")
                for class_name, prob in result['probabilities'].items():
                    f.write(f"    {class_name}: {prob:.2%}\n")
                f.write("\n")
        
        print(f"Prediction results saved to {results_file}")
    
    print("\n" + "=" * 80)
    print("Prediction Complete!")
    print("=" * 80)
    print(f"\nTotal images processed: {len(results)}")
    
    return results


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(
        description='Predict defects in aircraft skin images'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to the trained model'
    )
    parser.add_argument(
        '--image',
        type=str,
        help='Path to a single image or directory of images'
    )
    parser.add_argument(
        '--images',
        type=str,
        nargs='+',
        help='List of image paths'
    )
    parser.add_argument(
        '--img_size',
        type=int,
        default=224,
        help='Input image size (will be square)'
    )
    parser.add_argument(
        '--no_visualize',
        action='store_true',
        help='Disable visualization'
    )
    parser.add_argument(
        '--no_save',
        action='store_true',
        help='Do not save results to file'
    )
    
    args = parser.parse_args()
    
    # Determine image paths
    if args.image:
        image_paths = args.image
    elif args.images:
        image_paths = args.images
    else:
        print("Error: Please provide --image or --images argument")
        return
    
    # Make predictions
    predict_images(
        model_path=args.model_path,
        image_paths=image_paths,
        img_size=(args.img_size, args.img_size),
        visualize=not args.no_visualize,
        save_results=not args.no_save
    )


if __name__ == '__main__':
    main()
