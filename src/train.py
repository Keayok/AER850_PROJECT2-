"""
Training Script for Aircraft Skin Defect Classifier

This script trains the DCNN model on aircraft skin defect images.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from tensorflow import keras

from model import DefectClassifierDCNN
from data_loader import DefectDataLoader, verify_data_structure


def plot_training_history(history, save_path='results/training_history.png'):
    """
    Plot and save training history.
    
    Args:
        history: Keras training history object
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot loss
    axes[0, 1].plot(history.history['loss'], label='Training Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot precision (check if metric exists)
    precision_keys = [k for k in history.history.keys() if 'precision' in k.lower() and not k.startswith('val_')]
    val_precision_keys = [k for k in history.history.keys() if 'precision' in k.lower() and k.startswith('val_')]
    
    if precision_keys and val_precision_keys:
        axes[1, 0].plot(history.history[precision_keys[0]], label='Training Precision')
        axes[1, 0].plot(history.history[val_precision_keys[0]], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Plot recall (check if metric exists)
    recall_keys = [k for k in history.history.keys() if 'recall' in k.lower() and not k.startswith('val_')]
    val_recall_keys = [k for k in history.history.keys() if 'recall' in k.lower() and k.startswith('val_')]
    
    if recall_keys and val_recall_keys:
        axes[1, 1].plot(history.history[recall_keys[0]], label='Training Recall')
        axes[1, 1].plot(history.history[val_recall_keys[0]], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    plt.close()


def train_model(data_dir, model_type='custom', epochs=50, batch_size=32, 
                learning_rate=0.001, img_size=(224, 224), use_augmentation=True):
    """
    Train the defect classification model.
    
    Args:
        data_dir: Path to training data directory
        model_type: Type of model ('custom' or 'transfer')
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        img_size: Input image size (height, width)
        use_augmentation: Whether to use data augmentation
    """
    print("=" * 80)
    print("Aircraft Skin Defect Classifier - Training")
    print("=" * 80)
    
    # Verify data structure
    print("\nVerifying data structure...")
    if not verify_data_structure(data_dir):
        print("Error: Invalid data structure. Please organize your data correctly.")
        return
    
    # Initialize data loader
    print("\nInitializing data loader...")
    data_loader = DefectDataLoader(
        data_dir=data_dir,
        img_size=img_size,
        batch_size=batch_size,
        validation_split=0.2
    )
    
    # Create data generators
    print("Creating data generators...")
    train_gen, val_gen = data_loader.create_data_generators(augment=use_augmentation)
    
    print(f"\nTraining samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Class indices: {train_gen.class_indices}")
    
    # Calculate class weights to handle imbalance
    class_weights = data_loader.get_class_weights(train_gen)
    print(f"Class weights: {class_weights}")
    
    # Initialize model
    print(f"\nBuilding {model_type} model...")
    classifier = DefectClassifierDCNN(
        input_shape=(*img_size, 3),
        num_classes=3
    )
    
    if model_type == 'custom':
        classifier.build_custom_cnn()
    elif model_type == 'transfer':
        classifier.build_transfer_learning_model(base_model_name='VGG16')
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    classifier.compile_model(learning_rate=learning_rate)
    
    print("\nModel Summary:")
    classifier.get_model_summary()
    
    # Setup callbacks
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'models/defect_classifier_{model_type}_{timestamp}.h5'
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.CSVLogger(
            f'results/training_log_{timestamp}.csv'
        )
    ]
    
    # Train the model
    print("\n" + "=" * 80)
    print("Starting Training...")
    print("=" * 80 + "\n")
    
    history = classifier.model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Save final model
    final_model_path = f'models/defect_classifier_{model_type}_final.h5'
    classifier.save_model(final_model_path)
    
    # Plot training history
    plot_training_history(history, f'results/training_history_{timestamp}.png')
    
    # Print final results
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"\nBest model saved to: {model_path}")
    print(f"Final model saved to: {final_model_path}")
    print(f"\nFinal Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
    print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
    
    return classifier, history


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description='Train DCNN for aircraft skin defect classification'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/raw',
        help='Path to training data directory'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='custom',
        choices=['custom', 'transfer'],
        help='Model type: custom CNN or transfer learning'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for training'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Learning rate for optimizer'
    )
    parser.add_argument(
        '--img_size',
        type=int,
        default=224,
        help='Input image size (will be square)'
    )
    parser.add_argument(
        '--no_augmentation',
        action='store_true',
        help='Disable data augmentation'
    )
    
    args = parser.parse_args()
    
    # Train the model
    train_model(
        data_dir=args.data_dir,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        img_size=(args.img_size, args.img_size),
        use_augmentation=not args.no_augmentation
    )


if __name__ == '__main__':
    main()
