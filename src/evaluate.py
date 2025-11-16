"""
Evaluation Script for Aircraft Skin Defect Classifier

This script evaluates the trained DCNN model on test data and generates
comprehensive performance metrics and visualizations.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_score, recall_score, f1_score
)
import tensorflow as tf
from tensorflow import keras

from data_loader import DefectDataLoader


def plot_confusion_matrix(y_true, y_pred, class_names, save_path='results/confusion_matrix.png'):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_class_distribution(y_true, y_pred, class_names, save_path='results/class_distribution.png'):
    """
    Plot distribution of predictions vs true labels.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # True distribution
    unique, counts = np.unique(y_true, return_counts=True)
    axes[0].bar(unique, counts, color='skyblue', edgecolor='black')
    axes[0].set_title('True Class Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Class', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_xticks(range(len(class_names)))
    axes[0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Predicted distribution
    unique, counts = np.unique(y_pred, return_counts=True)
    axes[1].bar(unique, counts, color='lightcoral', edgecolor='black')
    axes[1].set_title('Predicted Class Distribution', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Class', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_xticks(range(len(class_names)))
    axes[1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Class distribution plot saved to {save_path}")
    plt.close()


def evaluate_model(model_path, test_dir, img_size=(224, 224), batch_size=32):
    """
    Evaluate the trained model on test data.
    
    Args:
        model_path: Path to the trained model
        test_dir: Path to test data directory
        img_size: Input image size (height, width)
        batch_size: Batch size for evaluation
    """
    print("=" * 80)
    print("Aircraft Skin Defect Classifier - Evaluation")
    print("=" * 80)
    
    # Load the trained model
    print(f"\nLoading model from {model_path}...")
    model = keras.models.load_model(model_path)
    
    print("\nModel Summary:")
    model.summary()
    
    # Initialize data loader
    print("\nLoading test data...")
    data_loader = DefectDataLoader(
        data_dir=test_dir,
        img_size=img_size,
        batch_size=batch_size
    )
    
    # Load test data
    test_gen = data_loader.load_test_data(test_dir)
    class_names = list(test_gen.class_indices.keys())
    
    print(f"Test samples: {test_gen.samples}")
    print(f"Classes: {class_names}")
    
    # Evaluate on test data
    print("\n" + "=" * 80)
    print("Evaluating model...")
    print("=" * 80 + "\n")
    
    results = model.evaluate(test_gen, verbose=1)
    
    print("\n" + "=" * 80)
    print("Test Results:")
    print("=" * 80)
    print(f"Test Loss: {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]:.4f}")
    print(f"Test Precision: {results[2]:.4f}")
    print(f"Test Recall: {results[3]:.4f}")
    
    # Get predictions
    print("\nGenerating predictions...")
    predictions = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes
    
    # Calculate additional metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print("\n" + "=" * 80)
    print("Detailed Metrics:")
    print("=" * 80)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1-Score (weighted): {f1:.4f}")
    
    # Classification report
    print("\n" + "=" * 80)
    print("Classification Report:")
    print("=" * 80)
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names)
    
    # Plot class distribution
    plot_class_distribution(y_true, y_pred, class_names)
    
    # Save metrics to file
    metrics_file = 'results/evaluation_metrics.txt'
    with open(metrics_file, 'w') as f:
        f.write("Aircraft Skin Defect Classifier - Evaluation Metrics\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Test Data: {test_dir}\n\n")
        f.write(f"Test Loss: {results[0]:.4f}\n")
        f.write(f"Test Accuracy: {results[1]:.4f}\n")
        f.write(f"Test Precision: {results[2]:.4f}\n")
        f.write(f"Test Recall: {results[3]:.4f}\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision (weighted): {precision:.4f}\n")
        f.write(f"Recall (weighted): {recall:.4f}\n")
        f.write(f"F1-Score (weighted): {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write("=" * 80 + "\n")
        f.write(classification_report(y_true, y_pred, target_names=class_names))
    
    print(f"\nEvaluation metrics saved to {metrics_file}")
    
    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description='Evaluate DCNN for aircraft skin defect classification'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to the trained model'
    )
    parser.add_argument(
        '--test_dir',
        type=str,
        default='data/raw',
        help='Path to test data directory'
    )
    parser.add_argument(
        '--img_size',
        type=int,
        default=224,
        help='Input image size (will be square)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for evaluation'
    )
    
    args = parser.parse_args()
    
    # Evaluate the model
    evaluate_model(
        model_path=args.model_path,
        test_dir=args.test_dir,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()
