"""
Deep Convolutional Neural Network for Aircraft Skin Defect Classification

This module defines the DCNN architecture for classifying aircraft skin defects
into three categories: Crack, Missing Screw Head, and Paint Degradation.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, ResNet50


class DefectClassifierDCNN:
    """
    Deep Convolutional Neural Network for defect classification.
    
    The network is designed to classify aircraft skin defects into:
    - Crack (Class 0)
    - Missing Screw Head (Class 1)
    - Paint Degradation/Paint-Off (Class 2)
    """
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=3):
        """
        Initialize the DCNN model.
        
        Args:
            input_shape: Tuple of input image dimensions (height, width, channels)
            num_classes: Number of defect classes (default: 3)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def build_custom_cnn(self):
        """
        Build a custom deep CNN architecture from scratch.
        
        Returns:
            Compiled Keras model
        """
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                         input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output Layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def build_transfer_learning_model(self, base_model_name='VGG16'):
        """
        Build a model using transfer learning with a pre-trained base.
        
        Args:
            base_model_name: Name of the base model ('VGG16' or 'ResNet50')
            
        Returns:
            Compiled Keras model
        """
        # Load pre-trained model
        if base_model_name == 'VGG16':
            base_model = VGG16(weights='imagenet', include_top=False, 
                              input_shape=self.input_shape)
        elif base_model_name == 'ResNet50':
            base_model = ResNet50(weights='imagenet', include_top=False, 
                                 input_shape=self.input_shape)
        else:
            raise ValueError(f"Unsupported base model: {base_model_name}")
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom classification head
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile the model with optimizer, loss, and metrics.
        
        Args:
            learning_rate: Learning rate for the optimizer
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_custom_cnn() or build_transfer_learning_model() first.")
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
    def get_model_summary(self):
        """
        Print model architecture summary.
        """
        if self.model is None:
            raise ValueError("Model not built yet.")
        return self.model.summary()
    
    def save_model(self, filepath):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not built yet.")
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        return self.model
