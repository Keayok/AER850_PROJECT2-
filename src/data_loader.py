"""
Data Preprocessing and Loading Utilities

This module provides functions for loading, preprocessing, and augmenting
aircraft skin defect images for the DCNN classifier.
"""

import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


class DefectDataLoader:
    """
    Data loader for aircraft skin defect images.
    """
    
    def __init__(self, data_dir, img_size=(224, 224), batch_size=32, validation_split=0.2):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Path to the data directory
            img_size: Target image size (height, width)
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.validation_split = validation_split
        
        # Define class names
        self.class_names = ['Crack', 'Missing_Screw_Head', 'Paint_Degradation']
        self.num_classes = len(self.class_names)
        
    def create_data_generators(self, augment=True):
        """
        Create data generators for training and validation.
        
        Args:
            augment: Whether to apply data augmentation
            
        Returns:
            Tuple of (train_generator, validation_generator)
        """
        if augment:
            # Data augmentation for training
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest',
                validation_split=self.validation_split
            )
        else:
            # No augmentation
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=self.validation_split
            )
        
        # Validation data generator (no augmentation)
        validation_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=self.validation_split
        )
        
        # Create training generator
        train_generator = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True,
            seed=42
        )
        
        # Create validation generator
        validation_generator = validation_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False,
            seed=42
        )
        
        return train_generator, validation_generator
    
    def load_test_data(self, test_dir):
        """
        Load test data for evaluation.
        
        Args:
            test_dir: Path to test data directory
            
        Returns:
            Test data generator
        """
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        return test_generator
    
    def preprocess_single_image(self, image_path):
        """
        Preprocess a single image for prediction.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image array
        """
        img = Image.open(image_path)
        img = img.resize(self.img_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    
    def get_class_weights(self, train_generator):
        """
        Calculate class weights for handling class imbalance.
        
        Args:
            train_generator: Training data generator
            
        Returns:
            Dictionary of class weights
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        # Get class labels from generator
        class_labels = train_generator.classes
        
        # Compute class weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(class_labels),
            y=class_labels
        )
        
        return dict(enumerate(class_weights))


def create_sample_data_structure():
    """
    Create sample data directory structure for users to organize their data.
    
    Expected structure:
    data/
    ├── raw/
    │   ├── Crack/
    │   ├── Missing_Screw_Head/
    │   └── Paint_Degradation/
    └── processed/
    """
    base_dir = 'data'
    raw_dir = os.path.join(base_dir, 'raw')
    
    # Create directories for each class
    class_dirs = ['Crack', 'Missing_Screw_Head', 'Paint_Degradation']
    
    for class_name in class_dirs:
        class_path = os.path.join(raw_dir, class_name)
        os.makedirs(class_path, exist_ok=True)
    
    print("Data directory structure created:")
    print(f"{base_dir}/")
    print("├── raw/")
    for class_name in class_dirs:
        print(f"│   ├── {class_name}/")
    print("└── processed/")
    
    return raw_dir


def verify_data_structure(data_dir):
    """
    Verify that the data directory has the correct structure.
    
    Args:
        data_dir: Path to the data directory
        
    Returns:
        True if structure is valid, False otherwise
    """
    class_names = ['Crack', 'Missing_Screw_Head', 'Paint_Degradation']
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} does not exist.")
        return False
    
    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        if not os.path.exists(class_path):
            print(f"Warning: Class directory {class_name} not found in {data_dir}")
            return False
        
        # Check if directory has images
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        print(f"Found {len(images)} images in {class_name}")
    
    return True
