"""
Example Workflow Script

This script demonstrates a complete workflow for training and using
the aircraft skin defect classifier.
"""

import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model import DefectClassifierDCNN
from data_loader import DefectDataLoader, create_sample_data_structure, verify_data_structure


def setup_project():
    """Setup project directories and structure."""
    print("=" * 80)
    print("AER850 Project 2 - Setup")
    print("=" * 80)
    
    # Create data directory structure
    print("\nCreating data directory structure...")
    create_sample_data_structure()
    
    print("\n" + "=" * 80)
    print("Setup Complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Add your training images to the data/raw/ subdirectories:")
    print("   - data/raw/Crack/")
    print("   - data/raw/Missing_Screw_Head/")
    print("   - data/raw/Paint_Degradation/")
    print("\n2. Run training:")
    print("   python src/train.py --data_dir data/raw --model_type custom --epochs 50")
    print("\n3. Evaluate the model:")
    print("   python src/evaluate.py --model_path models/defect_classifier_custom_final.h5")
    print("\n4. Make predictions:")
    print("   python src/predict.py --model_path models/defect_classifier_custom_final.h5 --image path/to/image.jpg")


def demonstrate_model_architecture():
    """Demonstrate the model architecture."""
    print("\n" + "=" * 80)
    print("Model Architecture Demo")
    print("=" * 80)
    
    # Create custom model
    print("\n1. Custom DCNN Architecture:")
    print("-" * 80)
    classifier = DefectClassifierDCNN(input_shape=(224, 224, 3), num_classes=3)
    classifier.build_custom_cnn()
    classifier.compile_model()
    classifier.get_model_summary()
    
    # Create transfer learning model
    print("\n2. Transfer Learning Architecture (VGG16):")
    print("-" * 80)
    classifier_tl = DefectClassifierDCNN(input_shape=(224, 224, 3), num_classes=3)
    classifier_tl.build_transfer_learning_model(base_model_name='VGG16')
    classifier_tl.compile_model()
    classifier_tl.get_model_summary()


def verify_setup():
    """Verify that the setup is correct."""
    print("\n" + "=" * 80)
    print("Verifying Setup")
    print("=" * 80)
    
    data_dir = 'data/raw'
    
    if os.path.exists(data_dir):
        print(f"\nChecking data directory: {data_dir}")
        result = verify_data_structure(data_dir)
        
        if result:
            print("\n✓ Data structure is valid!")
            print("\nYou're ready to start training.")
        else:
            print("\n✗ Data structure is incomplete.")
            print("\nPlease add images to the class directories:")
            print("  - data/raw/Crack/")
            print("  - data/raw/Missing_Screw_Head/")
            print("  - data/raw/Paint_Degradation/")
    else:
        print(f"\n✗ Data directory {data_dir} does not exist.")
        print("\nRun setup first: python example_workflow.py setup")


def main():
    """Main function to run different demos."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='AER850 Project 2 - Example Workflow'
    )
    parser.add_argument(
        'action',
        choices=['setup', 'architecture', 'verify'],
        help='Action to perform'
    )
    
    args = parser.parse_args()
    
    if args.action == 'setup':
        setup_project()
    elif args.action == 'architecture':
        demonstrate_model_architecture()
    elif args.action == 'verify':
        verify_setup()


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) == 1:
        # No arguments provided, show usage
        print("=" * 80)
        print("AER850 Project 2 - Example Workflow")
        print("=" * 80)
        print("\nUsage: python example_workflow.py [action]")
        print("\nActions:")
        print("  setup        - Setup project directory structure")
        print("  architecture - Display model architecture")
        print("  verify       - Verify data setup")
        print("\nExample:")
        print("  python example_workflow.py setup")
    else:
        main()
