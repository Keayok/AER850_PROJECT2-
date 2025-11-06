"""
AER850 Project 2 - Aircraft Skin Defect Classification

A Deep Convolutional Neural Network (DCNN) system for automated
classification of aircraft skin defects.
"""

__version__ = '1.0.0'
__author__ = 'AER850 Student'

from .model import DefectClassifierDCNN
from .data_loader import DefectDataLoader

__all__ = ['DefectClassifierDCNN', 'DefectDataLoader']
