"""
Machine learning models for HIIT methylation classification.

This module provides batch-aware classification models and evaluation tools
for DNA methylation data analysis in the context of High-Intensity Interval
Training (HIIT) studies.

Key Components:
    - ClassifierConfig: Configuration dataclass for classifier setup
    - BatchAwareClassifier: Classifier that handles batch effects as covariates
    - HIITClassificationPipeline: Complete analysis pipeline for HIIT studies
    - ModelEvaluator: Comprehensive model evaluation metrics
    - CrossValidationStrategy: Repeated stratified cross-validation
    - MultiVersionComparator: Compare models across preprocessing versions
"""

from .classifiers import (
    ClassifierConfig,
    BatchAwareClassifier,
    HIITClassificationPipeline
)
from .evaluation import ModelEvaluator, CrossValidationStrategy
from .multiversion import MultiVersionComparator

__all__ = [
    'ClassifierConfig',
    'BatchAwareClassifier',
    'HIITClassificationPipeline',
    'ModelEvaluator',
    'CrossValidationStrategy',
    'MultiVersionComparator'
]
