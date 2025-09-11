"""
HIIT DNA Methylation Analysis Package

A Python package for machine learning analysis of High-Intensity Interval Training (HIIT) 
intervention effects on DNA methylation patterns.
"""

__version__ = "0.1.0"
__author__ = "HIIT Methylation Analysis Team"

# Core module imports
from .data.preprocessing import preprocess_methylation_data
from .data.loaders import load_methylation_data, create_sample_data
from .models.classifiers import HIITMethylationClassifier
from .models.regressors import HIITMethylationRegressor
from .utils.metrics import methylation_metrics
from .utils.validation import cross_validate_methylation as validation
from .visualization.plots import plot_methylation_patterns, plot_hiit_effects

# Convenience aliases
preprocess = preprocess_methylation_data
metrics = methylation_metrics

__all__ = [
    'preprocess_methylation_data',
    'preprocess',
    'load_methylation_data',
    'create_sample_data',
    'HIITMethylationClassifier',
    'HIITMethylationRegressor',
    'methylation_metrics',
    'metrics',
    'validation',
    'plot_methylation_patterns',
    'plot_hiit_effects'
]