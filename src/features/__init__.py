"""
Feature selection module implementing the Ten-Level Framework.

This module provides a comprehensive feature selection framework for DNA methylation
biomarker identification, featuring graduated stringency levels for robust and
reproducible feature selection across different analysis contexts.

The Ten-Level Framework supports:
- Binary classification (e.g., HIIT intervention vs Control)
- Multiclass classification (e.g., training duration categories)
- Time-series trajectory analysis (temporal methylation changes)

Key Components:
- TenLevelFeatureSelector: Main orchestrator for the framework
- FeatureSelectionConfig: Configuration management for selection thresholds
- StatisticalFeatureSelector: Statistical testing methods (t-test, ANOVA)
- LassoFeatureSelector: L1 regularization-based selection
- ElasticNetFeatureSelector: Combined L1/L2 regularization
- TimeSeriesFeatureAnalyzer: Temporal pattern detection

Example:
    >>> from src.features import TenLevelFeatureSelector, FeatureSelectionConfig
    >>> config = FeatureSelectionConfig()
    >>> selector = TenLevelFeatureSelector(config)
    >>> features = selector.select_binary_features(data, labels, level="L5_moderate")
"""

from .selection import TenLevelFeatureSelector, FeatureSelectionConfig
from .statistical import (
    StatisticalFeatureSelector,
    run_ttest,
    run_anova,
    calculate_effect_size,
    adjust_pvalues
)
from .ml_selection import (
    LassoFeatureSelector,
    ElasticNetFeatureSelector,
    RandomForestFeatureSelector
)
from .time_series import TimeSeriesFeatureAnalyzer

__all__ = [
    # Core framework
    'TenLevelFeatureSelector',
    'FeatureSelectionConfig',
    # Statistical methods
    'StatisticalFeatureSelector',
    'run_ttest',
    'run_anova',
    'calculate_effect_size',
    'adjust_pvalues',
    # ML-based methods
    'LassoFeatureSelector',
    'ElasticNetFeatureSelector',
    'RandomForestFeatureSelector',
    # Time-series analysis
    'TimeSeriesFeatureAnalyzer'
]

__version__ = '1.0.0'
