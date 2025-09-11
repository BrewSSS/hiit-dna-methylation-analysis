"""
Utility functions for HIIT methylation analysis.
"""

from .metrics import methylation_metrics, hiit_response_score
from .validation import cross_validate_methylation, stratified_split_hiit
from .helpers import filter_cpg_sites, calculate_delta_methylation

__all__ = [
    'methylation_metrics',
    'hiit_response_score',
    'cross_validate_methylation',
    'stratified_split_hiit',
    'filter_cpg_sites',
    'calculate_delta_methylation'
]

# Convenience imports
metrics = methylation_metrics
validation = cross_validate_methylation