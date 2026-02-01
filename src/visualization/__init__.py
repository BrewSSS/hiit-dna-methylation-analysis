"""
Visualization module for HIIT methylation analysis.

This module provides publication-quality visualization functions for DNA methylation
data analysis, including dimensionality reduction plots, feature importance visualization,
differential methylation analysis, and publication figure generation.

Main Components
---------------
- Dimensionality reduction: PCA, t-SNE, UMAP visualizations
- Feature analysis: importance plots, volcano plots, heatmaps
- Statistical comparisons: boxplots, trajectory plots
- Model evaluation: ROC curves, confusion matrices
- Publication figures: PublicationFigureGenerator for journal-ready output

Examples
--------
>>> from src.visualization import plot_pca_visualization, PublicationFigureGenerator
>>> fig, ax, pca = plot_pca_visualization(data, labels)
>>>
>>> # Create publication-ready figures
>>> generator = PublicationFigureGenerator(style='nature')
>>> fig = generator.create_figure_methylation_landscape(data, sample_info)
>>> generator.save_figure(fig, 'figure1', formats=['pdf', 'png'])
"""

from .plots import (
    plot_pca_visualization,
    plot_tsne_visualization,
    plot_umap_visualization,
    plot_feature_importance,
    plot_volcano,
    plot_heatmap,
    plot_boxplot_comparison,
    plot_trajectory,
    plot_roc_curve,
    plot_roc_curves,
    plot_confusion_matrix
)
from .publication import PublicationFigureGenerator

__all__ = [
    # Dimensionality reduction
    'plot_pca_visualization',
    'plot_tsne_visualization',
    'plot_umap_visualization',
    # Feature analysis
    'plot_feature_importance',
    'plot_volcano',
    'plot_heatmap',
    # Statistical comparisons
    'plot_boxplot_comparison',
    'plot_trajectory',
    # Model evaluation
    'plot_roc_curve',
    'plot_roc_curves',
    'plot_confusion_matrix',
    # Publication figures
    'PublicationFigureGenerator'
]

__version__ = '1.0.0'
