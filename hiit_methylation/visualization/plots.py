"""
Plotting functions for HIIT DNA methylation analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Any, Tuple
import warnings


def plot_methylation_patterns(
    methylation_data: pd.DataFrame,
    metadata: pd.DataFrame,
    group_column: str = 'hiit_group',
    time_column: str = 'time_point',
    sample_cpg_sites: int = 50,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot methylation patterns across groups and time points.
    
    Parameters:
    -----------
    methylation_data : pd.DataFrame
        Methylation data
    metadata : pd.DataFrame
        Sample metadata
    group_column : str
        Column name for grouping variable
    time_column : str
        Column name for time variable
    sample_cpg_sites : int
        Number of CpG sites to sample for visualization
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    
    # Sample CpG sites for visualization
    if methylation_data.shape[1] > sample_cpg_sites:
        selected_sites = np.random.choice(
            methylation_data.columns, 
            size=sample_cpg_sites, 
            replace=False
        )
        plot_data = methylation_data[selected_sites]
    else:
        plot_data = methylation_data
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('DNA Methylation Patterns in HIIT Study', fontsize=16)
    
    # 1. Distribution of methylation values
    ax1 = axes[0, 0]
    plot_data.mean(axis=1).hist(bins=30, ax=ax1, alpha=0.7)
    ax1.set_title('Distribution of Mean Methylation')
    ax1.set_xlabel('Mean β-value')
    ax1.set_ylabel('Frequency')
    
    # 2. Methylation by group
    ax2 = axes[0, 1]
    if group_column in metadata.columns:
        for group in metadata[group_column].unique():
            group_samples = metadata[metadata[group_column] == group].index
            common_samples = plot_data.index.intersection(group_samples)
            if len(common_samples) > 0:
                group_data = plot_data.loc[common_samples].mean(axis=1)
                ax2.hist(group_data, alpha=0.7, label=group, bins=20)
        ax2.set_title(f'Methylation Distribution by {group_column}')
        ax2.set_xlabel('Mean β-value')
        ax2.set_ylabel('Frequency')
        ax2.legend()
    
    # 3. Methylation by time point
    ax3 = axes[1, 0]
    if time_column in metadata.columns:
        for timepoint in metadata[time_column].unique():
            time_samples = metadata[metadata[time_column] == timepoint].index
            common_samples = plot_data.index.intersection(time_samples)
            if len(common_samples) > 0:
                time_data = plot_data.loc[common_samples].mean(axis=1)
                ax3.hist(time_data, alpha=0.7, label=timepoint, bins=20)
        ax3.set_title(f'Methylation Distribution by {time_column}')
        ax3.set_xlabel('Mean β-value')
        ax3.set_ylabel('Frequency')
        ax3.legend()
    
    # 4. CpG site variance
    ax4 = axes[1, 1]
    site_variance = plot_data.var()
    ax4.hist(site_variance, bins=30, alpha=0.7)
    ax4.set_title('CpG Site Variance Distribution')
    ax4.set_xlabel('Variance')
    ax4.set_ylabel('Frequency')
    
    plt.tight_layout()
    return fig


def plot_hiit_effects(
    pre_methylation: pd.DataFrame,
    post_methylation: pd.DataFrame,
    metadata: pd.DataFrame,
    top_sites: int = 20,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Plot HIIT intervention effects on DNA methylation.
    
    Parameters:
    -----------
    pre_methylation : pd.DataFrame
        Pre-intervention methylation data
    post_methylation : pd.DataFrame
        Post-intervention methylation data
    metadata : pd.DataFrame
        Sample metadata
    top_sites : int
        Number of top differential sites to show
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    
    # Calculate methylation changes
    common_samples = pre_methylation.index.intersection(post_methylation.index)
    common_sites = pre_methylation.columns.intersection(post_methylation.columns)
    
    pre_data = pre_methylation.loc[common_samples, common_sites]
    post_data = post_methylation.loc[common_samples, common_sites]
    delta_methylation = post_data - pre_data
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('HIIT Effects on DNA Methylation', fontsize=16)
    
    # 1. Overall methylation changes
    ax1 = axes[0, 0]
    overall_changes = delta_methylation.mean(axis=1)
    ax1.hist(overall_changes, bins=30, alpha=0.7)
    ax1.axvline(0, color='red', linestyle='--', alpha=0.8)
    ax1.set_title('Distribution of Methylation Changes')
    ax1.set_xlabel('Δ β-value (Post - Pre)')
    ax1.set_ylabel('Frequency')
    
    # 2. Site-wise changes
    ax2 = axes[0, 1]
    site_changes = delta_methylation.mean(axis=0)
    top_increasing = site_changes.nlargest(top_sites//2)
    top_decreasing = site_changes.nsmallest(top_sites//2)
    
    x_pos = np.arange(len(top_increasing) + len(top_decreasing))
    values = pd.concat([top_decreasing, top_increasing])
    colors = ['blue'] * len(top_decreasing) + ['red'] * len(top_increasing)
    
    bars = ax2.bar(x_pos, values, color=colors, alpha=0.7)
    ax2.set_title(f'Top {top_sites} Differential CpG Sites')
    ax2.set_xlabel('CpG Sites')
    ax2.set_ylabel('Mean Δ β-value')
    ax2.axhline(0, color='black', linestyle='-', alpha=0.5)
    
    # 3. Group comparison if available
    ax3 = axes[1, 0]
    if 'hiit_group' in metadata.columns:
        for group in metadata['hiit_group'].unique():
            group_samples = metadata[metadata['hiit_group'] == group].index
            group_common = delta_methylation.index.intersection(group_samples)
            if len(group_common) > 0:
                group_changes = delta_methylation.loc[group_common].mean(axis=1)
                ax3.hist(group_changes, alpha=0.7, label=group, bins=20)
        ax3.set_title('Methylation Changes by Group')
        ax3.set_xlabel('Mean Δ β-value')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.axvline(0, color='red', linestyle='--', alpha=0.8)
    
    # 4. Correlation between pre and change
    ax4 = axes[1, 1]
    pre_means = pre_data.mean(axis=1)
    change_means = delta_methylation.mean(axis=1)
    ax4.scatter(pre_means, change_means, alpha=0.6)
    ax4.set_xlabel('Pre-intervention Mean β-value')
    ax4.set_ylabel('Mean Δ β-value')
    ax4.set_title('Pre-intervention vs Change')
    
    # Add correlation coefficient
    correlation = np.corrcoef(pre_means, change_means)[0, 1]
    ax4.text(0.05, 0.95, f'r = {correlation:.3f}', 
             transform=ax4.transAxes, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    
    plt.tight_layout()
    return fig


def plot_model_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str = 'classification',
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot model performance metrics and diagnostics.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    task_type : str
        'classification' or 'regression'
    class_names : List[str], optional
        Class names for classification
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    
    if task_type == 'classification':
        return _plot_classification_performance(y_true, y_pred, class_names, figsize)
    else:
        return _plot_regression_performance(y_true, y_pred, figsize)


def _plot_classification_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """Plot classification performance."""
    
    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.metrics import roc_curve, auc
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Classification Model Performance', fontsize=16)
    
    # 1. Confusion Matrix
    ax1 = axes[0, 0]
    cm = confusion_matrix(y_true, y_pred)
    im = ax1.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax1.set_title('Confusion Matrix')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    if class_names:
        ax1.set_xticks(range(len(class_names)))
        ax1.set_yticks(range(len(class_names)))
        ax1.set_xticklabels(class_names)
        ax1.set_yticklabels(class_names)
    
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # 2. ROC Curve (for binary classification)
    ax2 = axes[0, 1]
    if len(np.unique(y_true)) == 2:
        try:
            if y_pred.ndim > 1:
                # Probabilities provided
                fpr, tpr, _ = roc_curve(y_true, y_pred[:, 1])
            else:
                # Binary predictions
                fpr, tpr, _ = roc_curve(y_true, y_pred)
            
            roc_auc = auc(fpr, tpr)
            ax2.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
            ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax2.set_xlim([0.0, 1.0])
            ax2.set_ylim([0.0, 1.05])
            ax2.set_xlabel('False Positive Rate')
            ax2.set_ylabel('True Positive Rate')
            ax2.set_title('ROC Curve')
            ax2.legend(loc="lower right")
        except Exception as e:
            ax2.text(0.5, 0.5, 'ROC curve not available', 
                    ha='center', va='center', transform=ax2.transAxes)
    else:
        ax2.text(0.5, 0.5, 'ROC curve only for binary classification', 
                ha='center', va='center', transform=ax2.transAxes)
    
    # 3. Prediction distribution
    ax3 = axes[1, 0]
    if y_pred.ndim == 1:
        for class_label in np.unique(y_true):
            mask = y_true == class_label
            label_name = class_names[int(class_label)] if class_names else f'Class {class_label}'
            ax3.hist(y_pred[mask], alpha=0.7, label=label_name, bins=20)
    ax3.set_title('Prediction Distribution by True Class')
    ax3.set_xlabel('Predicted Value')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    
    # 4. Performance metrics text
    ax4 = axes[1, 1]
    ax4.axis('off')
    report = classification_report(y_true, y_pred, target_names=class_names)
    ax4.text(0.1, 0.9, report, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    return fig


def _plot_regression_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """Plot regression performance."""
    
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Regression Model Performance', fontsize=16)
    
    # 1. Predicted vs Actual
    ax1 = axes[0, 0]
    ax1.scatter(y_true, y_pred, alpha=0.6)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    ax1.set_xlabel('Actual Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title('Predicted vs Actual')
    
    # Add R² score
    r2 = r2_score(y_true, y_pred)
    ax1.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax1.transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    
    # 2. Residuals plot
    ax2 = axes[0, 1]
    residuals = y_true - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.6)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residuals vs Predicted')
    
    # 3. Residuals distribution
    ax3 = axes[1, 0]
    ax3.hist(residuals, bins=30, alpha=0.7)
    ax3.axvline(x=0, color='r', linestyle='--')
    ax3.set_xlabel('Residuals')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Residuals Distribution')
    
    # 4. Performance metrics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    metrics_text = f"""
Performance Metrics:
─────────────────────
R² Score:    {r2:.4f}
MSE:         {mse:.4f}
RMSE:        {rmse:.4f}
MAE:         {mae:.4f}

Sample Statistics:
─────────────────
Mean Actual:     {y_true.mean():.4f}
Mean Predicted:  {y_pred.mean():.4f}
Std Actual:      {y_true.std():.4f}
Std Predicted:   {y_pred.std():.4f}
    """
    
    ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    return fig


def plot_feature_importance(
    importance_scores: pd.Series,
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot feature importance scores.
    
    Parameters:
    -----------
    importance_scores : pd.Series
        Feature importance scores
    top_n : int
        Number of top features to display
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    
    # Get top features
    top_features = importance_scores.nlargest(top_n)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create horizontal bar plot
    y_pos = np.arange(len(top_features))
    bars = ax.barh(y_pos, top_features.values, alpha=0.7)
    
    # Customize plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features.index)
    ax.set_xlabel('Importance Score')
    ax.set_title(f'Top {top_n} Most Important Features')
    
    # Add value labels
    for i, (idx, val) in enumerate(top_features.items()):
        ax.text(val + 0.01 * max(top_features), i, f'{val:.3f}', 
                va='center', ha='left', fontsize=9)
    
    # Invert y-axis to show highest importance at top
    ax.invert_yaxis()
    
    plt.tight_layout()
    return fig