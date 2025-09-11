"""
Metrics and evaluation utilities for HIIT methylation analysis.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from scipy import stats
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import warnings


def methylation_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    task_type: str = 'classification'
) -> Dict[str, float]:
    """
    Calculate comprehensive metrics for methylation analysis.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray  
        Predicted values
    task_type : str
        'classification' or 'regression'
        
    Returns:
    --------
    Dict[str, float]
        Dictionary of evaluation metrics
    """
    
    if task_type == 'classification':
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
        
        # ROC AUC for binary classification
        if len(np.unique(y_true)) == 2:
            if hasattr(y_pred, 'shape') and y_pred.ndim > 1:
                # Probabilities provided
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred[:, 1])
            else:
                # Binary predictions - convert to probabilities
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
                    except ValueError:
                        pass
                        
    else:  # regression
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        
        metrics = {
            'r2_score': r2_score(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred)
        }
        
        # Pearson correlation
        correlation, _ = stats.pearsonr(y_true, y_pred)
        metrics['pearson_r'] = correlation
        
    return metrics


def hiit_response_score(
    pre_methylation: pd.DataFrame,
    post_methylation: pd.DataFrame,
    fitness_change: Optional[pd.Series] = None,
    weight_methylation: float = 0.7
) -> pd.Series:
    """
    Calculate HIIT response score based on methylation changes and fitness improvement.
    
    Parameters:
    -----------
    pre_methylation : pd.DataFrame
        Pre-intervention methylation data
    post_methylation : pd.DataFrame
        Post-intervention methylation data  
    fitness_change : pd.Series, optional
        Fitness improvement scores
    weight_methylation : float
        Weight for methylation component (0-1)
        
    Returns:
    --------
    pd.Series
        HIIT response scores for each sample
    """
    
    # Calculate methylation change magnitude
    delta_methylation = post_methylation - pre_methylation
    methylation_response = np.abs(delta_methylation).mean(axis=1)
    
    # Normalize methylation response to 0-1 scale
    methylation_response = (methylation_response - methylation_response.min()) / \
                          (methylation_response.max() - methylation_response.min())
    
    if fitness_change is not None:
        # Combine methylation and fitness changes
        fitness_normalized = (fitness_change - fitness_change.min()) / \
                           (fitness_change.max() - fitness_change.min())
        
        response_score = (weight_methylation * methylation_response + 
                         (1 - weight_methylation) * fitness_normalized)
    else:
        response_score = methylation_response
    
    return response_score


def calculate_differential_methylation(
    group1_methylation: pd.DataFrame,
    group2_methylation: pd.DataFrame,
    method: str = 'ttest',
    p_value_threshold: float = 0.05,
    effect_size_threshold: float = 0.1
) -> pd.DataFrame:
    """
    Calculate differential methylation between two groups.
    
    Parameters:
    -----------
    group1_methylation : pd.DataFrame
        Methylation data for group 1
    group2_methylation : pd.DataFrame
        Methylation data for group 2
    method : str
        Statistical test method ('ttest', 'mannwhitney')
    p_value_threshold : float
        P-value threshold for significance
    effect_size_threshold : float
        Minimum effect size threshold
        
    Returns:
    --------
    pd.DataFrame
        Differential methylation results
    """
    
    results = []
    
    # Get common CpG sites
    common_sites = group1_methylation.columns.intersection(group2_methylation.columns)
    
    for cpg_site in common_sites:
        values1 = group1_methylation[cpg_site].dropna()
        values2 = group2_methylation[cpg_site].dropna()
        
        if len(values1) < 3 or len(values2) < 3:
            continue
            
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(values1) - 1) * values1.var() + 
                             (len(values2) - 1) * values2.var()) / 
                            (len(values1) + len(values2) - 2))
        
        if pooled_std > 0:
            cohens_d = (values1.mean() - values2.mean()) / pooled_std
        else:
            cohens_d = 0
        
        # Statistical test
        if method == 'ttest':
            statistic, p_value = stats.ttest_ind(values1, values2)
        elif method == 'mannwhitney':
            statistic, p_value = stats.mannwhitneyu(values1, values2, 
                                                  alternative='two-sided')
        else:
            raise ValueError(f"Unknown method: {method}")
        
        results.append({
            'cpg_site': cpg_site,
            'group1_mean': values1.mean(),
            'group2_mean': values2.mean(),
            'mean_difference': values1.mean() - values2.mean(),
            'cohens_d': cohens_d,
            'p_value': p_value,
            'statistic': statistic,
            'significant': (p_value < p_value_threshold) and (abs(cohens_d) > effect_size_threshold)
        })
    
    results_df = pd.DataFrame(results)
    
    # Multiple testing correction (Bonferroni)
    if len(results_df) > 0:
        results_df['p_value_corrected'] = results_df['p_value'] * len(results_df)
        results_df['p_value_corrected'] = results_df['p_value_corrected'].clip(upper=1.0)
        results_df['significant_corrected'] = (
            (results_df['p_value_corrected'] < p_value_threshold) & 
            (np.abs(results_df['cohens_d']) > effect_size_threshold)
        )
    
    return results_df.sort_values('p_value')


def validate_methylation_data(methylation_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate methylation data quality and return summary statistics.
    
    Parameters:
    -----------
    methylation_df : pd.DataFrame
        Methylation data to validate
        
    Returns:
    --------
    Dict[str, Any]
        Validation results and quality metrics
    """
    
    results = {
        'shape': methylation_df.shape,
        'missing_values': methylation_df.isnull().sum().sum(),
        'missing_percentage': (methylation_df.isnull().sum().sum() / methylation_df.size) * 100
    }
    
    # Check value ranges
    numeric_data = methylation_df.select_dtypes(include=[np.number])
    if len(numeric_data.columns) > 0:
        results['min_value'] = numeric_data.min().min()
        results['max_value'] = numeric_data.max().max()
        results['values_in_range'] = ((numeric_data >= 0) & (numeric_data <= 1)).all().all()
        
        # Detect potential outliers
        outlier_threshold = 3
        z_scores = np.abs(stats.zscore(numeric_data, nan_policy='omit'))
        results['outliers_count'] = (z_scores > outlier_threshold).sum().sum()
        
        # Distribution statistics
        results['mean_methylation'] = numeric_data.mean().mean()
        results['std_methylation'] = numeric_data.std().mean()
        results['median_methylation'] = numeric_data.median().mean()
    
    # Sample-level statistics
    results['samples_with_missing'] = (methylation_df.isnull().any(axis=1)).sum()
    results['cpg_sites_with_missing'] = (methylation_df.isnull().any(axis=0)).sum()
    
    # Variance analysis
    site_variance = numeric_data.var()
    results['low_variance_sites'] = (site_variance < 0.001).sum()
    results['zero_variance_sites'] = (site_variance == 0).sum()
    
    return results


def calculate_methylation_stability(
    methylation_df: pd.DataFrame,
    time_points: pd.Series
) -> pd.DataFrame:
    """
    Calculate methylation stability across time points.
    
    Parameters:
    -----------
    methylation_df : pd.DataFrame
        Methylation data
    time_points : pd.Series
        Time point labels for each sample
        
    Returns:
    --------
    pd.DataFrame
        Stability metrics for each CpG site
    """
    
    stability_results = []
    
    for cpg_site in methylation_df.columns:
        site_data = pd.DataFrame({
            'methylation': methylation_df[cpg_site],
            'time_point': time_points
        }).dropna()
        
        # Calculate coefficient of variation within each time point
        cv_by_timepoint = site_data.groupby('time_point')['methylation'].agg([
            lambda x: x.std() / x.mean() if x.mean() != 0 else np.nan
        ]).iloc[:, 0]
        
        # Overall stability (inverse of CV)
        overall_cv = site_data['methylation'].std() / site_data['methylation'].mean() \
                    if site_data['methylation'].mean() != 0 else np.nan
        
        stability_results.append({
            'cpg_site': cpg_site,
            'overall_cv': overall_cv,
            'mean_cv_within_timepoints': cv_by_timepoint.mean(),
            'stability_score': 1 / (1 + overall_cv) if not np.isnan(overall_cv) else np.nan
        })
    
    return pd.DataFrame(stability_results)