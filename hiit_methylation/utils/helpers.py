"""
Helper functions for HIIT methylation analysis.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple


def filter_cpg_sites(
    methylation_df: pd.DataFrame,
    min_variance: float = 0.01,
    max_missing_rate: float = 0.1,
    min_mean: float = 0.1,
    max_mean: float = 0.9
) -> pd.DataFrame:
    """
    Filter CpG sites based on quality criteria.
    
    Parameters:
    -----------
    methylation_df : pd.DataFrame
        Methylation data
    min_variance : float
        Minimum variance threshold
    max_missing_rate : float
        Maximum missing value rate
    min_mean : float
        Minimum mean methylation level
    max_mean : float
        Maximum mean methylation level
        
    Returns:
    --------
    pd.DataFrame
        Filtered methylation data
    """
    
    print(f"Starting with {methylation_df.shape[1]} CpG sites")
    
    # Filter by missing values
    missing_rate = methylation_df.isnull().mean()
    sites_low_missing = missing_rate <= max_missing_rate
    methylation_df = methylation_df.loc[:, sites_low_missing]
    print(f"After missing value filter: {methylation_df.shape[1]} sites")
    
    # Filter by variance
    site_variance = methylation_df.var()
    sites_high_variance = site_variance >= min_variance
    methylation_df = methylation_df.loc[:, sites_high_variance]
    print(f"After variance filter: {methylation_df.shape[1]} sites")
    
    # Filter by mean methylation level
    site_mean = methylation_df.mean()
    sites_good_mean = (site_mean >= min_mean) & (site_mean <= max_mean)
    methylation_df = methylation_df.loc[:, sites_good_mean]
    print(f"After mean filter: {methylation_df.shape[1]} sites")
    
    return methylation_df


def calculate_delta_methylation(
    pre_methylation: pd.DataFrame,
    post_methylation: pd.DataFrame,
    method: str = 'difference'
) -> pd.DataFrame:
    """
    Calculate methylation changes between time points.
    
    Parameters:
    -----------
    pre_methylation : pd.DataFrame
        Pre-intervention methylation data
    post_methylation : pd.DataFrame
        Post-intervention methylation data
    method : str
        Method for calculating change ('difference', 'ratio', 'log_ratio')
        
    Returns:
    --------
    pd.DataFrame
        Methylation changes
    """
    
    # Ensure same samples and sites
    common_samples = pre_methylation.index.intersection(post_methylation.index)
    common_sites = pre_methylation.columns.intersection(post_methylation.columns)
    
    pre_data = pre_methylation.loc[common_samples, common_sites]
    post_data = post_methylation.loc[common_samples, common_sites]
    
    if method == 'difference':
        delta = post_data - pre_data
    elif method == 'ratio':
        # Avoid division by zero
        delta = post_data / (pre_data + 1e-6)
    elif method == 'log_ratio':
        # Log fold change
        delta = np.log2((post_data + 1e-6) / (pre_data + 1e-6))
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return delta


def identify_hiit_responders(
    fitness_changes: pd.Series,
    methylation_changes: Optional[pd.DataFrame] = None,
    fitness_threshold: float = 0.1,
    methylation_threshold: Optional[float] = None
) -> pd.Series:
    """
    Identify HIIT responders based on fitness and/or methylation changes.
    
    Parameters:
    -----------
    fitness_changes : pd.Series
        Fitness improvement scores
    methylation_changes : pd.DataFrame, optional
        Methylation changes for each sample
    fitness_threshold : float
        Minimum fitness improvement for responder classification
    methylation_threshold : float, optional
        Minimum methylation change for responder classification
        
    Returns:
    --------
    pd.Series
        Boolean series indicating responder status
    """
    
    # Fitness-based responders
    fitness_responders = fitness_changes >= fitness_threshold
    
    if methylation_changes is not None and methylation_threshold is not None:
        # Calculate overall methylation response
        methylation_response = np.abs(methylation_changes).mean(axis=1)
        methylation_responders = methylation_response >= methylation_threshold
        
        # Combined criteria
        responders = fitness_responders & methylation_responders
    else:
        responders = fitness_responders
    
    return responders


def create_hiit_summary_stats(
    methylation_data: pd.DataFrame,
    metadata: pd.DataFrame
) -> Dict[str, Any]:
    """
    Create summary statistics for HIIT study data.
    
    Parameters:
    -----------
    methylation_data : pd.DataFrame
        Methylation data
    metadata : pd.DataFrame
        Study metadata
        
    Returns:
    --------
    Dict[str, Any]
        Summary statistics
    """
    
    summary = {
        'data_shape': methylation_data.shape,
        'missing_data_percent': (methylation_data.isnull().sum().sum() / methylation_data.size) * 100
    }
    
    # Sample distribution
    if 'hiit_group' in metadata.columns:
        summary['group_distribution'] = metadata['hiit_group'].value_counts().to_dict()
    
    if 'time_point' in metadata.columns:
        summary['timepoint_distribution'] = metadata['time_point'].value_counts().to_dict()
    
    # Methylation statistics
    summary['methylation_stats'] = {
        'mean': float(methylation_data.mean().mean()),
        'median': float(methylation_data.median().median()),
        'std': float(methylation_data.std().mean()),
        'min': float(methylation_data.min().min()),
        'max': float(methylation_data.max().max())
    }
    
    # CpG site variance distribution
    site_variance = methylation_data.var()
    summary['variance_stats'] = {
        'mean_variance': float(site_variance.mean()),
        'low_variance_sites': int((site_variance < 0.001).sum()),
        'high_variance_sites': int((site_variance > 0.1).sum())
    }
    
    return summary


def annotate_cpg_sites(
    cpg_list: List[str],
    annotation_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Annotate CpG sites with genomic information.
    
    Parameters:
    -----------
    cpg_list : List[str]
        List of CpG site IDs
    annotation_df : pd.DataFrame, optional
        External annotation data
        
    Returns:
    --------
    pd.DataFrame
        Annotated CpG sites
    """
    
    # Create basic annotation dataframe
    annotations = pd.DataFrame({'cpg_id': cpg_list})
    
    if annotation_df is not None:
        # Merge with external annotations
        annotations = annotations.merge(
            annotation_df, 
            left_on='cpg_id', 
            right_index=True, 
            how='left'
        )
    else:
        # Add placeholder annotations
        annotations['chromosome'] = 'Unknown'
        annotations['position'] = np.nan
        annotations['gene'] = 'Unknown'
        annotations['region'] = 'Unknown'
    
    return annotations


def calculate_epigenetic_age_acceleration(
    methylation_data: pd.DataFrame,
    chronological_age: pd.Series,
    epigenetic_clock: str = 'horvath'
) -> pd.Series:
    """
    Calculate epigenetic age acceleration (placeholder implementation).
    
    Parameters:
    -----------
    methylation_data : pd.DataFrame
        Methylation data
    chronological_age : pd.Series
        Chronological ages
    epigenetic_clock : str
        Type of epigenetic clock to use
        
    Returns:
    --------
    pd.Series
        Age acceleration values
    """
    
    # This is a simplified placeholder - real implementation would use 
    # specific CpG sites and coefficients for each clock
    
    # Simulate epigenetic age based on mean methylation
    mean_methylation = methylation_data.mean(axis=1)
    
    # Simple linear relationship (this would be replaced with actual clock)
    epigenetic_age = 20 + mean_methylation * 50 + np.random.normal(0, 5, len(mean_methylation))
    
    # Age acceleration = epigenetic age - chronological age
    age_acceleration = epigenetic_age - chronological_age
    
    return age_acceleration


def create_cpg_island_features(
    methylation_data: pd.DataFrame,
    cpg_island_annotations: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Create summary features for CpG islands.
    
    Parameters:
    -----------
    methylation_data : pd.DataFrame
        Methylation data
    cpg_island_annotations : pd.DataFrame, optional
        CpG island annotations
        
    Returns:
    --------
    pd.DataFrame
        CpG island summary features
    """
    
    if cpg_island_annotations is None:
        # Create dummy island groupings
        n_islands = min(100, methylation_data.shape[1] // 10)
        island_assignments = np.random.choice(
            range(n_islands), 
            size=methylation_data.shape[1]
        )
        island_df = pd.DataFrame({
            'cpg_site': methylation_data.columns,
            'island_id': island_assignments
        })
    else:
        island_df = cpg_island_annotations
    
    # Calculate summary statistics for each island
    island_features = []
    
    for island_id in island_df['island_id'].unique():
        island_sites = island_df[island_df['island_id'] == island_id]['cpg_site']
        island_sites = [site for site in island_sites if site in methylation_data.columns]
        
        if len(island_sites) > 0:
            island_data = methylation_data[island_sites]
            
            # Summary statistics
            island_mean = island_data.mean(axis=1)
            island_std = island_data.std(axis=1)
            island_median = island_data.median(axis=1)
            
            island_features.extend([
                (f'island_{island_id}_mean', island_mean),
                (f'island_{island_id}_std', island_std),
                (f'island_{island_id}_median', island_median)
            ])
    
    # Combine all features
    feature_df = pd.DataFrame(index=methylation_data.index)
    for feature_name, feature_values in island_features:
        feature_df[feature_name] = feature_values
    
    return feature_df