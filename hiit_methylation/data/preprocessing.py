"""
Data preprocessing utilities for DNA methylation analysis in HIIT studies.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings


def preprocess_methylation_data(
    methylation_df: pd.DataFrame,
    remove_na_threshold: float = 0.1,
    normalize_method: str = 'logit',
    filter_variance: bool = True,
    variance_threshold: float = 0.01
) -> pd.DataFrame:
    """
    Preprocess DNA methylation beta values for machine learning analysis.
    
    Parameters:
    -----------
    methylation_df : pd.DataFrame
        DataFrame with CpG sites as columns and samples as rows
    remove_na_threshold : float
        Remove CpG sites with missing values above this threshold (0.1 = 10%)
    normalize_method : str
        Normalization method: 'logit', 'zscore', 'robust', or 'none'
    filter_variance : bool
        Whether to filter low-variance CpG sites
    variance_threshold : float
        Minimum variance threshold for CpG sites
        
    Returns:
    --------
    pd.DataFrame
        Preprocessed methylation data
    """
    
    print(f"Input data shape: {methylation_df.shape}")
    
    # Remove CpG sites with too many missing values
    missing_ratio = methylation_df.isnull().mean()
    sites_to_keep = missing_ratio <= remove_na_threshold
    methylation_df = methylation_df.loc[:, sites_to_keep]
    
    print(f"After removing high-missing CpG sites: {methylation_df.shape}")
    
    # Impute remaining missing values with column mean
    methylation_df = methylation_df.fillna(methylation_df.mean())
    
    # Ensure beta values are in valid range [0, 1]
    methylation_df = methylation_df.clip(lower=0.001, upper=0.999)
    
    # Filter low-variance sites
    if filter_variance:
        site_variance = methylation_df.var()
        high_var_sites = site_variance >= variance_threshold
        methylation_df = methylation_df.loc[:, high_var_sites]
        print(f"After variance filtering: {methylation_df.shape}")
    
    # Apply normalization
    if normalize_method == 'logit':
        methylation_df = logit_transform(methylation_df)
    elif normalize_method == 'zscore':
        scaler = StandardScaler()
        methylation_df = pd.DataFrame(
            scaler.fit_transform(methylation_df),
            index=methylation_df.index,
            columns=methylation_df.columns
        )
    elif normalize_method == 'robust':
        scaler = RobustScaler()
        methylation_df = pd.DataFrame(
            scaler.fit_transform(methylation_df),
            index=methylation_df.index,
            columns=methylation_df.columns
        )
    
    print(f"Final preprocessed data shape: {methylation_df.shape}")
    
    return methylation_df


def normalize_beta_values(beta_values: pd.DataFrame, method: str = 'logit') -> pd.DataFrame:
    """
    Normalize DNA methylation beta values.
    
    Parameters:
    -----------
    beta_values : pd.DataFrame
        Beta values in range [0, 1]
    method : str
        Normalization method
        
    Returns:
    --------
    pd.DataFrame
        Normalized values
    """
    
    if method == 'logit':
        return logit_transform(beta_values)
    elif method == 'zscore':
        return (beta_values - beta_values.mean()) / beta_values.std()
    elif method == 'robust':
        median = beta_values.median()
        mad = np.abs(beta_values - median).median()
        return (beta_values - median) / mad
    else:
        return beta_values


def logit_transform(beta_values: pd.DataFrame) -> pd.DataFrame:
    """
    Apply logit transformation to beta values.
    
    Parameters:
    -----------
    beta_values : pd.DataFrame
        Beta values in range [0, 1]
        
    Returns:
    --------
    pd.DataFrame
        Logit-transformed values
    """
    
    # Ensure values are in valid range for logit
    beta_clipped = beta_values.clip(lower=0.001, upper=0.999)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logit_values = np.log(beta_clipped / (1 - beta_clipped))
    
    return pd.DataFrame(
        logit_values,
        index=beta_values.index,
        columns=beta_values.columns
    )


def filter_cpg_sites_by_annotation(
    methylation_df: pd.DataFrame,
    annotation_df: pd.DataFrame,
    regions: Optional[List[str]] = None,
    chromosomes: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Filter CpG sites by genomic annotation.
    
    Parameters:
    -----------
    methylation_df : pd.DataFrame
        Methylation data
    annotation_df : pd.DataFrame
        CpG site annotations with columns like 'chr', 'position', 'region'
    regions : List[str], optional
        Genomic regions to include (e.g., ['promoter', 'gene_body'])
    chromosomes : List[str], optional
        Chromosomes to include (e.g., ['chr1', 'chr2'])
        
    Returns:
    --------
    pd.DataFrame
        Filtered methylation data
    """
    
    # Get intersection of CpG sites
    common_sites = methylation_df.columns.intersection(annotation_df.index)
    
    # Filter by regions if specified
    if regions:
        region_mask = annotation_df['region'].isin(regions)
        common_sites = common_sites.intersection(
            annotation_df.loc[region_mask].index
        )
    
    # Filter by chromosomes if specified
    if chromosomes:
        chr_mask = annotation_df['chr'].isin(chromosomes)
        common_sites = common_sites.intersection(
            annotation_df.loc[chr_mask].index
        )
    
    return methylation_df[common_sites]