"""
Validation utilities for HIIT methylation analysis.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from sklearn.model_selection import (
    cross_val_score, 
    StratifiedKFold, 
    KFold,
    train_test_split
)
from sklearn.base import BaseEstimator


def cross_validate_methylation(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 5,
    scoring: Optional[str] = None,
    stratify: bool = True
) -> Dict[str, Any]:
    """
    Perform cross-validation for methylation analysis models.
    
    Parameters:
    -----------
    model : BaseEstimator
        Scikit-learn model to validate
    X : pd.DataFrame
        Feature matrix (methylation data)
    y : pd.Series
        Target variable
    cv_folds : int
        Number of cross-validation folds
    scoring : str, optional
        Scoring metric
    stratify : bool
        Whether to use stratified CV for classification
        
    Returns:
    --------
    Dict[str, Any]
        Cross-validation results
    """
    
    # Choose appropriate CV strategy
    if stratify and hasattr(model, '_estimator_type') and model._estimator_type == 'classifier':
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    
    return {
        'scores': cv_scores,
        'mean_score': cv_scores.mean(),
        'std_score': cv_scores.std(),
        'min_score': cv_scores.min(),
        'max_score': cv_scores.max()
    }


def stratified_split_hiit(
    methylation_data: pd.DataFrame,
    metadata: pd.DataFrame,
    test_size: float = 0.2,
    random_state: Optional[int] = 42,
    stratify_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/test split for HIIT studies.
    
    Parameters:
    -----------
    methylation_data : pd.DataFrame
        Methylation data
    metadata : pd.DataFrame
        Sample metadata
    test_size : float
        Proportion of data for testing
    random_state : int, optional
        Random seed
    stratify_columns : List[str], optional
        Columns to stratify on (default: ['hiit_group', 'time_point'])
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        X_train, X_test, metadata_train, metadata_test
    """
    
    if stratify_columns is None:
        stratify_columns = ['hiit_group', 'time_point']
    
    # Create stratification variable
    available_columns = [col for col in stratify_columns if col in metadata.columns]
    
    if available_columns:
        stratify_var = metadata[available_columns[0]].astype(str)
        for col in available_columns[1:]:
            stratify_var += '_' + metadata[col].astype(str)
    else:
        stratify_var = None
        print("Warning: No stratification columns found, using random split")
    
    # Perform split
    indices_train, indices_test = train_test_split(
        methylation_data.index,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_var
    )
    
    X_train = methylation_data.loc[indices_train]
    X_test = methylation_data.loc[indices_test]
    metadata_train = metadata.loc[indices_train]
    metadata_test = metadata.loc[indices_test]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    if available_columns:
        print("Training set distribution:")
        for col in available_columns:
            print(f"  {col}: {metadata_train[col].value_counts().to_dict()}")
        
        print("Testing set distribution:")
        for col in available_columns:
            print(f"  {col}: {metadata_test[col].value_counts().to_dict()}")
    
    return X_train, X_test, metadata_train, metadata_test


def validate_hiit_study_design(metadata: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate HIIT study design and sample distribution.
    
    Parameters:
    -----------
    metadata : pd.DataFrame
        Study metadata
        
    Returns:
    --------
    Dict[str, Any]
        Validation results
    """
    
    results = {'is_valid': True, 'warnings': [], 'errors': []}
    
    # Check required columns
    required_columns = ['hiit_group', 'time_point']
    for col in required_columns:
        if col not in metadata.columns:
            results['errors'].append(f"Missing required column: {col}")
            results['is_valid'] = False
    
    if not results['is_valid']:
        return results
    
    # Check group distribution
    group_counts = metadata['hiit_group'].value_counts()
    if len(group_counts) < 2:
        results['errors'].append("Need at least 2 groups for comparison")
        results['is_valid'] = False
    
    # Check for balanced design
    min_group_size = group_counts.min()
    max_group_size = group_counts.max()
    if max_group_size / min_group_size > 3:
        results['warnings'].append(f"Imbalanced groups: {group_counts.to_dict()}")
    
    # Check time points
    timepoint_counts = metadata['time_point'].value_counts()
    if len(timepoint_counts) < 2:
        results['warnings'].append("Only one time point detected")
    
    # Check for paired samples
    if 'subject_id' in metadata.columns:
        paired_subjects = metadata.groupby('subject_id')['time_point'].nunique()
        unpaired_subjects = (paired_subjects == 1).sum()
        if unpaired_subjects > 0:
            results['warnings'].append(f"{unpaired_subjects} subjects with only one time point")
    
    # Sample size recommendations
    total_samples = len(metadata)
    if total_samples < 20:
        results['warnings'].append("Small sample size may limit statistical power")
    elif total_samples < 50:
        results['warnings'].append("Consider larger sample size for better generalizability")
    
    results['summary'] = {
        'total_samples': total_samples,
        'groups': group_counts.to_dict(),
        'time_points': timepoint_counts.to_dict()
    }
    
    return results


def assess_batch_effects(
    methylation_data: pd.DataFrame,
    batch_info: pd.Series,
    n_pcs: int = 5
) -> Dict[str, Any]:
    """
    Assess potential batch effects in methylation data.
    
    Parameters:
    -----------
    methylation_data : pd.DataFrame
        Methylation data
    batch_info : pd.Series
        Batch information for each sample
    n_pcs : int
        Number of principal components to analyze
        
    Returns:
    --------
    Dict[str, Any]
        Batch effect assessment results
    """
    
    from sklearn.decomposition import PCA
    from scipy.stats import f_oneway
    
    # Perform PCA
    pca = PCA(n_components=n_pcs)
    pcs = pca.fit_transform(methylation_data.fillna(0))
    
    # Test association between PCs and batch
    batch_effects = {}
    
    for i in range(n_pcs):
        pc_values = pcs[:, i]
        
        # Group PC values by batch
        batch_groups = [pc_values[batch_info == batch] for batch in batch_info.unique()]
        
        # ANOVA test
        f_stat, p_value = f_oneway(*batch_groups)
        
        batch_effects[f'PC{i+1}'] = {
            'variance_explained': pca.explained_variance_ratio_[i],
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    # Overall assessment
    significant_pcs = sum(1 for pc in batch_effects.values() if pc['significant'])
    total_variance_affected = sum(
        pc['variance_explained'] for pc in batch_effects.values() if pc['significant']
    )
    
    results = {
        'batch_effects': batch_effects,
        'significant_pcs': significant_pcs,
        'total_variance_affected': total_variance_affected,
        'assessment': 'high' if total_variance_affected > 0.3 else 
                     'moderate' if total_variance_affected > 0.1 else 'low'
    }
    
    return results


def temporal_validation_split(
    methylation_data: pd.DataFrame,
    metadata: pd.DataFrame,
    time_column: str = 'time_point',
    subject_column: str = 'subject_id'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create temporal validation split for longitudinal HIIT studies.
    
    Parameters:
    -----------
    methylation_data : pd.DataFrame
        Methylation data
    metadata : pd.DataFrame
        Sample metadata
    time_column : str
        Column containing time point information
    subject_column : str
        Column containing subject IDs
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        X_train, X_test, metadata_train, metadata_test
    """
    
    if subject_column not in metadata.columns:
        raise ValueError(f"Subject column '{subject_column}' not found in metadata")
    
    # Split subjects (not samples) for temporal validation
    unique_subjects = metadata[subject_column].unique()
    np.random.seed(42)
    test_subjects = np.random.choice(unique_subjects, size=len(unique_subjects)//4, replace=False)
    
    # Create train/test splits based on subjects
    test_mask = metadata[subject_column].isin(test_subjects)
    train_mask = ~test_mask
    
    X_train = methylation_data.loc[train_mask]
    X_test = methylation_data.loc[test_mask]
    metadata_train = metadata.loc[train_mask]
    metadata_test = metadata.loc[test_mask]
    
    print(f"Temporal validation split:")
    print(f"Training subjects: {len(metadata_train[subject_column].unique())}")
    print(f"Testing subjects: {len(metadata_test[subject_column].unique())}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    return X_train, X_test, metadata_train, metadata_test