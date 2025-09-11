"""
Data loading utilities for HIIT DNA methylation studies.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
from pathlib import Path


def load_methylation_data(
    filepath: str,
    sample_sheet: Optional[str] = None,
    transpose: bool = True
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load DNA methylation data from file.
    
    Parameters:
    -----------
    filepath : str
        Path to methylation data file (CSV, Excel, or HDF5)
    sample_sheet : str, optional
        Path to sample metadata file
    transpose : bool
        Whether to transpose data (True if CpG sites are in rows)
        
    Returns:
    --------
    Tuple[pd.DataFrame, Optional[pd.DataFrame]]
        Methylation data and sample metadata
    """
    
    filepath = Path(filepath)
    
    # Load methylation data based on file extension
    if filepath.suffix.lower() == '.csv':
        methylation_df = pd.read_csv(filepath, index_col=0)
    elif filepath.suffix.lower() in ['.xlsx', '.xls']:
        methylation_df = pd.read_excel(filepath, index_col=0)
    elif filepath.suffix.lower() == '.h5':
        methylation_df = pd.read_hdf(filepath, key='methylation')
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    # Transpose if needed (samples as rows, CpG sites as columns)
    if transpose:
        methylation_df = methylation_df.T
        
    print(f"Loaded methylation data: {methylation_df.shape}")
    
    # Load sample metadata if provided
    metadata_df = None
    if sample_sheet:
        metadata_df = load_hiit_metadata(sample_sheet)
        
        # Ensure sample IDs match
        common_samples = methylation_df.index.intersection(metadata_df.index)
        methylation_df = methylation_df.loc[common_samples]
        metadata_df = metadata_df.loc[common_samples]
        
        print(f"Matched samples with metadata: {len(common_samples)}")
    
    return methylation_df, metadata_df


def load_hiit_metadata(filepath: str) -> pd.DataFrame:
    """
    Load HIIT study metadata.
    
    Parameters:
    -----------
    filepath : str
        Path to metadata file
        
    Returns:
    --------
    pd.DataFrame
        Sample metadata with standardized columns
    """
    
    filepath = Path(filepath)
    
    if filepath.suffix.lower() == '.csv':
        metadata_df = pd.read_csv(filepath, index_col=0)
    elif filepath.suffix.lower() in ['.xlsx', '.xls']:
        metadata_df = pd.read_excel(filepath, index_col=0)
    else:
        raise ValueError(f"Unsupported metadata format: {filepath.suffix}")
    
    # Standardize column names
    column_mapping = {
        'intervention': 'hiit_group',
        'group': 'hiit_group', 
        'treatment': 'hiit_group',
        'timepoint': 'time_point',
        'time': 'time_point',
        'visit': 'time_point',
        'sex': 'gender',
        'bmi': 'BMI'
    }
    
    for old_name, new_name in column_mapping.items():
        if old_name in metadata_df.columns and new_name not in metadata_df.columns:
            metadata_df = metadata_df.rename(columns={old_name: new_name})
    
    # Ensure required columns exist
    required_columns = ['hiit_group', 'time_point']
    for col in required_columns:
        if col not in metadata_df.columns:
            print(f"Warning: Required column '{col}' not found in metadata")
    
    return metadata_df


def create_sample_data(
    n_samples: int = 100,
    n_cpg_sites: int = 1000,
    hiit_effect_size: float = 0.3,
    random_state: Optional[int] = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create synthetic DNA methylation data for testing and demonstration.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    n_cpg_sites : int
        Number of CpG sites to generate
    hiit_effect_size : float
        Effect size of HIIT intervention on methylation
    random_state : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        Synthetic methylation data and metadata
    """
    
    if random_state:
        np.random.seed(random_state)
    
    # Generate sample IDs
    sample_ids = [f"Sample_{i:03d}" for i in range(n_samples)]
    cpg_ids = [f"cg{i:08d}" for i in range(n_cpg_sites)]
    
    # Create metadata
    metadata = pd.DataFrame({
        'hiit_group': np.random.choice(['Control', 'HIIT'], n_samples),
        'time_point': np.random.choice(['Pre', 'Post'], n_samples),
        'age': np.random.normal(35, 10, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'BMI': np.random.normal(25, 5, n_samples)
    }, index=sample_ids)
    
    # Generate baseline methylation values
    methylation_data = np.random.beta(2, 2, (n_samples, n_cpg_sites))
    
    # Add HIIT effects to subset of CpG sites
    n_affected_sites = int(n_cpg_sites * 0.1)  # 10% of sites affected
    affected_sites = np.random.choice(n_cpg_sites, n_affected_sites, replace=False)
    
    for i, sample_id in enumerate(sample_ids):
        if (metadata.loc[sample_id, 'hiit_group'] == 'HIIT' and 
            metadata.loc[sample_id, 'time_point'] == 'Post'):
            # Add HIIT effect
            effect = np.random.normal(hiit_effect_size, 0.1, n_affected_sites)
            methylation_data[i, affected_sites] += effect
    
    # Ensure values stay in [0, 1] range
    methylation_data = np.clip(methylation_data, 0.001, 0.999)
    
    methylation_df = pd.DataFrame(
        methylation_data,
        index=sample_ids,
        columns=cpg_ids
    )
    
    print(f"Generated synthetic data: {methylation_df.shape}")
    print(f"HIIT-affected CpG sites: {n_affected_sites}")
    
    return methylation_df, metadata