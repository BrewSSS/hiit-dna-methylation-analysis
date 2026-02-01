"""
Methylation data preprocessing module.

This module provides functionality for preprocessing DNA methylation data,
including variance filtering, missing value imputation, batch correction,
and standardization.
"""

import logging
from typing import Optional, Dict, List, Tuple, Union, Literal

import numpy as np
import pandas as pd
from scipy import stats

try:
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class MethylationPreprocessor:
    """
    A class for preprocessing DNA methylation beta value data.

    This class provides methods for quality control and preprocessing of
    methylation data including variance filtering, missing value handling,
    batch effect correction, and standardization.

    Attributes:
        std_threshold: Minimum standard deviation threshold for variance filtering.
        missing_threshold: Maximum allowed missing rate per probe.

    Example:
        >>> preprocessor = MethylationPreprocessor(std_threshold=0.02)
        >>> filtered_data = preprocessor.filter_low_variance(methylation_data)
        >>> imputed_data = preprocessor.handle_missing_values(filtered_data)
        >>> standardized = preprocessor.standardize(imputed_data)
    """

    def __init__(
        self,
        std_threshold: float = 0.02,
        missing_threshold: float = 0.2
    ) -> None:
        """
        Initialize the methylation preprocessor.

        Args:
            std_threshold: Minimum standard deviation for keeping probes.
                Probes with std < threshold are considered low variance.
            missing_threshold: Maximum fraction of missing values allowed
                per probe. Probes exceeding this are removed.

        Example:
            >>> preprocessor = MethylationPreprocessor(std_threshold=0.02)
        """
        self.std_threshold = std_threshold
        self.missing_threshold = missing_threshold
        self._fitted = False
        self._probe_stats: Optional[pd.DataFrame] = None
        self._scaler: Optional[StandardScaler] = None

    def filter_low_variance(
        self,
        data: pd.DataFrame,
        threshold: Optional[float] = None,
        return_stats: bool = False
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Remove CpG probes with low variance across samples.

        Low variance probes provide little discriminative information
        and can increase noise in downstream analyses.

        Args:
            data: DataFrame with probes as rows and samples as columns.
            threshold: Custom std threshold. If None, uses instance default.
            return_stats: If True, also return probe statistics.

        Returns:
            Filtered DataFrame with low variance probes removed.
            If return_stats is True, returns tuple of (filtered_data, stats_df).

        Example:
            >>> preprocessor = MethylationPreprocessor(std_threshold=0.02)
            >>> filtered = preprocessor.filter_low_variance(data)
            >>> print(f"Retained {filtered.shape[0]} of {data.shape[0]} probes")
        """
        threshold = threshold if threshold is not None else self.std_threshold

        logger.info(f"Filtering probes with std < {threshold}")
        original_count = data.shape[0]

        # Calculate statistics
        probe_stats = pd.DataFrame({
            'mean': data.mean(axis=1),
            'std': data.std(axis=1),
            'min': data.min(axis=1),
            'max': data.max(axis=1),
            'missing_rate': data.isna().mean(axis=1)
        })

        self._probe_stats = probe_stats

        # Filter by variance
        high_variance_mask = probe_stats['std'] >= threshold
        filtered_data = data.loc[high_variance_mask]

        removed_count = original_count - filtered_data.shape[0]
        logger.info(
            f"Removed {removed_count} low variance probes "
            f"({removed_count/original_count*100:.1f}%)"
        )
        logger.info(f"Retained {filtered_data.shape[0]} probes")

        if return_stats:
            return filtered_data, probe_stats

        return filtered_data

    def handle_missing_values(
        self,
        data: pd.DataFrame,
        strategy: Literal['median', 'mean', 'knn', 'drop'] = 'median',
        drop_threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Handle missing values in methylation data.

        Args:
            data: DataFrame with probes as rows and samples as columns.
            strategy: Imputation strategy:
                - 'median': Replace with probe median (default)
                - 'mean': Replace with probe mean
                - 'knn': K-nearest neighbors imputation (requires sklearn)
                - 'drop': Drop probes with any missing values
            drop_threshold: For 'median'/'mean' strategies, first drop probes
                with missing rate above this threshold. If None, uses instance default.

        Returns:
            DataFrame with missing values handled.

        Raises:
            ValueError: If strategy is invalid.

        Example:
            >>> preprocessor = MethylationPreprocessor()
            >>> imputed = preprocessor.handle_missing_values(data, strategy='median')
            >>> assert imputed.isna().sum().sum() == 0
        """
        logger.info(f"Handling missing values with strategy: {strategy}")

        total_missing = data.isna().sum().sum()
        total_values = data.size
        missing_rate = total_missing / total_values if total_values > 0 else 0

        logger.info(f"Total missing values: {total_missing} ({missing_rate*100:.2f}%)")

        if total_missing == 0:
            logger.info("No missing values found")
            return data.copy()

        # First, optionally drop probes with too many missing values
        drop_threshold = drop_threshold if drop_threshold is not None else self.missing_threshold

        if strategy != 'drop' and drop_threshold is not None:
            probe_missing_rate = data.isna().mean(axis=1)
            valid_probes = probe_missing_rate <= drop_threshold
            dropped_count = (~valid_probes).sum()

            if dropped_count > 0:
                logger.info(
                    f"Dropping {dropped_count} probes with missing rate > {drop_threshold}"
                )
                data = data.loc[valid_probes]

        # Apply imputation strategy
        if strategy == 'median':
            # Impute with probe median
            result = data.T.fillna(data.median(axis=1)).T

        elif strategy == 'mean':
            # Impute with probe mean
            result = data.T.fillna(data.mean(axis=1)).T

        elif strategy == 'drop':
            # Drop all probes with any missing values
            result = data.dropna(axis=0)
            logger.info(f"Dropped {data.shape[0] - result.shape[0]} probes with missing values")

        elif strategy == 'knn':
            if not SKLEARN_AVAILABLE:
                raise ImportError("sklearn required for KNN imputation")

            from sklearn.impute import KNNImputer

            imputer = KNNImputer(n_neighbors=5)
            imputed_values = imputer.fit_transform(data.T).T
            result = pd.DataFrame(
                imputed_values,
                index=data.index,
                columns=data.columns
            )

        else:
            raise ValueError(f"Invalid imputation strategy: {strategy}")

        remaining_missing = result.isna().sum().sum()
        logger.info(f"Remaining missing values: {remaining_missing}")

        return result

    def apply_batch_correction(
        self,
        data: pd.DataFrame,
        batch_info: pd.Series,
        method: Literal['combat', 'median_centering'] = 'median_centering'
    ) -> pd.DataFrame:
        """
        Apply batch effect correction to methylation data.

        Args:
            data: DataFrame with probes as rows and samples as columns.
            batch_info: Series mapping sample IDs to batch labels.
            method: Correction method:
                - 'combat': ComBat correction (requires pycombat)
                - 'median_centering': Simple median centering per batch

        Returns:
            Batch-corrected DataFrame.

        Raises:
            ValueError: If batch_info samples don't match data columns.

        Example:
            >>> batches = pd.Series({'sample1': 'batch1', 'sample2': 'batch2'})
            >>> corrected = preprocessor.apply_batch_correction(data, batches)
        """
        logger.info(f"Applying batch correction with method: {method}")

        # Ensure batch_info aligns with data columns
        common_samples = data.columns.intersection(batch_info.index)
        if len(common_samples) == 0:
            raise ValueError("No common samples between data and batch_info")

        if len(common_samples) < len(data.columns):
            logger.warning(
                f"Only {len(common_samples)} of {len(data.columns)} samples "
                "have batch information"
            )

        data = data[common_samples]
        batch_info = batch_info[common_samples]

        unique_batches = batch_info.unique()
        logger.info(f"Found {len(unique_batches)} batches: {list(unique_batches)}")

        if method == 'median_centering':
            # Simple median centering per batch
            result = data.copy()
            global_median = data.median(axis=1)

            for batch in unique_batches:
                batch_samples = batch_info[batch_info == batch].index
                batch_data = data[batch_samples]
                batch_median = batch_data.median(axis=1)

                # Center batch to global median
                correction = global_median - batch_median
                result[batch_samples] = batch_data.add(correction, axis=0)

            logger.info("Applied median centering batch correction")

        elif method == 'combat':
            try:
                from combat.pycombat import pycombat

                # pycombat expects data with samples as columns
                result = pycombat(data, batch_info)
                logger.info("Applied ComBat batch correction")

            except ImportError:
                logger.warning(
                    "pycombat not available, falling back to median centering"
                )
                return self.apply_batch_correction(
                    data, batch_info, method='median_centering'
                )

        else:
            raise ValueError(f"Invalid batch correction method: {method}")

        return result

    def standardize(
        self,
        data: pd.DataFrame,
        method: Literal['zscore', 'minmax'] = 'zscore',
        axis: int = 1
    ) -> pd.DataFrame:
        """
        Standardize methylation data.

        Args:
            data: DataFrame with probes as rows and samples as columns.
            method: Standardization method:
                - 'zscore': Z-score normalization (mean=0, std=1)
                - 'minmax': Min-max scaling to [0, 1]
            axis: Axis for standardization:
                - 0: Standardize per sample (column)
                - 1: Standardize per probe (row)

        Returns:
            Standardized DataFrame.

        Example:
            >>> standardized = preprocessor.standardize(data, method='zscore')
            >>> print(f"Mean: {standardized.mean().mean():.6f}")  # ~0
        """
        logger.info(f"Standardizing data with method: {method}, axis: {axis}")

        if method == 'zscore':
            if axis == 1:
                # Per-probe standardization
                mean = data.mean(axis=1)
                std = data.std(axis=1)
                std = std.replace(0, 1)  # Avoid division by zero
                result = data.sub(mean, axis=0).div(std, axis=0)
            else:
                # Per-sample standardization
                mean = data.mean(axis=0)
                std = data.std(axis=0)
                std = std.replace(0, 1)
                result = data.sub(mean, axis=1).div(std, axis=1)

        elif method == 'minmax':
            if axis == 1:
                min_val = data.min(axis=1)
                max_val = data.max(axis=1)
                range_val = max_val - min_val
                range_val = range_val.replace(0, 1)
                result = data.sub(min_val, axis=0).div(range_val, axis=0)
            else:
                min_val = data.min(axis=0)
                max_val = data.max(axis=0)
                range_val = max_val - min_val
                range_val = range_val.replace(0, 1)
                result = data.sub(min_val, axis=1).div(range_val, axis=1)

        else:
            raise ValueError(f"Invalid standardization method: {method}")

        return result

    def create_data_versions(
        self,
        data: pd.DataFrame,
        batch_info: Optional[pd.Series] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Create multiple preprocessed versions of the data.

        Creates original, standardized, and optionally batch-corrected
        versions of the methylation data for comparison.

        Args:
            data: DataFrame with probes as rows and samples as columns.
            batch_info: Optional series mapping samples to batches.

        Returns:
            Dictionary with keys 'original', 'standardized', and optionally
            'batch_corrected' mapping to respective DataFrames.

        Example:
            >>> versions = preprocessor.create_data_versions(data, batch_info)
            >>> for name, df in versions.items():
            ...     print(f"{name}: {df.shape}")
        """
        logger.info("Creating multiple data versions")

        versions: Dict[str, pd.DataFrame] = {
            'original': data.copy(),
            'standardized': self.standardize(data, method='zscore')
        }

        if batch_info is not None:
            versions['batch_corrected'] = self.apply_batch_correction(
                data, batch_info, method='median_centering'
            )
            versions['batch_corrected_standardized'] = self.standardize(
                versions['batch_corrected'], method='zscore'
            )

        logger.info(f"Created {len(versions)} data versions")

        return versions


def normalize_beta_values(
    data: pd.DataFrame,
    clip_range: Tuple[float, float] = (0.0, 1.0)
) -> pd.DataFrame:
    """
    Normalize methylation beta values to ensure valid range.

    Beta values should be between 0 and 1, representing the fraction
    of methylated cytosines at each CpG site.

    Args:
        data: DataFrame with methylation beta values.
        clip_range: Tuple of (min, max) for clipping values.

    Returns:
        DataFrame with values clipped to valid range.

    Example:
        >>> normalized = normalize_beta_values(data)
        >>> assert normalized.min().min() >= 0
        >>> assert normalized.max().max() <= 1
    """
    min_val, max_val = clip_range

    # Check for out-of-range values
    below_min = (data < min_val).sum().sum()
    above_max = (data > max_val).sum().sum()

    if below_min > 0 or above_max > 0:
        logger.warning(
            f"Found {below_min} values below {min_val} and "
            f"{above_max} values above {max_val}. Clipping to range."
        )

    return data.clip(lower=min_val, upper=max_val)


def calculate_missing_rate(
    data: pd.DataFrame,
    axis: Optional[int] = None
) -> Union[float, pd.Series]:
    """
    Calculate the missing value rate in methylation data.

    Args:
        data: DataFrame to analyze.
        axis: Axis for calculation:
            - None: Overall missing rate (returns float)
            - 0: Missing rate per column/sample (returns Series)
            - 1: Missing rate per row/probe (returns Series)

    Returns:
        Missing rate as float (overall) or Series (per axis).

    Example:
        >>> overall_rate = calculate_missing_rate(data)
        >>> probe_rates = calculate_missing_rate(data, axis=1)
        >>> sample_rates = calculate_missing_rate(data, axis=0)
    """
    if axis is None:
        return data.isna().sum().sum() / data.size

    return data.isna().mean(axis=axis)
