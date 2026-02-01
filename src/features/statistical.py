"""
Statistical feature selection methods for DNA methylation analysis.

This module provides statistical testing methods for feature selection,
including t-tests for binary comparisons, ANOVA for multiclass comparisons,
effect size calculations, and multiple testing correction.

The methods are optimized for high-dimensional methylation data with
support for parallel processing.

Example:
    >>> from src.features.statistical import (
    ...     StatisticalFeatureSelector,
    ...     run_ttest,
    ...     calculate_effect_size
    ... )
    >>>
    >>> # Quick function-based usage
    >>> pvalues = run_ttest(methylation_data, binary_labels)
    >>> effect_sizes = calculate_effect_size(methylation_data, binary_labels)
    >>>
    >>> # Class-based usage with full statistics
    >>> selector = StatisticalFeatureSelector()
    >>> results = selector.fit(methylation_data, labels)
"""

from typing import Tuple, Optional, Union, List
import numpy as np
import pandas as pd
from scipy import stats
from joblib import Parallel, delayed
import logging

# Configure module logger
logger = logging.getLogger(__name__)


def run_ttest(
    data: pd.DataFrame,
    labels: np.ndarray,
    equal_var: bool = False
) -> np.ndarray:
    """
    Run independent t-test for binary classification.

    Performs Welch's t-test (default) or Student's t-test on each feature
    to compare two groups. Designed for features as rows, samples as columns.

    Args:
        data: Feature matrix with features as rows and samples as columns.
              Shape: (n_features, n_samples)
        labels: Binary label array (0/1 or boolean) of length n_samples.
        equal_var: If True, use Student's t-test assuming equal variance.
                  If False (default), use Welch's t-test.

    Returns:
        Array of p-values for each feature, shape (n_features,).

    Raises:
        ValueError: If labels are not binary or dimensions mismatch.

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> # Sample data: 1000 features, 50 samples
        >>> data = pd.DataFrame(
        ...     np.random.randn(1000, 50),
        ...     index=[f"feature_{i}" for i in range(1000)]
        ... )
        >>> labels = np.array([0]*25 + [1]*25)
        >>>
        >>> pvalues = run_ttest(data, labels)
        >>> print(f"Significant features (p<0.05): {(pvalues < 0.05).sum()}")
    """
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)

    if len(unique_labels) != 2:
        raise ValueError(
            f"Binary labels required, got {len(unique_labels)} unique values"
        )

    if len(labels) != data.shape[1]:
        raise ValueError(
            f"Label length ({len(labels)}) does not match "
            f"number of samples ({data.shape[1]})"
        )

    # Split data by group
    group0_mask = labels == unique_labels[0]
    group1_mask = labels == unique_labels[1]

    group0_data = data.iloc[:, group0_mask].values
    group1_data = data.iloc[:, group1_mask].values

    # Vectorized t-test
    _, pvalues = stats.ttest_ind(
        group0_data, group1_data,
        axis=1, equal_var=equal_var
    )

    # Handle NaN p-values (constant features)
    pvalues = np.nan_to_num(pvalues, nan=1.0)

    logger.debug("Completed t-test for %d features", len(pvalues))

    return pvalues


def run_anova(
    data: pd.DataFrame,
    labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run one-way ANOVA for multiclass classification.

    Performs one-way ANOVA on each feature to compare multiple groups.
    Returns both F-statistics and p-values.

    Args:
        data: Feature matrix with features as rows and samples as columns.
              Shape: (n_features, n_samples)
        labels: Multiclass label array (integers) of length n_samples.

    Returns:
        Tuple of (f_statistics, p_values), each of shape (n_features,).

    Raises:
        ValueError: If less than 2 classes or dimensions mismatch.

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> # Sample data: 1000 features, 60 samples (3 groups)
        >>> data = pd.DataFrame(
        ...     np.random.randn(1000, 60),
        ...     index=[f"feature_{i}" for i in range(1000)]
        ... )
        >>> labels = np.array([0]*20 + [1]*20 + [2]*20)
        >>>
        >>> f_stats, pvalues = run_anova(data, labels)
        >>> print(f"Mean F-statistic: {f_stats.mean():.3f}")
    """
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        raise ValueError(
            f"At least 2 classes required, got {len(unique_labels)}"
        )

    if len(labels) != data.shape[1]:
        raise ValueError(
            f"Label length ({len(labels)}) does not match "
            f"number of samples ({data.shape[1]})"
        )

    # Group data by labels
    groups = [data.iloc[:, labels == label].values for label in unique_labels]

    # Vectorized ANOVA using scipy
    f_stats, pvalues = stats.f_oneway(*groups, axis=1)

    # Handle NaN values (constant features or single sample groups)
    f_stats = np.nan_to_num(f_stats, nan=0.0)
    pvalues = np.nan_to_num(pvalues, nan=1.0)

    logger.debug("Completed ANOVA for %d features across %d classes",
                len(pvalues), len(unique_labels))

    return f_stats, pvalues


def calculate_effect_size(
    data: pd.DataFrame,
    labels: np.ndarray,
    method: str = "cohens_d"
) -> np.ndarray:
    """
    Calculate effect size for binary comparison.

    Computes Cohen's d (default) or other effect size measures for
    each feature comparing two groups.

    Args:
        data: Feature matrix with features as rows and samples as columns.
        labels: Binary label array (0/1 or boolean) of length n_samples.
        method: Effect size method:
               - "cohens_d": Cohen's d (standardized mean difference)
               - "hedges_g": Hedges' g (bias-corrected Cohen's d)
               - "glass_delta": Glass's delta (using control group SD)

    Returns:
        Array of effect sizes for each feature, shape (n_features,).
        Positive values indicate higher values in group 1.

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> data = pd.DataFrame(np.random.randn(1000, 50))
        >>> labels = np.array([0]*25 + [1]*25)
        >>>
        >>> effect_sizes = calculate_effect_size(data, labels)
        >>> large_effects = np.abs(effect_sizes) > 0.8
        >>> print(f"Features with large effect: {large_effects.sum()}")
    """
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)

    if len(unique_labels) != 2:
        raise ValueError(
            f"Binary labels required, got {len(unique_labels)} unique values"
        )

    # Split data by group
    group0_mask = labels == unique_labels[0]
    group1_mask = labels == unique_labels[1]

    group0_data = data.iloc[:, group0_mask].values
    group1_data = data.iloc[:, group1_mask].values

    # Calculate means
    mean0 = np.mean(group0_data, axis=1)
    mean1 = np.mean(group1_data, axis=1)

    # Calculate standard deviations
    std0 = np.std(group0_data, axis=1, ddof=1)
    std1 = np.std(group1_data, axis=1, ddof=1)

    n0 = group0_data.shape[1]
    n1 = group1_data.shape[1]

    if method == "cohens_d":
        # Pooled standard deviation
        pooled_std = np.sqrt(
            ((n0 - 1) * std0**2 + (n1 - 1) * std1**2) / (n0 + n1 - 2)
        )
        # Avoid division by zero
        pooled_std = np.where(pooled_std == 0, 1e-10, pooled_std)
        effect_sizes = (mean1 - mean0) / pooled_std

    elif method == "hedges_g":
        # Cohen's d with bias correction
        pooled_std = np.sqrt(
            ((n0 - 1) * std0**2 + (n1 - 1) * std1**2) / (n0 + n1 - 2)
        )
        pooled_std = np.where(pooled_std == 0, 1e-10, pooled_std)
        d = (mean1 - mean0) / pooled_std

        # Hedges' correction factor
        correction = 1 - (3 / (4 * (n0 + n1) - 9))
        effect_sizes = d * correction

    elif method == "glass_delta":
        # Use group 0 (control) standard deviation
        std0 = np.where(std0 == 0, 1e-10, std0)
        effect_sizes = (mean1 - mean0) / std0

    else:
        raise ValueError(f"Unknown method: {method}")

    # Handle NaN values
    effect_sizes = np.nan_to_num(effect_sizes, nan=0.0)

    logger.debug("Calculated %s for %d features", method, len(effect_sizes))

    return effect_sizes


def calculate_eta_squared(
    data: pd.DataFrame,
    labels: np.ndarray
) -> np.ndarray:
    """
    Calculate eta-squared effect size for multiclass comparison.

    Eta-squared represents the proportion of variance in the dependent
    variable explained by group membership.

    Args:
        data: Feature matrix with features as rows and samples as columns.
        labels: Multiclass label array (integers) of length n_samples.

    Returns:
        Array of eta-squared values for each feature, shape (n_features,).
        Values range from 0 to 1.

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> data = pd.DataFrame(np.random.randn(1000, 60))
        >>> labels = np.array([0]*20 + [1]*20 + [2]*20)
        >>>
        >>> eta_sq = calculate_eta_squared(data, labels)
        >>> print(f"Mean eta-squared: {eta_sq.mean():.4f}")
    """
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)

    n_features = data.shape[0]
    n_samples = data.shape[1]

    # Calculate grand mean
    grand_mean = data.values.mean(axis=1)

    # Calculate SS_total (total sum of squares)
    ss_total = np.sum((data.values - grand_mean[:, np.newaxis])**2, axis=1)

    # Calculate SS_between (between-group sum of squares)
    ss_between = np.zeros(n_features)
    for label in unique_labels:
        mask = labels == label
        n_group = mask.sum()
        group_mean = data.iloc[:, mask].values.mean(axis=1)
        ss_between += n_group * (group_mean - grand_mean)**2

    # Calculate eta-squared
    eta_squared = np.where(ss_total > 0, ss_between / ss_total, 0)

    logger.debug("Calculated eta-squared for %d features", n_features)

    return eta_squared


def adjust_pvalues(
    pvalues: np.ndarray,
    method: str = 'fdr_bh'
) -> np.ndarray:
    """
    Adjust p-values for multiple testing correction.

    Applies various multiple testing correction methods to control
    for false discoveries when testing many features simultaneously.

    Args:
        pvalues: Array of raw p-values.
        method: Correction method:
               - 'fdr_bh': Benjamini-Hochberg FDR (default)
               - 'fdr_by': Benjamini-Yekutieli FDR
               - 'bonferroni': Bonferroni correction
               - 'holm': Holm-Bonferroni step-down
               - 'hommel': Hommel's procedure

    Returns:
        Array of adjusted p-values (same shape as input).

    Example:
        >>> import numpy as np
        >>>
        >>> pvalues = np.array([0.001, 0.01, 0.02, 0.03, 0.5])
        >>> adjusted = adjust_pvalues(pvalues, method='fdr_bh')
        >>> print(f"Adjusted p-values: {adjusted}")
    """
    from statsmodels.stats.multitest import multipletests

    pvalues = np.asarray(pvalues)

    # Handle edge cases
    if len(pvalues) == 0:
        return pvalues

    # Replace NaN with 1.0
    pvalues_clean = np.nan_to_num(pvalues, nan=1.0)

    # Clip to valid range
    pvalues_clean = np.clip(pvalues_clean, 0, 1)

    # Apply correction
    _, adjusted, _, _ = multipletests(pvalues_clean, method=method)

    logger.debug("Applied %s correction to %d p-values", method, len(pvalues))

    return adjusted


class StatisticalFeatureSelector:
    """
    Statistical feature selector for DNA methylation data.

    Provides a unified interface for statistical feature selection,
    combining multiple testing methods with effect size filtering
    and multiple testing correction.

    Attributes:
        n_jobs: Number of parallel jobs for computation.
        results_: Dictionary storing selection results after fitting.

    Example:
        >>> from src.features.statistical import StatisticalFeatureSelector
        >>>
        >>> selector = StatisticalFeatureSelector(n_jobs=4)
        >>>
        >>> # Fit and get selected features
        >>> selector.fit(methylation_data, labels)
        >>> features = selector.get_significant_features(
        ...     pvalue_threshold=0.05,
        ...     effect_size_threshold=0.3
        ... )
        >>>
        >>> # Access detailed statistics
        >>> print(selector.results_['statistics'].head())
    """

    def __init__(self, n_jobs: int = -1):
        """
        Initialize StatisticalFeatureSelector.

        Args:
            n_jobs: Number of parallel jobs. -1 uses all available cores.

        Example:
            >>> selector = StatisticalFeatureSelector(n_jobs=4)
        """
        self.n_jobs = n_jobs
        self.results_: dict = {}
        self._is_fitted = False

    def fit(
        self,
        data: pd.DataFrame,
        labels: np.ndarray,
        test_type: str = "auto"
    ) -> 'StatisticalFeatureSelector':
        """
        Fit the selector by computing statistical tests.

        Args:
            data: Feature matrix with features as rows and samples as columns.
            labels: Label array of length n_samples.
            test_type: Type of test to perform:
                      - "auto": Automatically choose based on label count
                      - "ttest": Force t-test (binary)
                      - "anova": Force ANOVA (multiclass)

        Returns:
            Self for method chaining.

        Example:
            >>> selector = StatisticalFeatureSelector()
            >>> selector.fit(data, labels)
            >>> print(selector.results_['statistics'].head())
        """
        labels = np.asarray(labels)
        n_classes = len(np.unique(labels))

        # Determine test type
        if test_type == "auto":
            test_type = "ttest" if n_classes == 2 else "anova"

        logger.info("Fitting with %s test (%d classes)", test_type, n_classes)

        # Run appropriate test
        if test_type == "ttest":
            pvalues = run_ttest(data, labels)
            effect_sizes = calculate_effect_size(data, labels)

            stats_df = pd.DataFrame({
                'pvalue': pvalues,
                'effect_size': effect_sizes
            }, index=data.index)

        elif test_type == "anova":
            f_stats, pvalues = run_anova(data, labels)
            eta_sq = calculate_eta_squared(data, labels)

            stats_df = pd.DataFrame({
                'f_statistic': f_stats,
                'pvalue': pvalues,
                'eta_squared': eta_sq
            }, index=data.index)

        else:
            raise ValueError(f"Unknown test_type: {test_type}")

        # Apply FDR correction
        stats_df['pvalue_adjusted'] = adjust_pvalues(pvalues, method='fdr_bh')

        # Store results
        self.results_ = {
            'test_type': test_type,
            'n_classes': n_classes,
            'n_features': len(data),
            'statistics': stats_df
        }

        self._is_fitted = True

        logger.info("Fitting complete. Statistics computed for %d features.",
                   len(data))

        return self

    def get_significant_features(
        self,
        pvalue_threshold: float = 0.05,
        effect_size_threshold: float = 0.0,
        use_adjusted_pvalue: bool = True
    ) -> List[str]:
        """
        Get features meeting significance thresholds.

        Args:
            pvalue_threshold: Maximum p-value for significance.
            effect_size_threshold: Minimum absolute effect size.
            use_adjusted_pvalue: If True, use FDR-adjusted p-values.

        Returns:
            List of significant feature names.

        Raises:
            ValueError: If selector has not been fitted.

        Example:
            >>> selector = StatisticalFeatureSelector()
            >>> selector.fit(data, labels)
            >>> features = selector.get_significant_features(
            ...     pvalue_threshold=0.01,
            ...     effect_size_threshold=0.5
            ... )
        """
        if not self._is_fitted:
            raise ValueError("Selector not fitted. Call fit() first.")

        stats = self.results_['statistics']

        # Select p-value column
        pval_col = 'pvalue_adjusted' if use_adjusted_pvalue else 'pvalue'

        # Build selection mask
        mask = stats[pval_col] < pvalue_threshold

        # Add effect size filter if available and threshold > 0
        if effect_size_threshold > 0:
            if 'effect_size' in stats.columns:
                mask &= np.abs(stats['effect_size']) >= effect_size_threshold
            elif 'eta_squared' in stats.columns:
                mask &= stats['eta_squared'] >= effect_size_threshold

        selected = stats.index[mask].tolist()

        logger.info("Selected %d features (p<%.3f, effect>%.2f)",
                   len(selected), pvalue_threshold, effect_size_threshold)

        return selected

    def get_top_features(
        self,
        n_features: int = 100,
        rank_by: str = "pvalue"
    ) -> List[str]:
        """
        Get top N features ranked by statistical criterion.

        Args:
            n_features: Number of top features to return.
            rank_by: Ranking criterion:
                    - "pvalue": Rank by ascending p-value
                    - "effect_size": Rank by descending absolute effect size
                    - "combined": Rank by combined score

        Returns:
            List of top feature names.

        Example:
            >>> selector = StatisticalFeatureSelector()
            >>> selector.fit(data, labels)
            >>> top_features = selector.get_top_features(n_features=50)
        """
        if not self._is_fitted:
            raise ValueError("Selector not fitted. Call fit() first.")

        stats = self.results_['statistics'].copy()

        if rank_by == "pvalue":
            # Sort by ascending p-value
            sorted_stats = stats.sort_values('pvalue')

        elif rank_by == "effect_size":
            # Sort by descending absolute effect size
            if 'effect_size' in stats.columns:
                stats['_abs_effect'] = np.abs(stats['effect_size'])
                sorted_stats = stats.sort_values('_abs_effect', ascending=False)
            elif 'eta_squared' in stats.columns:
                sorted_stats = stats.sort_values('eta_squared', ascending=False)
            else:
                raise ValueError("No effect size column available")

        elif rank_by == "combined":
            # Combined score: -log10(p) * |effect_size|
            if 'effect_size' in stats.columns:
                effect_col = np.abs(stats['effect_size'])
            elif 'eta_squared' in stats.columns:
                effect_col = stats['eta_squared']
            else:
                effect_col = 1.0

            stats['_combined'] = -np.log10(stats['pvalue'] + 1e-300) * effect_col
            sorted_stats = stats.sort_values('_combined', ascending=False)

        else:
            raise ValueError(f"Unknown rank_by: {rank_by}")

        top_features = sorted_stats.head(n_features).index.tolist()

        logger.info("Selected top %d features ranked by %s",
                   len(top_features), rank_by)

        return top_features

    def get_statistics(self) -> pd.DataFrame:
        """
        Get the full statistics DataFrame.

        Returns:
            DataFrame with test statistics for all features.

        Example:
            >>> selector = StatisticalFeatureSelector()
            >>> selector.fit(data, labels)
            >>> stats_df = selector.get_statistics()
            >>> print(stats_df.describe())
        """
        if not self._is_fitted:
            raise ValueError("Selector not fitted. Call fit() first.")

        return self.results_['statistics'].copy()
