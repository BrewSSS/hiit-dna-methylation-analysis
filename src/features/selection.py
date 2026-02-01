"""
Ten-Level Feature Selection Framework.

This module implements the core Ten-Level Feature Selection Framework, a novel
methodology for graduated stringency feature selection in DNA methylation studies.
The framework enables systematic biomarker identification across different
analysis contexts with reproducible and configurable thresholds.

The ten levels range from extremely strict (L1) to most relaxed (L10), allowing
researchers to balance between specificity and sensitivity based on their
study requirements.

Example:
    >>> from src.features.selection import TenLevelFeatureSelector, FeatureSelectionConfig
    >>>
    >>> # Initialize with default configuration
    >>> config = FeatureSelectionConfig()
    >>> selector = TenLevelFeatureSelector(config)
    >>>
    >>> # Select features for binary classification
    >>> binary_features = selector.select_binary_features(
    ...     data=methylation_data,
    ...     labels=binary_labels,
    ...     level="L5_moderate"
    ... )
    >>>
    >>> # Run full multi-strategy selection
    >>> results = selector.run_full_selection(
    ...     data=methylation_data,
    ...     binary_labels=binary_labels,
    ...     multiclass_labels=duration_labels,
    ...     timepoints=timepoint_array
    ... )
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from collections import Counter
import logging

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class FeatureSelectionConfig:
    """
    Configuration for ten-level feature selection thresholds.

    This class defines the thresholds for each of the ten stringency levels
    used in the feature selection framework. Each level specifies:
    - p-value threshold for statistical significance
    - effect size threshold (Cohen's d for binary, eta-squared for multiclass)
    - variance threshold for filtering low-variance features
    - F-statistic threshold for ANOVA-based selection
    - Whether to apply FDR correction

    The levels progress from extremely strict (L1) to most relaxed (L10),
    providing flexibility for different research contexts.

    Attributes:
        LEVELS: Dictionary mapping level names to threshold parameters.
        variance_threshold: Global minimum variance threshold.
        n_jobs: Number of parallel jobs for computation.
        random_state: Random seed for reproducibility.

    Example:
        >>> config = FeatureSelectionConfig()
        >>> print(config.LEVELS["L5_moderate"])
        {'pvalue': 0.05, 'effect_size': 0.3, 'variance': 0.025,
         'f_stat': 4.0, 'fdr_correction': False}

        >>> # Custom configuration
        >>> custom_config = FeatureSelectionConfig(
        ...     variance_threshold=0.01,
        ...     n_jobs=4
        ... )
    """

    variance_threshold: float = 0.01
    n_jobs: int = -1
    random_state: int = 42
    LEVELS: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize the ten-level thresholds after dataclass initialization."""
        if not self.LEVELS:
            self.LEVELS = {
                "L1_extremely_strict": {
                    "pvalue": 0.001,
                    "effect_size": 0.8,
                    "variance": 0.05,
                    "f_stat": 10.0,
                    "fdr_correction": True,
                    "description": "Extremely strict - highest confidence biomarkers"
                },
                "L2_strict": {
                    "pvalue": 0.005,
                    "effect_size": 0.6,
                    "variance": 0.04,
                    "f_stat": 8.0,
                    "fdr_correction": True,
                    "description": "Strict - high confidence biomarkers"
                },
                "L3_moderately_strict": {
                    "pvalue": 0.01,
                    "effect_size": 0.5,
                    "variance": 0.035,
                    "f_stat": 6.0,
                    "fdr_correction": True,
                    "description": "Moderately strict - confident biomarkers"
                },
                "L4_standard": {
                    "pvalue": 0.01,
                    "effect_size": 0.4,
                    "variance": 0.03,
                    "f_stat": 5.0,
                    "fdr_correction": True,
                    "description": "Standard - conventional significance threshold"
                },
                "L5_moderate": {
                    "pvalue": 0.05,
                    "effect_size": 0.3,
                    "variance": 0.025,
                    "f_stat": 4.0,
                    "fdr_correction": False,
                    "description": "Moderate - balanced sensitivity/specificity"
                },
                "L6_moderately_relaxed": {
                    "pvalue": 0.05,
                    "effect_size": 0.25,
                    "variance": 0.02,
                    "f_stat": 3.0,
                    "fdr_correction": False,
                    "description": "Moderately relaxed - increased sensitivity"
                },
                "L7_relaxed": {
                    "pvalue": 0.1,
                    "effect_size": 0.2,
                    "variance": 0.02,
                    "f_stat": 2.5,
                    "fdr_correction": False,
                    "description": "Relaxed - exploratory analysis"
                },
                "L8_more_relaxed": {
                    "pvalue": 0.1,
                    "effect_size": 0.15,
                    "variance": 0.02,
                    "f_stat": 2.0,
                    "fdr_correction": False,
                    "description": "More relaxed - discovery-oriented"
                },
                "L9_even_more_relaxed": {
                    "pvalue": 0.2,
                    "effect_size": 0.1,
                    "variance": 0.02,
                    "f_stat": 1.5,
                    "fdr_correction": False,
                    "description": "Even more relaxed - hypothesis generation"
                },
                "L10_most_relaxed": {
                    "pvalue": 0.3,
                    "effect_size": 0.05,
                    "variance": 0.02,
                    "f_stat": 1.0,
                    "fdr_correction": False,
                    "description": "Most relaxed - comprehensive screening"
                },
            }

    def get_level_thresholds(self, level: str) -> Dict[str, Any]:
        """
        Get thresholds for a specific stringency level.

        Args:
            level: The stringency level name (e.g., "L5_moderate").

        Returns:
            Dictionary containing threshold parameters for the level.

        Raises:
            ValueError: If the specified level is not valid.

        Example:
            >>> config = FeatureSelectionConfig()
            >>> thresholds = config.get_level_thresholds("L3_moderately_strict")
            >>> print(thresholds["pvalue"])
            0.01
        """
        if level not in self.LEVELS:
            valid_levels = list(self.LEVELS.keys())
            raise ValueError(
                f"Invalid level '{level}'. Valid levels are: {valid_levels}"
            )
        return self.LEVELS[level]

    def list_levels(self) -> List[str]:
        """
        List all available stringency levels.

        Returns:
            List of level names ordered from strictest to most relaxed.

        Example:
            >>> config = FeatureSelectionConfig()
            >>> levels = config.list_levels()
            >>> print(levels[0])
            'L1_extremely_strict'
        """
        return list(self.LEVELS.keys())

    def get_level_description(self, level: str) -> str:
        """
        Get a human-readable description of a stringency level.

        Args:
            level: The stringency level name.

        Returns:
            Description string for the level.

        Example:
            >>> config = FeatureSelectionConfig()
            >>> desc = config.get_level_description("L5_moderate")
            >>> print(desc)
            'Moderate - balanced sensitivity/specificity'
        """
        thresholds = self.get_level_thresholds(level)
        return thresholds.get("description", "No description available")


class TenLevelFeatureSelector:
    """
    Ten-Level Feature Selection Framework.

    A novel methodology implementing graduated stringency levels for
    robust biomarker identification in DNA methylation studies. This
    framework provides systematic feature selection across three
    complementary analysis strategies:

    1. Binary classification (e.g., HIIT intervention vs Control)
    2. Multiclass classification (e.g., 4W/8W/12W training duration)
    3. Time-series trajectory analysis (temporal methylation patterns)

    The framework combines statistical testing, effect size filtering,
    and optional machine learning-based selection to identify the most
    robust and biologically relevant features.

    Attributes:
        config: FeatureSelectionConfig instance with threshold settings.
        statistical_selector: StatisticalFeatureSelector for p-value computation.
        results_: Dictionary storing selection results after fitting.

    Example:
        >>> from src.features.selection import TenLevelFeatureSelector
        >>>
        >>> # Basic usage
        >>> selector = TenLevelFeatureSelector()
        >>> features = selector.select_binary_features(
        ...     data=methylation_df,
        ...     labels=treatment_labels,
        ...     level="L5_moderate"
        ... )
        >>> print(f"Selected {len(features)} features")

        >>> # Full multi-strategy selection
        >>> results = selector.run_full_selection(
        ...     data=methylation_df,
        ...     binary_labels=treatment_labels,
        ...     multiclass_labels=duration_labels,
        ...     timepoints=timepoint_array
        ... )
        >>> print(results["summary"])
    """

    def __init__(self, config: Optional[FeatureSelectionConfig] = None):
        """
        Initialize the Ten-Level Feature Selector.

        Args:
            config: Optional FeatureSelectionConfig instance. If None,
                   uses default configuration.

        Example:
            >>> # With default config
            >>> selector = TenLevelFeatureSelector()

            >>> # With custom config
            >>> custom_config = FeatureSelectionConfig(n_jobs=4)
            >>> selector = TenLevelFeatureSelector(config=custom_config)
        """
        self.config = config if config is not None else FeatureSelectionConfig()
        self._statistical_selector = None
        self._ml_selectors = {}
        self._timeseries_analyzer = None
        self.results_: Dict[str, Any] = {}

        logger.info("TenLevelFeatureSelector initialized with %d levels",
                   len(self.config.LEVELS))

    @property
    def statistical_selector(self):
        """Lazy initialization of statistical selector."""
        if self._statistical_selector is None:
            from .statistical import StatisticalFeatureSelector
            self._statistical_selector = StatisticalFeatureSelector(
                n_jobs=self.config.n_jobs
            )
        return self._statistical_selector

    def _validate_data(
        self,
        data: pd.DataFrame,
        labels: Optional[np.ndarray] = None
    ) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        """
        Validate input data and labels.

        Args:
            data: Feature matrix (features x samples or samples x features).
            labels: Optional label array for supervised selection.

        Returns:
            Tuple of validated (data, labels).

        Raises:
            ValueError: If data format is invalid or dimensions mismatch.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")

        if data.empty:
            raise ValueError("Data DataFrame is empty")

        if labels is not None:
            labels = np.asarray(labels)
            if len(labels) != data.shape[1]:
                # Try transposed orientation
                if len(labels) == data.shape[0]:
                    logger.info("Transposing data to match label dimensions")
                    data = data.T
                else:
                    raise ValueError(
                        f"Label length ({len(labels)}) does not match "
                        f"data dimensions ({data.shape})"
                    )

        return data, labels

    def _apply_variance_filter(
        self,
        data: pd.DataFrame,
        threshold: float
    ) -> pd.DataFrame:
        """
        Filter features by variance threshold.

        Args:
            data: Feature matrix (features x samples).
            threshold: Minimum variance threshold.

        Returns:
            Filtered DataFrame with only high-variance features.
        """
        variances = data.var(axis=1)
        mask = variances >= threshold
        filtered_data = data.loc[mask]

        n_removed = (~mask).sum()
        if n_removed > 0:
            logger.info("Removed %d low-variance features (threshold=%.4f)",
                       n_removed, threshold)

        return filtered_data

    def select_binary_features(
        self,
        data: pd.DataFrame,
        labels: np.ndarray,
        level: str = "L5_moderate",
        return_statistics: bool = False
    ) -> Union[List[str], Tuple[List[str], pd.DataFrame]]:
        """
        Select features for binary classification at specified stringency.

        Uses independent t-test for statistical significance and Cohen's d
        for effect size filtering. Features must pass both thresholds to
        be selected.

        Args:
            data: Feature matrix with features as rows and samples as columns.
                  Index should contain feature identifiers.
            labels: Binary label array (0/1 or boolean) matching sample count.
            level: Stringency level name (default: "L5_moderate").
            return_statistics: If True, also return statistics DataFrame.

        Returns:
            List of selected feature names. If return_statistics=True,
            returns tuple of (feature_list, statistics_dataframe).

        Raises:
            ValueError: If data/labels are invalid or level is unknown.

        Example:
            >>> selector = TenLevelFeatureSelector()
            >>>
            >>> # Basic selection
            >>> features = selector.select_binary_features(
            ...     data=methylation_df,
            ...     labels=treatment_labels,
            ...     level="L4_standard"
            ... )
            >>>
            >>> # With statistics
            >>> features, stats = selector.select_binary_features(
            ...     data=methylation_df,
            ...     labels=treatment_labels,
            ...     level="L4_standard",
            ...     return_statistics=True
            ... )
            >>> print(stats.head())
        """
        # Validate inputs
        data, labels = self._validate_data(data, labels)
        thresholds = self.config.get_level_thresholds(level)

        logger.info("Binary feature selection at level '%s' (p<%.3f, |d|>%.2f)",
                   level, thresholds["pvalue"], thresholds["effect_size"])

        # Apply variance filter
        data = self._apply_variance_filter(data, thresholds["variance"])

        # Compute statistics
        from .statistical import run_ttest, calculate_effect_size, adjust_pvalues

        pvalues = run_ttest(data, labels)
        effect_sizes = calculate_effect_size(data, labels)

        # Apply FDR correction if specified
        if thresholds["fdr_correction"]:
            pvalues_adj = adjust_pvalues(pvalues, method='fdr_bh')
            pvalues_to_use = pvalues_adj
            logger.info("Applied FDR correction (Benjamini-Hochberg)")
        else:
            pvalues_to_use = pvalues

        # Apply thresholds
        mask = (
            (pvalues_to_use < thresholds["pvalue"]) &
            (np.abs(effect_sizes) >= thresholds["effect_size"])
        )

        selected_features = data.index[mask].tolist()

        logger.info("Selected %d features from %d candidates",
                   len(selected_features), len(data))

        if return_statistics:
            stats_df = pd.DataFrame({
                'pvalue': pvalues,
                'pvalue_adjusted': pvalues_adj if thresholds["fdr_correction"] else pvalues,
                'effect_size': effect_sizes,
                'selected': mask
            }, index=data.index)
            return selected_features, stats_df

        return selected_features

    def select_multiclass_features(
        self,
        data: pd.DataFrame,
        labels: np.ndarray,
        level: str = "L5_moderate",
        return_statistics: bool = False
    ) -> Union[List[str], Tuple[List[str], pd.DataFrame]]:
        """
        Select features for multiclass classification.

        Uses one-way ANOVA for statistical significance and F-statistic
        threshold for effect size filtering. Suitable for comparing
        multiple treatment groups or time points.

        Args:
            data: Feature matrix with features as rows and samples as columns.
            labels: Multiclass label array (integers) matching sample count.
            level: Stringency level name (default: "L5_moderate").
            return_statistics: If True, also return statistics DataFrame.

        Returns:
            List of selected feature names. If return_statistics=True,
            returns tuple of (feature_list, statistics_dataframe).

        Example:
            >>> selector = TenLevelFeatureSelector()
            >>>
            >>> # Select features distinguishing training durations
            >>> features = selector.select_multiclass_features(
            ...     data=methylation_df,
            ...     labels=duration_labels,  # e.g., [0, 1, 2] for 4W, 8W, 12W
            ...     level="L5_moderate"
            ... )
        """
        # Validate inputs
        data, labels = self._validate_data(data, labels)
        thresholds = self.config.get_level_thresholds(level)

        n_classes = len(np.unique(labels))
        logger.info("Multiclass feature selection at level '%s' "
                   "(p<%.3f, F>%.1f, %d classes)",
                   level, thresholds["pvalue"], thresholds["f_stat"], n_classes)

        # Apply variance filter
        data = self._apply_variance_filter(data, thresholds["variance"])

        # Compute ANOVA statistics
        from .statistical import run_anova, adjust_pvalues

        f_stats, pvalues = run_anova(data, labels)

        # Apply FDR correction if specified
        if thresholds["fdr_correction"]:
            pvalues_adj = adjust_pvalues(pvalues, method='fdr_bh')
            pvalues_to_use = pvalues_adj
            logger.info("Applied FDR correction (Benjamini-Hochberg)")
        else:
            pvalues_to_use = pvalues

        # Apply thresholds
        mask = (
            (pvalues_to_use < thresholds["pvalue"]) &
            (f_stats >= thresholds["f_stat"])
        )

        selected_features = data.index[mask].tolist()

        logger.info("Selected %d features from %d candidates",
                   len(selected_features), len(data))

        if return_statistics:
            stats_df = pd.DataFrame({
                'f_statistic': f_stats,
                'pvalue': pvalues,
                'pvalue_adjusted': pvalues_adj if thresholds["fdr_correction"] else pvalues,
                'selected': mask
            }, index=data.index)
            return selected_features, stats_df

        return selected_features

    def select_timeseries_features(
        self,
        data: pd.DataFrame,
        timepoints: np.ndarray,
        level: str = "L5_moderate",
        trend_type: str = "any",
        return_statistics: bool = False
    ) -> Union[List[str], Tuple[List[str], pd.DataFrame]]:
        """
        Select features showing significant temporal trends.

        Identifies features with monotonic trends (increasing or decreasing)
        or significant temporal variation across time points. Useful for
        detecting biomarkers of progressive adaptation.

        Args:
            data: Feature matrix with features as rows and samples as columns.
            timepoints: Numeric timepoint array for each sample.
            level: Stringency level name (default: "L5_moderate").
            trend_type: Type of trend to detect:
                       - "increasing": Only increasing trends
                       - "decreasing": Only decreasing trends
                       - "monotonic": Either direction
                       - "any": Any significant temporal pattern
            return_statistics: If True, also return statistics DataFrame.

        Returns:
            List of selected feature names with significant temporal patterns.

        Example:
            >>> selector = TenLevelFeatureSelector()
            >>>
            >>> # Find features with increasing methylation over time
            >>> features = selector.select_timeseries_features(
            ...     data=methylation_df,
            ...     timepoints=week_array,  # e.g., [0, 4, 8, 12]
            ...     level="L5_moderate",
            ...     trend_type="increasing"
            ... )
        """
        # Validate inputs
        data, _ = self._validate_data(data)
        timepoints = np.asarray(timepoints)
        thresholds = self.config.get_level_thresholds(level)

        if len(timepoints) != data.shape[1]:
            raise ValueError(
                f"Timepoints length ({len(timepoints)}) does not match "
                f"number of samples ({data.shape[1]})"
            )

        logger.info("Time-series feature selection at level '%s' (trend=%s)",
                   level, trend_type)

        # Apply variance filter
        data = self._apply_variance_filter(data, thresholds["variance"])

        # Use time-series analyzer
        from .time_series import TimeSeriesFeatureAnalyzer

        analyzer = TimeSeriesFeatureAnalyzer(n_jobs=self.config.n_jobs)

        # Detect monotonic trends
        trend_results = analyzer.detect_monotonic_trend(data, timepoints)

        # Apply significance threshold
        pvalue_mask = trend_results['pvalue'] < thresholds["pvalue"]

        # Apply trend type filter
        if trend_type == "increasing":
            direction_mask = trend_results['correlation'] > 0
        elif trend_type == "decreasing":
            direction_mask = trend_results['correlation'] < 0
        elif trend_type == "monotonic":
            direction_mask = np.abs(trend_results['correlation']) > 0.3
        else:  # "any"
            direction_mask = np.ones(len(trend_results), dtype=bool)

        mask = pvalue_mask & direction_mask
        selected_features = data.index[mask].tolist()

        logger.info("Selected %d features with temporal trends from %d candidates",
                   len(selected_features), len(data))

        if return_statistics:
            stats_df = trend_results.copy()
            stats_df['selected'] = mask
            return selected_features, stats_df

        return selected_features

    def get_consensus_features(
        self,
        binary_features: List[str],
        multiclass_features: List[str],
        timeseries_features: Optional[List[str]] = None,
        min_overlap: int = 2
    ) -> List[str]:
        """
        Find features selected by multiple strategies.

        Identifies robust features that are consistently selected across
        different analysis strategies, improving confidence in biomarker
        identification.

        Args:
            binary_features: Features from binary classification selection.
            multiclass_features: Features from multiclass classification selection.
            timeseries_features: Optional features from time-series analysis.
            min_overlap: Minimum number of strategies that must select a feature
                        for it to be included in consensus (default: 2).

        Returns:
            List of consensus feature names.

        Example:
            >>> selector = TenLevelFeatureSelector()
            >>>
            >>> # Get features from different strategies
            >>> binary = selector.select_binary_features(data, binary_labels)
            >>> multi = selector.select_multiclass_features(data, multi_labels)
            >>> temporal = selector.select_timeseries_features(data, timepoints)
            >>>
            >>> # Find consensus features
            >>> consensus = selector.get_consensus_features(
            ...     binary, multi, temporal,
            ...     min_overlap=2
            ... )
        """
        # Combine all feature lists
        all_features = list(binary_features) + list(multiclass_features)
        if timeseries_features is not None:
            all_features += list(timeseries_features)

        # Count occurrences
        feature_counts = Counter(all_features)

        # Select features meeting overlap threshold
        consensus = [
            feature for feature, count in feature_counts.items()
            if count >= min_overlap
        ]

        # Sort by count (most consistent first)
        consensus.sort(key=lambda x: feature_counts[x], reverse=True)

        logger.info("Found %d consensus features (min_overlap=%d)",
                   len(consensus), min_overlap)

        return consensus

    def run_full_selection(
        self,
        data: pd.DataFrame,
        binary_labels: np.ndarray,
        multiclass_labels: np.ndarray,
        timepoints: Optional[np.ndarray] = None,
        levels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run complete feature selection across all levels and strategies.

        Performs comprehensive feature selection using all three strategies
        (binary, multiclass, time-series) across all or specified stringency
        levels. Returns detailed results including feature counts, statistics,
        and consensus features.

        Args:
            data: Feature matrix with features as rows and samples as columns.
            binary_labels: Binary label array for two-group comparison.
            multiclass_labels: Multiclass label array for multi-group comparison.
            timepoints: Optional timepoint array for temporal analysis.
            levels: Optional list of specific levels to run. If None, runs all.

        Returns:
            Dictionary containing:
            - 'binary': Dict of level -> selected features
            - 'multiclass': Dict of level -> selected features
            - 'timeseries': Dict of level -> selected features (if timepoints provided)
            - 'consensus': Dict of level -> consensus features
            - 'summary': DataFrame summarizing feature counts per level
            - 'statistics': Dict of detailed statistics per strategy

        Example:
            >>> selector = TenLevelFeatureSelector()
            >>>
            >>> results = selector.run_full_selection(
            ...     data=methylation_df,
            ...     binary_labels=treatment_labels,
            ...     multiclass_labels=duration_labels,
            ...     timepoints=week_array
            ... )
            >>>
            >>> # Access summary
            >>> print(results["summary"])
            >>>
            >>> # Get L5 consensus features
            >>> l5_consensus = results["consensus"]["L5_moderate"]
        """
        levels = levels or self.config.list_levels()

        logger.info("Running full selection across %d levels", len(levels))

        # Initialize result containers
        results = {
            'binary': {},
            'multiclass': {},
            'timeseries': {},
            'consensus': {},
            'statistics': {
                'binary': {},
                'multiclass': {},
                'timeseries': {}
            },
            'summary': None
        }

        summary_data = []

        for level in levels:
            logger.info("Processing level: %s", level)

            # Binary selection
            binary_features, binary_stats = self.select_binary_features(
                data, binary_labels, level=level, return_statistics=True
            )
            results['binary'][level] = binary_features
            results['statistics']['binary'][level] = binary_stats

            # Multiclass selection
            multi_features, multi_stats = self.select_multiclass_features(
                data, multiclass_labels, level=level, return_statistics=True
            )
            results['multiclass'][level] = multi_features
            results['statistics']['multiclass'][level] = multi_stats

            # Time-series selection (if timepoints provided)
            if timepoints is not None:
                ts_features, ts_stats = self.select_timeseries_features(
                    data, timepoints, level=level, return_statistics=True
                )
                results['timeseries'][level] = ts_features
                results['statistics']['timeseries'][level] = ts_stats
            else:
                ts_features = []

            # Consensus features
            consensus = self.get_consensus_features(
                binary_features, multi_features,
                ts_features if timepoints is not None else None,
                min_overlap=2
            )
            results['consensus'][level] = consensus

            # Summary row
            summary_data.append({
                'level': level,
                'binary_count': len(binary_features),
                'multiclass_count': len(multi_features),
                'timeseries_count': len(ts_features) if timepoints is not None else 0,
                'consensus_count': len(consensus)
            })

        # Create summary DataFrame
        results['summary'] = pd.DataFrame(summary_data)
        results['summary'].set_index('level', inplace=True)

        # Store results
        self.results_ = results

        logger.info("Full selection complete. Results stored in 'results_' attribute.")

        return results

    def get_recommended_level(
        self,
        target_features: int = 100,
        strategy: str = "binary"
    ) -> str:
        """
        Recommend a stringency level based on target feature count.

        Analyzes previous selection results to suggest an appropriate
        stringency level that yields approximately the target number
        of features.

        Args:
            target_features: Desired approximate number of features.
            strategy: Selection strategy to base recommendation on
                     ("binary", "multiclass", "timeseries", or "consensus").

        Returns:
            Recommended level name.

        Raises:
            ValueError: If run_full_selection has not been called yet.

        Example:
            >>> selector = TenLevelFeatureSelector()
            >>> results = selector.run_full_selection(data, binary_labels, multi_labels)
            >>>
            >>> # Get recommended level for ~100 features
            >>> level = selector.get_recommended_level(target_features=100)
            >>> print(f"Recommended level: {level}")
        """
        if not self.results_:
            raise ValueError(
                "No results available. Run 'run_full_selection' first."
            )

        if strategy not in ['binary', 'multiclass', 'timeseries', 'consensus']:
            raise ValueError(f"Invalid strategy: {strategy}")

        count_col = f"{strategy}_count"
        summary = self.results_['summary']

        # Find level closest to target
        diffs = np.abs(summary[count_col] - target_features)
        recommended_idx = diffs.idxmin()

        logger.info("Recommended level '%s' for ~%d features (actual: %d)",
                   recommended_idx, target_features,
                   summary.loc[recommended_idx, count_col])

        return recommended_idx

    def export_results(
        self,
        output_path: str,
        format: str = "csv"
    ) -> None:
        """
        Export selection results to file.

        Args:
            output_path: Path to output file (without extension).
            format: Output format ("csv", "json", or "pickle").

        Raises:
            ValueError: If no results available or invalid format.

        Example:
            >>> selector = TenLevelFeatureSelector()
            >>> results = selector.run_full_selection(data, labels, multi_labels)
            >>> selector.export_results("feature_selection_results", format="csv")
        """
        if not self.results_:
            raise ValueError("No results to export. Run selection first.")

        import json as json_lib
        import pickle

        if format == "csv":
            # Export summary
            self.results_['summary'].to_csv(f"{output_path}_summary.csv")

            # Export feature lists per level
            for strategy in ['binary', 'multiclass', 'consensus']:
                for level, features in self.results_[strategy].items():
                    level_short = level.split('_')[0]
                    pd.DataFrame({'feature': features}).to_csv(
                        f"{output_path}_{strategy}_{level_short}.csv",
                        index=False
                    )

            logger.info("Results exported to %s_*.csv", output_path)

        elif format == "json":
            # Prepare JSON-serializable results
            json_results = {
                'binary': self.results_['binary'],
                'multiclass': self.results_['multiclass'],
                'timeseries': self.results_['timeseries'],
                'consensus': self.results_['consensus'],
                'summary': self.results_['summary'].to_dict()
            }

            with open(f"{output_path}.json", 'w') as f:
                json_lib.dump(json_results, f, indent=2)

            logger.info("Results exported to %s.json", output_path)

        elif format == "pickle":
            with open(f"{output_path}.pkl", 'wb') as f:
                pickle.dump(self.results_, f)

            logger.info("Results exported to %s.pkl", output_path)

        else:
            raise ValueError(f"Unsupported format: {format}")
