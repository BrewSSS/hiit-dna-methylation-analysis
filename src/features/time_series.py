"""
Time-series feature analysis for DNA methylation data.

This module provides methods for analyzing temporal patterns in
DNA methylation data, including trend detection, trajectory scoring,
and temporal clustering.

Particularly useful for longitudinal studies examining methylation
changes over time (e.g., during exercise interventions).

Example:
    >>> from src.features.time_series import TimeSeriesFeatureAnalyzer
    >>>
    >>> analyzer = TimeSeriesFeatureAnalyzer()
    >>> trend_results = analyzer.detect_monotonic_trend(
    ...     data=methylation_df,
    ...     timepoints=week_array
    ... )
    >>> print(trend_results.head())
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
import logging

# Configure module logger
logger = logging.getLogger(__name__)


class TimeSeriesFeatureAnalyzer:
    """
    Analyzer for temporal patterns in DNA methylation data.

    Provides methods for detecting monotonic trends, calculating
    trajectory scores, and clustering features by temporal patterns.
    Designed for longitudinal methylation studies with multiple
    time points.

    Attributes:
        n_jobs: Number of parallel jobs for computation.
        random_state: Random seed for reproducibility.
        results_: Dictionary storing analysis results.

    Example:
        >>> from src.features.time_series import TimeSeriesFeatureAnalyzer
        >>>
        >>> # Initialize analyzer
        >>> analyzer = TimeSeriesFeatureAnalyzer(n_jobs=4)
        >>>
        >>> # Detect monotonic trends
        >>> trends = analyzer.detect_monotonic_trend(
        ...     data=methylation_df,
        ...     timepoints=np.array([0, 4, 8, 12])  # weeks
        ... )
        >>>
        >>> # Get features with significant increasing trends
        >>> increasing = trends[
        ...     (trends['pvalue'] < 0.05) &
        ...     (trends['correlation'] > 0.5)
        ... ]
    """

    def __init__(
        self,
        n_jobs: int = -1,
        random_state: int = 42
    ):
        """
        Initialize TimeSeriesFeatureAnalyzer.

        Args:
            n_jobs: Number of parallel jobs (-1 uses all cores).
            random_state: Random seed for reproducibility.

        Example:
            >>> analyzer = TimeSeriesFeatureAnalyzer(n_jobs=4)
        """
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.results_: Dict[str, Any] = {}

    def detect_monotonic_trend(
        self,
        data: pd.DataFrame,
        timepoints: np.ndarray,
        method: str = "spearman"
    ) -> pd.DataFrame:
        """
        Detect monotonic trends in feature values over time.

        Calculates correlation between feature values and time for
        each feature to identify monotonically increasing or decreasing
        patterns.

        Args:
            data: Feature matrix with features as rows and samples as columns.
                  Shape: (n_features, n_samples)
            timepoints: Numeric timepoint array for each sample.
                       Length must match number of columns in data.
            method: Correlation method:
                   - "spearman": Spearman rank correlation (default)
                   - "pearson": Pearson linear correlation
                   - "kendall": Kendall tau correlation

        Returns:
            DataFrame with columns:
            - 'correlation': Correlation coefficient with time
            - 'pvalue': Statistical significance of correlation
            - 'direction': 'increasing', 'decreasing', or 'stable'
            - 'abs_correlation': Absolute correlation value

        Raises:
            ValueError: If timepoints length doesn't match sample count.

        Example:
            >>> analyzer = TimeSeriesFeatureAnalyzer()
            >>>
            >>> # Detect trends across 4 time points
            >>> trends = analyzer.detect_monotonic_trend(
            ...     data=methylation_df,
            ...     timepoints=np.array([0, 4, 8, 12])
            ... )
            >>>
            >>> # Filter for significant increasing trends
            >>> increasing = trends[
            ...     (trends['pvalue'] < 0.05) &
            ...     (trends['direction'] == 'increasing')
            ... ]
            >>> print(f"Found {len(increasing)} features with increasing trends")
        """
        timepoints = np.asarray(timepoints)

        if len(timepoints) != data.shape[1]:
            raise ValueError(
                f"Timepoints length ({len(timepoints)}) does not match "
                f"number of samples ({data.shape[1]})"
            )

        logger.info("Detecting monotonic trends for %d features using %s correlation",
                   len(data), method)

        # Select correlation function
        if method == "spearman":
            corr_func = stats.spearmanr
        elif method == "pearson":
            corr_func = stats.pearsonr
        elif method == "kendall":
            corr_func = stats.kendalltau
        else:
            raise ValueError(f"Unknown method: {method}")

        def compute_correlation(row_values):
            """Compute correlation for a single feature."""
            try:
                corr, pval = corr_func(timepoints, row_values)
                return corr, pval
            except Exception:
                return 0.0, 1.0

        # Compute correlations for all features
        results = []
        for idx in data.index:
            row_values = data.loc[idx].values
            corr, pval = compute_correlation(row_values)
            results.append({
                'feature': idx,
                'correlation': corr,
                'pvalue': pval
            })

        results_df = pd.DataFrame(results)
        results_df.set_index('feature', inplace=True)

        # Add derived columns
        results_df['abs_correlation'] = np.abs(results_df['correlation'])

        # Classify direction
        def classify_direction(corr, pval, threshold=0.3):
            if pval > 0.1:
                return 'stable'
            elif corr > threshold:
                return 'increasing'
            elif corr < -threshold:
                return 'decreasing'
            else:
                return 'stable'

        results_df['direction'] = results_df.apply(
            lambda row: classify_direction(row['correlation'], row['pvalue']),
            axis=1
        )

        # Store results
        self.results_['monotonic_trends'] = results_df

        logger.info(
            "Trend detection complete. Increasing: %d, Decreasing: %d, Stable: %d",
            (results_df['direction'] == 'increasing').sum(),
            (results_df['direction'] == 'decreasing').sum(),
            (results_df['direction'] == 'stable').sum()
        )

        return results_df

    def calculate_trajectory_score(
        self,
        data: pd.DataFrame,
        timepoints: np.ndarray,
        method: str = "linear"
    ) -> pd.DataFrame:
        """
        Calculate trajectory scores quantifying temporal patterns.

        Computes scores that capture the magnitude and consistency
        of temporal changes in feature values.

        Args:
            data: Feature matrix (features x samples).
            timepoints: Numeric timepoint array for each sample.
            method: Scoring method:
                   - "linear": Linear regression slope
                   - "range": Total range of change
                   - "cv": Coefficient of variation over time
                   - "auc": Area under the trajectory curve

        Returns:
            DataFrame with trajectory scores and related statistics.

        Example:
            >>> analyzer = TimeSeriesFeatureAnalyzer()
            >>>
            >>> # Calculate linear trajectory scores
            >>> scores = analyzer.calculate_trajectory_score(
            ...     data=methylation_df,
            ...     timepoints=week_array,
            ...     method="linear"
            ... )
            >>>
            >>> # Get features with largest positive slopes
            >>> top_increasing = scores.nlargest(50, 'score')
        """
        timepoints = np.asarray(timepoints)

        if len(timepoints) != data.shape[1]:
            raise ValueError(
                f"Timepoints length ({len(timepoints)}) does not match "
                f"number of samples ({data.shape[1]})"
            )

        logger.info("Calculating trajectory scores using '%s' method for %d features",
                   method, len(data))

        unique_timepoints = np.unique(timepoints)
        n_timepoints = len(unique_timepoints)

        results = []

        for idx in data.index:
            row_values = data.loc[idx].values

            if method == "linear":
                # Linear regression slope
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    timepoints, row_values
                )
                score = slope
                extra = {
                    'r_squared': r_value**2,
                    'pvalue': p_value,
                    'std_error': std_err
                }

            elif method == "range":
                # Calculate mean values at each timepoint
                means = [row_values[timepoints == t].mean() for t in unique_timepoints]
                score = max(means) - min(means)
                extra = {
                    'min_value': min(means),
                    'max_value': max(means),
                    'timepoint_of_max': unique_timepoints[np.argmax(means)]
                }

            elif method == "cv":
                # Coefficient of variation
                mean_val = np.mean(row_values)
                std_val = np.std(row_values)
                score = std_val / mean_val if mean_val != 0 else 0
                extra = {
                    'mean': mean_val,
                    'std': std_val
                }

            elif method == "auc":
                # Area under curve (using trapezoidal rule on means)
                means = [row_values[timepoints == t].mean() for t in unique_timepoints]
                score = np.trapz(means, unique_timepoints)
                extra = {
                    'baseline': means[0],
                    'final': means[-1]
                }

            else:
                raise ValueError(f"Unknown method: {method}")

            result = {'feature': idx, 'score': score}
            result.update(extra)
            results.append(result)

        results_df = pd.DataFrame(results)
        results_df.set_index('feature', inplace=True)

        # Sort by absolute score
        results_df['abs_score'] = np.abs(results_df['score'])
        results_df = results_df.sort_values('abs_score', ascending=False)

        # Store results
        self.results_['trajectory_scores'] = results_df

        logger.info("Trajectory scoring complete. Mean score: %.6f",
                   results_df['score'].mean())

        return results_df

    def cluster_trajectories(
        self,
        data: pd.DataFrame,
        timepoints: np.ndarray,
        n_clusters: int = 5,
        method: str = "kmeans"
    ) -> Tuple[pd.DataFrame, Dict[int, List[str]]]:
        """
        Cluster features by their temporal trajectory patterns.

        Groups features with similar temporal patterns, useful for
        identifying co-regulated features or functional modules.

        Args:
            data: Feature matrix (features x samples).
            timepoints: Numeric timepoint array for each sample.
            n_clusters: Number of clusters to form.
            method: Clustering method:
                   - "kmeans": K-means on trajectory means
                   - "hierarchical": Hierarchical clustering
                   - "dtw": Dynamic time warping distance (if available)

        Returns:
            Tuple of:
            - DataFrame with cluster assignments
            - Dictionary mapping cluster ID to feature list

        Example:
            >>> analyzer = TimeSeriesFeatureAnalyzer()
            >>>
            >>> # Cluster features into 5 trajectory patterns
            >>> assignments, clusters = analyzer.cluster_trajectories(
            ...     data=methylation_df,
            ...     timepoints=week_array,
            ...     n_clusters=5
            ... )
            >>>
            >>> # Get features in cluster 0
            >>> cluster0_features = clusters[0]
        """
        timepoints = np.asarray(timepoints)
        unique_timepoints = np.sort(np.unique(timepoints))

        logger.info("Clustering %d features into %d trajectory clusters using %s",
                   len(data), n_clusters, method)

        # Compute mean trajectory for each feature
        trajectories = []
        feature_names = []

        for idx in data.index:
            row_values = data.loc[idx].values
            means = [row_values[timepoints == t].mean() for t in unique_timepoints]
            trajectories.append(means)
            feature_names.append(idx)

        trajectories = np.array(trajectories)

        # Standardize trajectories for clustering
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        trajectories_scaled = scaler.fit_transform(trajectories)

        # Perform clustering
        if method == "kmeans":
            clusterer = KMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                n_init=10
            )
            cluster_labels = clusterer.fit_predict(trajectories_scaled)

        elif method == "hierarchical":
            linkage_matrix = linkage(trajectories_scaled, method='ward')
            cluster_labels = fcluster(
                linkage_matrix,
                n_clusters,
                criterion='maxclust'
            ) - 1  # Convert to 0-indexed

        else:
            raise ValueError(f"Unknown clustering method: {method}")

        # Create results DataFrame
        results_df = pd.DataFrame({
            'feature': feature_names,
            'cluster': cluster_labels
        })
        results_df.set_index('feature', inplace=True)

        # Add trajectory statistics
        for i, tp in enumerate(unique_timepoints):
            results_df[f'mean_t{int(tp)}'] = trajectories[:, i]

        # Create cluster dictionary
        cluster_dict = {}
        for cluster_id in range(n_clusters):
            cluster_features = results_df[results_df['cluster'] == cluster_id].index.tolist()
            cluster_dict[cluster_id] = cluster_features

        # Store results
        self.results_['trajectory_clusters'] = {
            'assignments': results_df,
            'cluster_dict': cluster_dict,
            'n_clusters': n_clusters,
            'method': method
        }

        # Log cluster sizes
        for cluster_id, features in cluster_dict.items():
            logger.info("Cluster %d: %d features", cluster_id, len(features))

        return results_df, cluster_dict

    def get_cluster_centroids(self) -> pd.DataFrame:
        """
        Get centroid trajectories for each cluster.

        Returns:
            DataFrame with cluster centroids (mean trajectory per cluster).

        Raises:
            ValueError: If cluster_trajectories has not been called.

        Example:
            >>> analyzer = TimeSeriesFeatureAnalyzer()
            >>> assignments, clusters = analyzer.cluster_trajectories(
            ...     data=methylation_df,
            ...     timepoints=week_array,
            ...     n_clusters=5
            ... )
            >>> centroids = analyzer.get_cluster_centroids()
            >>> print(centroids)
        """
        if 'trajectory_clusters' not in self.results_:
            raise ValueError(
                "No clustering results. Call cluster_trajectories() first."
            )

        assignments = self.results_['trajectory_clusters']['assignments']

        # Extract trajectory columns (mean_t*)
        traj_cols = [col for col in assignments.columns if col.startswith('mean_t')]

        # Compute centroids
        centroids = assignments.groupby('cluster')[traj_cols].mean()

        return centroids

    def detect_response_patterns(
        self,
        data: pd.DataFrame,
        timepoints: np.ndarray,
        baseline_timepoint: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Detect response patterns relative to baseline.

        Classifies features by their response pattern:
        - Early responders (change at early timepoints)
        - Late responders (change at later timepoints)
        - Sustained response (consistent change)
        - Transient response (temporary change)
        - Non-responders (no significant change)

        Args:
            data: Feature matrix (features x samples).
            timepoints: Numeric timepoint array for each sample.
            baseline_timepoint: Timepoint to use as baseline.
                              Defaults to minimum timepoint.

        Returns:
            DataFrame with response pattern classifications.

        Example:
            >>> analyzer = TimeSeriesFeatureAnalyzer()
            >>>
            >>> # Detect response patterns
            >>> patterns = analyzer.detect_response_patterns(
            ...     data=methylation_df,
            ...     timepoints=week_array,
            ...     baseline_timepoint=0
            ... )
            >>>
            >>> # Get early responders
            >>> early = patterns[patterns['pattern'] == 'early_responder']
        """
        timepoints = np.asarray(timepoints)
        unique_timepoints = np.sort(np.unique(timepoints))

        if baseline_timepoint is None:
            baseline_timepoint = unique_timepoints[0]

        logger.info("Detecting response patterns for %d features", len(data))

        results = []

        for idx in data.index:
            row_values = data.loc[idx].values

            # Calculate mean at each timepoint
            means = {
                t: row_values[timepoints == t].mean()
                for t in unique_timepoints
            }

            baseline_value = means[baseline_timepoint]

            # Calculate fold changes from baseline
            fold_changes = {
                t: (v - baseline_value) / (abs(baseline_value) + 1e-10)
                for t, v in means.items()
                if t != baseline_timepoint
            }

            # Determine response pattern
            fc_values = list(fold_changes.values())
            fc_timepoints = list(fold_changes.keys())

            if len(fc_values) == 0:
                pattern = 'non_responder'
                max_fc = 0
                time_of_max = baseline_timepoint
            else:
                max_fc = max(fc_values, key=abs)
                max_idx = fc_values.index(max_fc)
                time_of_max = fc_timepoints[max_idx]

                # Classify pattern based on timing and persistence
                if abs(max_fc) < 0.1:
                    pattern = 'non_responder'
                elif time_of_max == unique_timepoints[1]:  # First post-baseline
                    if abs(fc_values[-1]) > 0.5 * abs(max_fc):
                        pattern = 'early_sustained'
                    else:
                        pattern = 'early_transient'
                elif time_of_max == unique_timepoints[-1]:  # Last timepoint
                    pattern = 'late_responder'
                else:
                    if abs(fc_values[-1]) > 0.5 * abs(max_fc):
                        pattern = 'sustained_responder'
                    else:
                        pattern = 'transient_responder'

            results.append({
                'feature': idx,
                'pattern': pattern,
                'max_fold_change': max_fc,
                'time_of_max_change': time_of_max,
                'baseline_value': baseline_value,
                'final_value': means[unique_timepoints[-1]]
            })

        results_df = pd.DataFrame(results)
        results_df.set_index('feature', inplace=True)

        # Store results
        self.results_['response_patterns'] = results_df

        # Log pattern distribution
        pattern_counts = results_df['pattern'].value_counts()
        for pattern, count in pattern_counts.items():
            logger.info("Pattern '%s': %d features", pattern, count)

        return results_df

    def get_features_by_pattern(
        self,
        pattern: str
    ) -> List[str]:
        """
        Get features matching a specific response pattern.

        Args:
            pattern: Response pattern name (e.g., 'early_sustained').

        Returns:
            List of feature names matching the pattern.

        Raises:
            ValueError: If detect_response_patterns has not been called.

        Example:
            >>> analyzer = TimeSeriesFeatureAnalyzer()
            >>> patterns = analyzer.detect_response_patterns(data, timepoints)
            >>> early_features = analyzer.get_features_by_pattern('early_sustained')
        """
        if 'response_patterns' not in self.results_:
            raise ValueError(
                "No response patterns. Call detect_response_patterns() first."
            )

        patterns_df = self.results_['response_patterns']
        features = patterns_df[patterns_df['pattern'] == pattern].index.tolist()

        return features

    def compute_temporal_variance(
        self,
        data: pd.DataFrame,
        timepoints: np.ndarray
    ) -> pd.DataFrame:
        """
        Compute variance components for temporal analysis.

        Decomposes total variance into between-timepoint and
        within-timepoint components.

        Args:
            data: Feature matrix (features x samples).
            timepoints: Numeric timepoint array for each sample.

        Returns:
            DataFrame with variance components.

        Example:
            >>> analyzer = TimeSeriesFeatureAnalyzer()
            >>> variance = analyzer.compute_temporal_variance(data, timepoints)
            >>> high_temporal = variance[variance['temporal_ratio'] > 0.5]
        """
        timepoints = np.asarray(timepoints)
        unique_timepoints = np.unique(timepoints)

        logger.info("Computing temporal variance for %d features", len(data))

        results = []

        for idx in data.index:
            row_values = data.loc[idx].values

            # Total variance
            total_var = np.var(row_values)

            # Between-timepoint variance (variance of means)
            means = [row_values[timepoints == t].mean() for t in unique_timepoints]
            between_var = np.var(means)

            # Within-timepoint variance (average variance within each timepoint)
            within_vars = [
                np.var(row_values[timepoints == t])
                for t in unique_timepoints
            ]
            within_var = np.mean(within_vars)

            # Temporal variance ratio
            temporal_ratio = between_var / (total_var + 1e-10)

            results.append({
                'feature': idx,
                'total_variance': total_var,
                'between_variance': between_var,
                'within_variance': within_var,
                'temporal_ratio': temporal_ratio
            })

        results_df = pd.DataFrame(results)
        results_df.set_index('feature', inplace=True)
        results_df = results_df.sort_values('temporal_ratio', ascending=False)

        # Store results
        self.results_['temporal_variance'] = results_df

        return results_df
