"""
Multi-version data comparison utilities for HIIT methylation analysis.

This module provides tools to compare model performance across different
preprocessing versions, enabling systematic evaluation of preprocessing
strategies on classification outcomes.

Classes:
    MultiVersionComparator: Compare models across preprocessing versions
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


class MultiVersionComparator:
    """
    Compare model performance across different data preprocessing versions.

    This class is a core innovation of the HIIT methylation project, enabling
    systematic comparison of how different preprocessing strategies affect
    classification performance. It supports statistical testing to determine
    significant differences between versions.

    Parameters
    ----------
    versions : list of str, default=['original', 'standardized', 'batch_standardized']
        Names of preprocessing versions to compare.
    cv : int, default=5
        Number of cross-validation folds.
    n_repeats : int, default=3
        Number of CV repetitions.
    random_state : int, default=42
        Random seed for reproducibility.

    Attributes
    ----------
    results_ : dict
        Dictionary storing comparison results.
    version_scores_ : dict
        Dictionary mapping version names to score arrays.
    statistical_results_ : dict
        Results of statistical comparisons between versions.

    Examples
    --------
    >>> comparator = MultiVersionComparator(
    ...     versions=['original', 'standardized', 'batch_corrected']
    ... )
    >>> X_versions = {
    ...     'original': X_orig,
    ...     'standardized': X_std,
    ...     'batch_corrected': X_batch
    ... }
    >>> results = comparator.compare_preprocessing(X_versions, y, model)
    >>> comparator.plot_version_comparison(results)
    """

    def __init__(
        self,
        versions: Optional[List[str]] = None,
        cv: int = 5,
        n_repeats: int = 3,
        random_state: int = 42
    ):
        self.versions = versions or [
            'original', 'standardized', 'batch_standardized'
        ]
        self.cv = cv
        self.n_repeats = n_repeats
        self.random_state = random_state

        # Initialize storage
        self.results_ = {}
        self.version_scores_ = {}
        self.statistical_results_ = {}

    def _get_cv_splitter(self) -> StratifiedKFold:
        """Get cross-validation splitter."""
        return StratifiedKFold(
            n_splits=self.cv,
            shuffle=True,
            random_state=self.random_state
        )

    def prepare_versions(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        batch_info: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        include_original: bool = True,
        include_standardized: bool = True,
        include_batch_standardized: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Prepare multiple versions of the data for comparison.

        This method creates standardized versions of the input data,
        optionally incorporating batch information for batch-aware
        standardization.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Original feature matrix.
        batch_info : array-like of shape (n_samples,) or (n_samples, n_batches), optional
            Batch information. Can be:
            - 1D array of batch labels
            - 2D one-hot encoded batch matrix
        include_original : bool, default=True
            Whether to include the original (unprocessed) data.
        include_standardized : bool, default=True
            Whether to include globally standardized data.
        include_batch_standardized : bool, default=True
            Whether to include batch-wise standardized data.
            Requires batch_info.

        Returns
        -------
        X_versions : dict
            Dictionary mapping version names to preprocessed feature matrices.
            Possible keys: 'original', 'standardized', 'batch_standardized'.

        Notes
        -----
        The batch-standardized version applies z-score normalization
        within each batch separately, which can help remove batch-specific
        location and scale effects while preserving within-batch variation.

        Examples
        --------
        >>> comparator = MultiVersionComparator()
        >>> X_versions = comparator.prepare_versions(X, batch_info=batch_labels)
        >>> results = comparator.compare_preprocessing(X_versions, y, model)
        """
        from sklearn.preprocessing import StandardScaler

        X_array = np.asarray(X) if isinstance(X, pd.DataFrame) else X.copy()

        X_versions = {}

        # Original data
        if include_original:
            X_versions['original'] = X_array.copy()

        # Globally standardized
        if include_standardized:
            scaler = StandardScaler()
            X_versions['standardized'] = scaler.fit_transform(X_array)

        # Batch-wise standardized
        if include_batch_standardized and batch_info is not None:
            batch_array = np.asarray(batch_info)

            # Handle one-hot encoded batch info
            if batch_array.ndim == 2:
                # Convert one-hot to labels
                batch_labels = np.argmax(batch_array, axis=1)
            else:
                batch_labels = batch_array

            unique_batches = np.unique(batch_labels)

            X_batch_std = np.zeros_like(X_array, dtype=float)

            for batch in unique_batches:
                batch_mask = batch_labels == batch
                batch_data = X_array[batch_mask]

                if len(batch_data) > 1:
                    # Standardize within batch
                    batch_scaler = StandardScaler()
                    X_batch_std[batch_mask] = batch_scaler.fit_transform(batch_data)
                else:
                    # Single sample in batch - use global mean/std
                    global_mean = np.mean(X_array, axis=0)
                    global_std = np.std(X_array, axis=0)
                    global_std[global_std == 0] = 1.0
                    X_batch_std[batch_mask] = (batch_data - global_mean) / global_std

            X_versions['batch_standardized'] = X_batch_std

        elif include_batch_standardized and batch_info is None:
            warnings.warn(
                "batch_info not provided; skipping batch_standardized version."
            )

        return X_versions

    def compare_preprocessing(
        self,
        X_versions: Dict[str, Union[np.ndarray, pd.DataFrame]],
        y: Union[np.ndarray, pd.Series],
        model: BaseEstimator,
        scoring: str = 'roc_auc',
        n_jobs: int = -1
    ) -> Dict[str, Any]:
        """
        Compare model performance across different preprocessing versions.

        Parameters
        ----------
        X_versions : dict
            Dictionary mapping version names to feature matrices.
            Each value should be array-like of shape (n_samples, n_features).
        y : array-like of shape (n_samples,)
            Target labels (same for all versions).
        model : BaseEstimator
            Sklearn-compatible classifier to evaluate.
        scoring : str, default='roc_auc'
            Scoring metric for evaluation.
        n_jobs : int, default=-1
            Number of parallel jobs for cross-validation.

        Returns
        -------
        results : dict
            Dictionary containing:
            - 'version_scores': Scores for each version
            - 'summary': Summary statistics per version
            - 'ranking': Versions ranked by performance
        """
        y_array = np.asarray(y)

        results = {
            'version_scores': {},
            'summary': {},
            'ranking': []
        }

        version_means = []

        for version_name in self.versions:
            if version_name not in X_versions:
                warnings.warn(
                    f"Version '{version_name}' not found in X_versions. Skipping."
                )
                continue

            X = X_versions[version_name]
            X_array = np.asarray(X) if isinstance(X, pd.DataFrame) else X

            # Perform repeated cross-validation
            all_scores = []
            for repeat in range(self.n_repeats):
                cv_splitter = StratifiedKFold(
                    n_splits=self.cv,
                    shuffle=True,
                    random_state=self.random_state + repeat
                )

                scores = cross_val_score(
                    clone(model),
                    X_array,
                    y_array,
                    cv=cv_splitter,
                    scoring=scoring,
                    n_jobs=n_jobs
                )
                all_scores.extend(scores)

            all_scores = np.array(all_scores)

            # Store results
            results['version_scores'][version_name] = all_scores
            self.version_scores_[version_name] = all_scores

            # Calculate summary statistics
            n = len(all_scores)
            mean = np.mean(all_scores)
            std = np.std(all_scores, ddof=1)
            se = std / np.sqrt(n)

            # 95% confidence interval
            t_critical = stats.t.ppf(0.975, df=n - 1)
            ci_lower = mean - t_critical * se
            ci_upper = mean + t_critical * se

            results['summary'][version_name] = {
                'mean': mean,
                'std': std,
                'se': se,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'median': np.median(all_scores),
                'min': np.min(all_scores),
                'max': np.max(all_scores),
                'n_folds': n
            }

            version_means.append((version_name, mean))

        # Rank versions by mean performance
        version_means.sort(key=lambda x: x[1], reverse=True)
        results['ranking'] = [v[0] for v in version_means]

        self.results_ = results
        return results

    def statistical_comparison(
        self,
        results: Optional[Dict[str, Any]] = None,
        test: str = 'wilcoxon',
        alpha: float = 0.05,
        correction: str = 'bonferroni'
    ) -> Dict[str, Any]:
        """
        Perform statistical tests comparing versions.

        Parameters
        ----------
        results : dict, optional
            Results from compare_preprocessing(). If None, uses stored results.
        test : str, default='wilcoxon'
            Statistical test to use. Options: 'wilcoxon', 'ttest', 'mannwhitney'.
        alpha : float, default=0.05
            Significance level.
        correction : str, default='bonferroni'
            Multiple testing correction method.

        Returns
        -------
        statistical_results : dict
            Dictionary containing:
            - 'pairwise_tests': P-values for all pairwise comparisons
            - 'significant_pairs': Pairs with significant differences
            - 'effect_sizes': Effect sizes for each comparison
        """
        if results is None:
            results = self.results_

        if not results or 'version_scores' not in results:
            raise ValueError(
                "No results available. Run compare_preprocessing() first."
            )

        version_scores = results['version_scores']
        versions = list(version_scores.keys())
        n_versions = len(versions)

        # Number of pairwise comparisons
        n_comparisons = n_versions * (n_versions - 1) // 2

        statistical_results = {
            'pairwise_tests': {},
            'significant_pairs': [],
            'effect_sizes': {},
            'test_used': test,
            'alpha': alpha,
            'correction': correction,
            'corrected_alpha': alpha / n_comparisons if correction == 'bonferroni' else alpha
        }

        # Perform pairwise comparisons
        for i, v1 in enumerate(versions):
            for j, v2 in enumerate(versions):
                if i >= j:
                    continue

                scores1 = version_scores[v1]
                scores2 = version_scores[v2]

                # Ensure equal lengths for paired tests
                min_len = min(len(scores1), len(scores2))
                s1 = scores1[:min_len]
                s2 = scores2[:min_len]

                # Perform statistical test
                try:
                    if test == 'wilcoxon':
                        statistic, p_value = stats.wilcoxon(s1, s2)
                    elif test == 'ttest':
                        statistic, p_value = stats.ttest_rel(s1, s2)
                    elif test == 'mannwhitney':
                        statistic, p_value = stats.mannwhitneyu(
                            scores1, scores2, alternative='two-sided'
                        )
                    else:
                        raise ValueError(f"Unknown test: {test}")
                except Exception as e:
                    warnings.warn(f"Statistical test failed for {v1} vs {v2}: {e}")
                    statistic, p_value = np.nan, np.nan

                pair_key = f"{v1}_vs_{v2}"

                statistical_results['pairwise_tests'][pair_key] = {
                    'statistic': statistic,
                    'p_value': p_value,
                    'significant': p_value < statistical_results['corrected_alpha']
                }

                # Calculate effect size (Cohen's d)
                pooled_std = np.sqrt(
                    (np.var(s1, ddof=1) + np.var(s2, ddof=1)) / 2
                )
                if pooled_std > 0:
                    cohens_d = (np.mean(s1) - np.mean(s2)) / pooled_std
                else:
                    cohens_d = 0.0

                statistical_results['effect_sizes'][pair_key] = {
                    'cohens_d': cohens_d,
                    'interpretation': self._interpret_effect_size(cohens_d)
                }

                # Track significant pairs
                if p_value < statistical_results['corrected_alpha']:
                    winner = v1 if np.mean(s1) > np.mean(s2) else v2
                    statistical_results['significant_pairs'].append({
                        'pair': (v1, v2),
                        'p_value': p_value,
                        'effect_size': cohens_d,
                        'better_version': winner
                    })

        self.statistical_results_ = statistical_results
        return statistical_results

    def _interpret_effect_size(self, d: float) -> str:
        """
        Interpret Cohen's d effect size.

        Parameters
        ----------
        d : float
            Cohen's d value.

        Returns
        -------
        interpretation : str
            Verbal interpretation of effect size.
        """
        d_abs = abs(d)
        if d_abs < 0.2:
            return 'negligible'
        elif d_abs < 0.5:
            return 'small'
        elif d_abs < 0.8:
            return 'medium'
        else:
            return 'large'

    def generate_comparison_report(
        self,
        results: Optional[Dict[str, Any]] = None,
        include_stats: bool = True
    ) -> str:
        """
        Generate a markdown report of the comparison results.

        Parameters
        ----------
        results : dict, optional
            Results from compare_preprocessing(). If None, uses stored results.
        include_stats : bool, default=True
            Whether to include statistical test results.

        Returns
        -------
        report : str
            Markdown-formatted comparison report.
        """
        if results is None:
            results = self.results_

        if not results:
            return "No results available. Run compare_preprocessing() first."

        lines = [
            "# Multi-Version Preprocessing Comparison Report",
            "",
            "## Overview",
            "",
            f"Number of versions compared: {len(results.get('version_scores', {}))}",
            f"Cross-validation folds: {self.cv}",
            f"Repetitions: {self.n_repeats}",
            "",
            "## Performance Summary",
            "",
            "| Version | Mean Score | Std | 95% CI | Rank |",
            "|---------|------------|-----|--------|------|"
        ]

        ranking = results.get('ranking', [])
        for rank, version in enumerate(ranking, 1):
            summary = results['summary'].get(version, {})
            mean = summary.get('mean', np.nan)
            std = summary.get('std', np.nan)
            ci_lower = summary.get('ci_lower', np.nan)
            ci_upper = summary.get('ci_upper', np.nan)

            lines.append(
                f"| {version} | {mean:.4f} | {std:.4f} | "
                f"[{ci_lower:.4f}, {ci_upper:.4f}] | {rank} |"
            )

        lines.extend(["", "## Best Performing Version", ""])

        if ranking:
            best = ranking[0]
            best_summary = results['summary'].get(best, {})
            lines.append(f"**{best}** achieved the highest mean score of "
                        f"{best_summary.get('mean', 0):.4f} (+/- {best_summary.get('std', 0):.4f})")

        # Add statistical comparison if available and requested
        if include_stats and self.statistical_results_:
            stats_results = self.statistical_results_

            lines.extend([
                "",
                "## Statistical Comparison",
                "",
                f"Test used: {stats_results.get('test_used', 'N/A')}",
                f"Significance level (alpha): {stats_results.get('alpha', 0.05)}",
                f"Correction method: {stats_results.get('correction', 'N/A')}",
                f"Corrected alpha: {stats_results.get('corrected_alpha', 0.05):.4f}",
                "",
                "### Pairwise Comparisons",
                "",
                "| Comparison | P-value | Effect Size (d) | Interpretation | Significant |",
                "|------------|---------|-----------------|----------------|-------------|"
            ])

            pairwise = stats_results.get('pairwise_tests', {})
            effects = stats_results.get('effect_sizes', {})

            for pair_key, test_result in pairwise.items():
                p_val = test_result.get('p_value', np.nan)
                sig = 'Yes' if test_result.get('significant', False) else 'No'

                effect_info = effects.get(pair_key, {})
                d = effect_info.get('cohens_d', np.nan)
                interp = effect_info.get('interpretation', 'N/A')

                lines.append(
                    f"| {pair_key.replace('_vs_', ' vs ')} | "
                    f"{p_val:.4f} | {d:.4f} | {interp} | {sig} |"
                )

            # Significant findings
            sig_pairs = stats_results.get('significant_pairs', [])
            if sig_pairs:
                lines.extend([
                    "",
                    "### Significant Findings",
                    ""
                ])
                for finding in sig_pairs:
                    v1, v2 = finding['pair']
                    winner = finding['better_version']
                    lines.append(
                        f"- **{winner}** significantly outperforms "
                        f"the other version (p={finding['p_value']:.4f}, "
                        f"d={finding['effect_size']:.4f})"
                    )
            else:
                lines.extend([
                    "",
                    "### No Significant Differences",
                    "",
                    "No statistically significant differences were found between "
                    "preprocessing versions after multiple testing correction."
                ])

        lines.extend([
            "",
            "---",
            "*Report generated by MultiVersionComparator*"
        ])

        return "\n".join(lines)

    def plot_version_comparison(
        self,
        results: Optional[Dict[str, Any]] = None,
        figsize: Tuple[int, int] = (12, 5),
        title: str = 'Preprocessing Version Comparison',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize comparison across preprocessing versions.

        Parameters
        ----------
        results : dict, optional
            Results from compare_preprocessing(). If None, uses stored results.
        figsize : tuple, default=(12, 5)
            Figure size.
        title : str, default='Preprocessing Version Comparison'
            Plot title.
        save_path : str, optional
            Path to save the figure.

        Returns
        -------
        fig : Figure
            Matplotlib figure object.
        """
        if results is None:
            results = self.results_

        if not results or 'version_scores' not in results:
            raise ValueError(
                "No results available. Run compare_preprocessing() first."
            )

        version_scores = results['version_scores']
        summary = results['summary']

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        versions = list(version_scores.keys())
        n_versions = len(versions)
        colors = plt.cm.Set2(np.linspace(0, 1, n_versions))

        # Box plot comparison
        ax1 = axes[0]
        box_data = [version_scores[v] for v in versions]
        bp = ax1.boxplot(box_data, labels=versions, patch_artist=True)

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax1.set_ylabel('Score')
        ax1.set_title('Score Distribution by Version')
        ax1.tick_params(axis='x', rotation=45)

        # Add individual points
        for i, (version, scores) in enumerate(version_scores.items()):
            x = np.random.normal(i + 1, 0.04, size=len(scores))
            ax1.scatter(x, scores, alpha=0.3, color=colors[i], s=20)

        # Bar plot with error bars
        ax2 = axes[1]
        means = [summary[v]['mean'] for v in versions]
        stds = [summary[v]['std'] for v in versions]
        ci_errors = [
            [summary[v]['mean'] - summary[v]['ci_lower'] for v in versions],
            [summary[v]['ci_upper'] - summary[v]['mean'] for v in versions]
        ]

        x_pos = np.arange(n_versions)
        bars = ax2.bar(x_pos, means, yerr=ci_errors, capsize=5, color=colors,
                       alpha=0.7, edgecolor='black')

        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(versions, rotation=45, ha='right')
        ax2.set_ylabel('Mean Score')
        ax2.set_title('Mean Score with 95% CI')

        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax2.annotate(f'{mean:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

        fig.suptitle(title, fontsize=14)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_pairwise_comparison(
        self,
        version1: str,
        version2: str,
        figsize: Tuple[int, int] = (10, 4),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create detailed pairwise comparison visualization.

        Parameters
        ----------
        version1 : str
            Name of first version.
        version2 : str
            Name of second version.
        figsize : tuple, default=(10, 4)
            Figure size.
        save_path : str, optional
            Path to save the figure.

        Returns
        -------
        fig : Figure
            Matplotlib figure object.
        """
        if version1 not in self.version_scores_ or version2 not in self.version_scores_:
            raise ValueError(
                f"Version scores not found. Available: {list(self.version_scores_.keys())}"
            )

        scores1 = self.version_scores_[version1]
        scores2 = self.version_scores_[version2]

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Paired difference plot
        ax1 = axes[0]
        min_len = min(len(scores1), len(scores2))
        differences = scores1[:min_len] - scores2[:min_len]

        ax1.hist(differences, bins='auto', edgecolor='black', alpha=0.7)
        ax1.axvline(x=0, color='red', linestyle='--', label='No difference')
        ax1.axvline(x=np.mean(differences), color='green', linestyle='-',
                   label=f'Mean diff: {np.mean(differences):.4f}')
        ax1.set_xlabel(f'Score Difference ({version1} - {version2})')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Differences')
        ax1.legend()

        # Scatter plot
        ax2 = axes[1]
        ax2.scatter(scores1[:min_len], scores2[:min_len], alpha=0.5)
        lims = [
            min(ax2.get_xlim()[0], ax2.get_ylim()[0]),
            max(ax2.get_xlim()[1], ax2.get_ylim()[1])
        ]
        ax2.plot(lims, lims, 'k--', alpha=0.5, label='y = x')
        ax2.set_xlabel(f'{version1} Score')
        ax2.set_ylabel(f'{version2} Score')
        ax2.set_title('Pairwise Score Comparison')
        ax2.legend()

        # Box plot comparison
        ax3 = axes[2]
        bp = ax3.boxplot([scores1, scores2], labels=[version1, version2],
                        patch_artist=True)
        colors = ['#66b3ff', '#ff9999']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax3.set_ylabel('Score')
        ax3.set_title('Side-by-Side Comparison')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def export_results(
        self,
        filepath: str,
        format: str = 'csv'
    ) -> None:
        """
        Export comparison results to file.

        Parameters
        ----------
        filepath : str
            Output file path.
        format : str, default='csv'
            Export format: 'csv', 'json', or 'excel'.
        """
        if not self.results_:
            raise ValueError(
                "No results to export. Run compare_preprocessing() first."
            )

        # Create summary DataFrame
        summary_data = []
        for version, stats in self.results_.get('summary', {}).items():
            row = {'version': version}
            row.update(stats)
            summary_data.append(row)

        df = pd.DataFrame(summary_data)

        if format == 'csv':
            df.to_csv(filepath, index=False)
        elif format == 'json':
            df.to_json(filepath, orient='records', indent=2)
        elif format == 'excel':
            df.to_excel(filepath, index=False)
        else:
            raise ValueError(f"Unknown format: {format}. Use 'csv', 'json', or 'excel'.")

        print(f"Results exported to {filepath}")
