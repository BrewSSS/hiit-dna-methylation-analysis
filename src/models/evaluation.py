"""
Model evaluation utilities for HIIT methylation classification.

This module provides comprehensive evaluation tools including metrics
calculation, cross-validation strategies, and visualization functions.

Classes:
    ModelEvaluator: Calculate and report classification metrics
    CrossValidationStrategy: Repeated stratified cross-validation
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    auc
)
from sklearn.model_selection import (
    StratifiedKFold,
    RepeatedStratifiedKFold,
    cross_val_score,
    cross_val_predict
)
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings


class ModelEvaluator:
    """
    Comprehensive model evaluation for classification tasks.

    This class calculates various classification metrics and provides
    visualization tools for model performance analysis.

    Parameters
    ----------
    metrics : list of str, optional
        List of metrics to calculate. If None, uses all available metrics.
        Available: 'accuracy', 'precision', 'recall', 'f1', 'auc',
                  'specificity', 'sensitivity'
    average : str, default='weighted'
        Averaging method for multiclass metrics.
        Options: 'micro', 'macro', 'weighted', 'binary'

    Attributes
    ----------
    results_ : dict
        Dictionary storing evaluation results.
    confusion_matrix_ : ndarray
        Confusion matrix from the last evaluation.

    Examples
    --------
    >>> evaluator = ModelEvaluator(metrics=['accuracy', 'f1', 'auc'])
    >>> results = evaluator.evaluate(y_true, y_pred, y_prob)
    >>> evaluator.plot_confusion_matrix(y_true, y_pred)
    """

    AVAILABLE_METRICS = [
        'accuracy', 'precision', 'recall', 'f1', 'auc',
        'specificity', 'sensitivity'
    ]

    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        average: str = 'weighted'
    ):
        if metrics is None:
            self.metrics = self.AVAILABLE_METRICS.copy()
        else:
            # Validate metrics
            invalid = set(metrics) - set(self.AVAILABLE_METRICS)
            if invalid:
                raise ValueError(
                    f"Invalid metrics: {invalid}. "
                    f"Available: {self.AVAILABLE_METRICS}"
                )
            self.metrics = metrics

        self.average = average
        self.results_ = {}
        self.confusion_matrix_ = None

    def _calculate_specificity(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Calculate specificity (true negative rate).

        Parameters
        ----------
        y_true : array-like
            True labels.
        y_pred : array-like
            Predicted labels.

        Returns
        -------
        specificity : float
            Specificity score.
        """
        cm = confusion_matrix(y_true, y_pred)

        if cm.shape[0] == 2:
            # Binary classification
            tn, fp, fn, tp = cm.ravel()
            if tn + fp == 0:
                return 0.0
            return tn / (tn + fp)
        else:
            # Multiclass: average specificity
            specificities = []
            for i in range(cm.shape[0]):
                # True negatives: sum of all elements except row i and column i
                tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
                # False positives: sum of column i except diagonal
                fp = np.sum(cm[:, i]) - cm[i, i]
                if tn + fp > 0:
                    specificities.append(tn / (tn + fp))
            return np.mean(specificities) if specificities else 0.0

    def _calculate_sensitivity(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Calculate sensitivity (recall/true positive rate).

        This is equivalent to recall but provided for clarity in
        medical/clinical contexts.

        Parameters
        ----------
        y_true : array-like
            True labels.
        y_pred : array-like
            Predicted labels.

        Returns
        -------
        sensitivity : float
            Sensitivity score.
        """
        return recall_score(y_true, y_pred, average=self.average, zero_division=0)

    def evaluate(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        y_prob: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> Dict[str, float]:
        """
        Calculate all specified metrics.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True class labels.
        y_pred : array-like of shape (n_samples,)
            Predicted class labels.
        y_prob : array-like of shape (n_samples,) or (n_samples, n_classes), optional
            Predicted probabilities. Required for AUC calculation.

        Returns
        -------
        results : dict
            Dictionary mapping metric names to scores.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        results = {}
        n_classes = len(np.unique(y_true))
        is_binary = n_classes == 2

        for metric in self.metrics:
            if metric == 'accuracy':
                results['accuracy'] = accuracy_score(y_true, y_pred)

            elif metric == 'precision':
                avg = 'binary' if is_binary else self.average
                results['precision'] = precision_score(
                    y_true, y_pred, average=avg, zero_division=0
                )

            elif metric == 'recall':
                avg = 'binary' if is_binary else self.average
                results['recall'] = recall_score(
                    y_true, y_pred, average=avg, zero_division=0
                )

            elif metric == 'f1':
                avg = 'binary' if is_binary else self.average
                results['f1'] = f1_score(
                    y_true, y_pred, average=avg, zero_division=0
                )

            elif metric == 'auc':
                if y_prob is not None:
                    y_prob_array = np.asarray(y_prob)
                    try:
                        if is_binary:
                            # For binary, use probability of positive class
                            if y_prob_array.ndim == 2:
                                y_prob_array = y_prob_array[:, 1]
                            results['auc'] = roc_auc_score(y_true, y_prob_array)
                        else:
                            # For multiclass, use one-vs-rest
                            results['auc'] = roc_auc_score(
                                y_true, y_prob_array,
                                multi_class='ovr',
                                average=self.average
                            )
                    except ValueError as e:
                        warnings.warn(f"Could not calculate AUC: {e}")
                        results['auc'] = np.nan
                else:
                    results['auc'] = np.nan

            elif metric == 'specificity':
                results['specificity'] = self._calculate_specificity(y_true, y_pred)

            elif metric == 'sensitivity':
                results['sensitivity'] = self._calculate_sensitivity(y_true, y_pred)

        self.results_ = results
        self.confusion_matrix_ = confusion_matrix(y_true, y_pred)

        return results

    def bootstrap_confidence_interval(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        metric: str = 'auc',
        y_prob: Optional[Union[np.ndarray, pd.Series]] = None,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        random_state: int = 42
    ) -> Dict[str, float]:
        """
        Calculate bootstrap confidence intervals for a metric.

        Uses the percentile method to estimate confidence intervals
        by resampling with replacement.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True class labels.
        y_pred : array-like of shape (n_samples,)
            Predicted class labels.
        metric : str, default='auc'
            Metric to compute confidence interval for.
            Must be one of AVAILABLE_METRICS.
        y_prob : array-like, optional
            Predicted probabilities. Required when metric='auc'.
        n_bootstrap : int, default=1000
            Number of bootstrap iterations.
        confidence : float, default=0.95
            Confidence level (e.g., 0.95 for 95% CI).
        random_state : int, default=42
            Random seed for reproducibility.

        Returns
        -------
        result : dict
            Dictionary containing:
            - 'point_estimate': The metric computed on the full data
            - 'ci_lower': Lower bound of confidence interval
            - 'ci_upper': Upper bound of confidence interval
            - 'bootstrap_scores': Array of bootstrap metric values
            - 'std': Standard deviation of bootstrap distribution
        """
        if metric not in self.AVAILABLE_METRICS:
            raise ValueError(
                f"Invalid metric: {metric}. Available: {self.AVAILABLE_METRICS}"
            )

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_prob is not None:
            y_prob = np.asarray(y_prob)

        rng = np.random.RandomState(random_state)
        n_samples = len(y_true)
        bootstrap_scores = []

        # Create a temporary evaluator for single-metric computation
        single_evaluator = ModelEvaluator(metrics=[metric], average=self.average)

        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = rng.randint(0, n_samples, size=n_samples)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            y_prob_boot = y_prob[indices] if y_prob is not None else None

            # Ensure at least two classes in the bootstrap sample
            if len(np.unique(y_true_boot)) < 2:
                continue

            try:
                result = single_evaluator.evaluate(
                    y_true_boot, y_pred_boot, y_prob_boot
                )
                score = result.get(metric, np.nan)
                if not np.isnan(score):
                    bootstrap_scores.append(score)
            except Exception:
                continue

        bootstrap_scores = np.array(bootstrap_scores)

        if len(bootstrap_scores) == 0:
            warnings.warn("No valid bootstrap samples. Returning NaN values.")
            return {
                'point_estimate': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'bootstrap_scores': np.array([]),
                'std': np.nan
            }

        # Compute point estimate on full data
        point_result = single_evaluator.evaluate(y_true, y_pred, y_prob)
        point_estimate = point_result.get(metric, np.nan)

        # Percentile confidence interval
        alpha = 1 - confidence
        ci_lower = np.percentile(bootstrap_scores, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_scores, 100 * (1 - alpha / 2))

        return {
            'point_estimate': point_estimate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'bootstrap_scores': bootstrap_scores,
            'std': np.std(bootstrap_scores)
        }

    def generate_report(
        self,
        results: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate a comprehensive evaluation report.

        Parameters
        ----------
        results : dict
            Dictionary of evaluation results, typically from evaluate() or
            from a collection of evaluation runs. Supports nested dicts
            for multi-model comparisons.
        output_path : str, optional
            File path to write the report. If None, only returns the string.

        Returns
        -------
        report : str
            Formatted evaluation report text.
        """
        lines = [
            "=" * 60,
            "MODEL EVALUATION REPORT",
            "=" * 60,
            ""
        ]

        # Handle single evaluation results
        if all(isinstance(v, (int, float, np.floating)) for v in results.values()):
            lines.append("Metric Results:")
            lines.append("-" * 40)
            for metric, value in results.items():
                if isinstance(value, float):
                    lines.append(f"  {metric:>20s}: {value:.4f}")
                else:
                    lines.append(f"  {metric:>20s}: {value}")

        # Handle multi-model results
        elif all(isinstance(v, dict) for v in results.values()):
            for model_name, model_results in results.items():
                lines.append(f"\nModel: {model_name}")
                lines.append("-" * 40)
                for metric, value in model_results.items():
                    if isinstance(value, (int, float, np.floating)):
                        lines.append(f"  {metric:>20s}: {value:.4f}")
                    else:
                        lines.append(f"  {metric:>20s}: {value}")

        # Append confusion matrix if available
        if self.confusion_matrix_ is not None:
            lines.append("")
            lines.append("Confusion Matrix:")
            lines.append(str(self.confusion_matrix_))

        lines.extend(["", "=" * 60])
        report = "\n".join(lines)

        if output_path is not None:
            with open(output_path, 'w') as f:
                f.write(report)

        return report

    def generate_classification_report(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        target_names: Optional[List[str]] = None,
        output_format: str = 'dict'
    ) -> Union[str, Dict]:
        """
        Generate a detailed classification report.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True class labels.
        y_pred : array-like of shape (n_samples,)
            Predicted class labels.
        target_names : list of str, optional
            Display names for target classes.
        output_format : str, default='dict'
            Output format: 'dict' or 'text'.

        Returns
        -------
        report : dict or str
            Classification report in specified format.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if output_format == 'text':
            return classification_report(
                y_true, y_pred, target_names=target_names, zero_division=0
            )
        else:
            return classification_report(
                y_true, y_pred, target_names=target_names,
                output_dict=True, zero_division=0
            )

    def plot_confusion_matrix(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        labels: Optional[List[str]] = None,
        normalize: bool = False,
        cmap: str = 'Blues',
        figsize: Tuple[int, int] = (8, 6),
        title: str = 'Confusion Matrix',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot confusion matrix as a heatmap.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True class labels.
        y_pred : array-like of shape (n_samples,)
            Predicted class labels.
        labels : list of str, optional
            Display labels for classes.
        normalize : bool, default=False
            Whether to normalize the confusion matrix.
        cmap : str, default='Blues'
            Colormap for the heatmap.
        figsize : tuple, default=(8, 6)
            Figure size.
        title : str, default='Confusion Matrix'
            Plot title.
        save_path : str, optional
            Path to save the figure.

        Returns
        -------
        fig : Figure
            Matplotlib figure object.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'

        if labels is None:
            labels = np.unique(np.concatenate([y_true, y_pred]))

        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax
        )

        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(title)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_roc_curve(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_prob: Union[np.ndarray, pd.Series],
        labels: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (8, 6),
        title: str = 'ROC Curve',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot ROC curve with AUC score.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True class labels.
        y_prob : array-like of shape (n_samples,) or (n_samples, n_classes)
            Predicted probabilities.
        labels : list of str, optional
            Class labels for legend.
        figsize : tuple, default=(8, 6)
            Figure size.
        title : str, default='ROC Curve'
            Plot title.
        save_path : str, optional
            Path to save the figure.

        Returns
        -------
        fig : Figure
            Matplotlib figure object.
        """
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        fig, ax = plt.subplots(figsize=figsize)

        unique_classes = np.unique(y_true)
        n_classes = len(unique_classes)

        if n_classes == 2:
            # Binary classification
            if y_prob.ndim == 2:
                y_prob = y_prob[:, 1]

            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)

            ax.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        else:
            # Multiclass: plot one-vs-rest for each class
            colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

            for i, cls in enumerate(unique_classes):
                y_true_binary = (y_true == cls).astype(int)
                y_prob_cls = y_prob[:, i] if y_prob.ndim == 2 else y_prob

                fpr, tpr, _ = roc_curve(y_true_binary, y_prob_cls)
                roc_auc = auc(fpr, tpr)

                label = labels[i] if labels else f'Class {cls}'
                ax.plot(
                    fpr, tpr, color=colors[i], lw=2,
                    label=f'{label} (AUC = {roc_auc:.3f})'
                )

        # Plot diagonal reference line
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc='lower right')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_precision_recall_curve(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_prob: Union[np.ndarray, pd.Series],
        figsize: Tuple[int, int] = (8, 6),
        title: str = 'Precision-Recall Curve',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot precision-recall curve.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True class labels.
        y_prob : array-like of shape (n_samples,)
            Predicted probabilities for the positive class.
        figsize : tuple, default=(8, 6)
            Figure size.
        title : str, default='Precision-Recall Curve'
            Plot title.
        save_path : str, optional
            Path to save the figure.

        Returns
        -------
        fig : Figure
            Matplotlib figure object.
        """
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        if y_prob.ndim == 2:
            y_prob = y_prob[:, 1]

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)

        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(recall, precision, lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')

        # Baseline (proportion of positive class)
        baseline = np.mean(y_true)
        ax.axhline(y=baseline, color='k', linestyle='--', lw=1,
                   label=f'Baseline ({baseline:.3f})')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(title)
        ax.legend(loc='lower left')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


class CrossValidationStrategy:
    """
    Repeated stratified cross-validation strategy for model evaluation.

    This class provides a robust cross-validation framework with repeated
    stratified splits and comprehensive score tracking.

    Parameters
    ----------
    cv : int, default=5
        Number of cross-validation folds.
    n_repeats : int, default=3
        Number of repetitions for repeated CV.
    stratified : bool, default=True
        Whether to use stratified splitting.
    random_state : int, default=42
        Random seed for reproducibility.
    scoring : str or list, default='roc_auc'
        Scoring metric(s) for evaluation.

    Attributes
    ----------
    cv_results_ : dict
        Dictionary storing all CV results.
    fold_scores_ : list
        Scores for each fold.

    Examples
    --------
    >>> cv_strategy = CrossValidationStrategy(cv=5, n_repeats=3)
    >>> results = cv_strategy.cross_validate(model, X, y)
    >>> summary = cv_strategy.get_summary_statistics()
    """

    def __init__(
        self,
        cv: int = 5,
        n_repeats: int = 3,
        stratified: bool = True,
        random_state: int = 42,
        scoring: Union[str, List[str]] = 'roc_auc'
    ):
        self.cv = cv
        self.n_repeats = n_repeats
        self.stratified = stratified
        self.random_state = random_state
        self.scoring = scoring

        self.cv_results_ = {}
        self.fold_scores_ = []

    @staticmethod
    def stratified_kfold(
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42
    ) -> StratifiedKFold:
        """
        Create a stratified K-fold cross-validation splitter.

        Factory method for creating a simple stratified K-fold splitter
        without repeated cross-validation.

        Parameters
        ----------
        n_splits : int, default=5
            Number of folds.
        shuffle : bool, default=True
            Whether to shuffle samples before splitting.
        random_state : int, default=42
            Random seed for reproducibility.

        Returns
        -------
        cv : StratifiedKFold
            Configured StratifiedKFold splitter.

        Examples
        --------
        >>> cv = CrossValidationStrategy.stratified_kfold(n_splits=5)
        >>> for train_idx, test_idx in cv.split(X, y):
        ...     # Train/test on fold
        ...     pass
        """
        return StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )

    @staticmethod
    def repeated_stratified_kfold(
        n_splits: int = 5,
        n_repeats: int = 3,
        random_state: int = 42
    ) -> RepeatedStratifiedKFold:
        """
        Create a repeated stratified K-fold cross-validation splitter.

        Factory method for creating a repeated stratified K-fold splitter
        that runs multiple rounds of stratified cross-validation.

        Parameters
        ----------
        n_splits : int, default=5
            Number of folds per repetition.
        n_repeats : int, default=3
            Number of times to repeat cross-validation.
        random_state : int, default=42
            Random seed for reproducibility.

        Returns
        -------
        cv : RepeatedStratifiedKFold
            Configured RepeatedStratifiedKFold splitter.

        Examples
        --------
        >>> cv = CrossValidationStrategy.repeated_stratified_kfold(
        ...     n_splits=5, n_repeats=3
        ... )
        >>> # This will create 15 total folds (5 folds x 3 repeats)
        """
        return RepeatedStratifiedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_state
        )

    @staticmethod
    def leave_one_subject_out(
        subject_ids: Union[np.ndarray, pd.Series, List]
    ):
        """
        Create a leave-one-subject-out cross-validation splitter.

        This is particularly useful for repeated measures designs where
        the same subject has multiple samples (e.g., pre/post measurements).
        Ensures all samples from one subject are in the test set together.

        Parameters
        ----------
        subject_ids : array-like of shape (n_samples,)
            Subject identifiers for each sample. Samples with the same
            subject_id will be grouped together.

        Returns
        -------
        cv : LeaveOneGroupOut
            Configured LeaveOneGroupOut splitter.

        Notes
        -----
        This returns a LeaveOneGroupOut splitter that can be used with
        sklearn's cross_val_score by passing subject_ids as the `groups`
        parameter.

        Examples
        --------
        >>> subject_ids = ['S1', 'S1', 'S2', 'S2', 'S3', 'S3']  # 2 samples per subject
        >>> cv = CrossValidationStrategy.leave_one_subject_out(subject_ids)
        >>> for train_idx, test_idx in cv.split(X, y, groups=subject_ids):
        ...     # Train on all subjects except one, test on held-out subject
        ...     pass
        """
        from sklearn.model_selection import LeaveOneGroupOut
        return LeaveOneGroupOut()

    def _get_cv_splitter(self) -> Union[StratifiedKFold, RepeatedStratifiedKFold]:
        """Get the cross-validation splitter."""
        if self.n_repeats > 1:
            return RepeatedStratifiedKFold(
                n_splits=self.cv,
                n_repeats=self.n_repeats,
                random_state=self.random_state
            )
        else:
            return StratifiedKFold(
                n_splits=self.cv,
                shuffle=True,
                random_state=self.random_state
            )

    def cross_validate(
        self,
        model: BaseEstimator,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        n_jobs: int = -1,
        return_estimator: bool = False
    ) -> Dict[str, Any]:
        """
        Perform repeated stratified cross-validation.

        Parameters
        ----------
        model : BaseEstimator
            Sklearn-compatible classifier.
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            Target labels.
        n_jobs : int, default=-1
            Number of parallel jobs.
        return_estimator : bool, default=False
            Whether to return fitted estimators for each fold.

        Returns
        -------
        results : dict
            Dictionary containing:
            - 'scores': Array of scores for each fold
            - 'mean': Mean score across folds
            - 'std': Standard deviation of scores
            - 'estimators': List of fitted estimators (if requested)
        """
        from sklearn.model_selection import cross_validate as sklearn_cv
        from sklearn.base import clone

        X_array = np.asarray(X) if isinstance(X, pd.DataFrame) else X
        y_array = np.asarray(y)

        cv_splitter = self._get_cv_splitter()

        # Handle multiple scoring metrics
        if isinstance(self.scoring, list):
            scoring_dict = {s: s for s in self.scoring}
        else:
            scoring_dict = self.scoring

        cv_results = sklearn_cv(
            clone(model),
            X_array,
            y_array,
            cv=cv_splitter,
            scoring=scoring_dict,
            n_jobs=n_jobs,
            return_estimator=return_estimator,
            return_train_score=True
        )

        # Process results
        results = {}

        if isinstance(self.scoring, list):
            for metric in self.scoring:
                test_key = f'test_{metric}'
                train_key = f'train_{metric}'

                results[metric] = {
                    'test_scores': cv_results[test_key],
                    'train_scores': cv_results[train_key],
                    'mean_test': np.mean(cv_results[test_key]),
                    'std_test': np.std(cv_results[test_key]),
                    'mean_train': np.mean(cv_results[train_key]),
                    'std_train': np.std(cv_results[train_key])
                }
        else:
            results['scores'] = cv_results['test_score']
            results['train_scores'] = cv_results['train_score']
            results['mean'] = np.mean(cv_results['test_score'])
            results['std'] = np.std(cv_results['test_score'])

        results['fit_time'] = cv_results['fit_time']
        results['score_time'] = cv_results['score_time']

        if return_estimator:
            results['estimators'] = cv_results['estimator']

        self.cv_results_ = results
        self.fold_scores_ = (
            cv_results['test_score'] if 'test_score' in cv_results
            else cv_results[f'test_{self.scoring[0]}']
        )

        return results

    def get_cv_scores(self) -> np.ndarray:
        """
        Get scores for each cross-validation fold.

        Returns
        -------
        scores : ndarray
            Array of scores for each fold.
        """
        return np.array(self.fold_scores_)

    def get_summary_statistics(
        self,
        confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """
        Calculate summary statistics including confidence intervals.

        Parameters
        ----------
        confidence_level : float, default=0.95
            Confidence level for interval calculation.

        Returns
        -------
        summary : dict
            Dictionary containing:
            - 'mean': Mean score
            - 'std': Standard deviation
            - 'median': Median score
            - 'min': Minimum score
            - 'max': Maximum score
            - 'ci_lower': Lower bound of confidence interval
            - 'ci_upper': Upper bound of confidence interval
            - 'n_folds': Number of folds
        """
        scores = self.get_cv_scores()

        if len(scores) == 0:
            raise ValueError(
                "No CV scores available. Run cross_validate() first."
            )

        n = len(scores)
        mean = np.mean(scores)
        std = np.std(scores, ddof=1)
        se = std / np.sqrt(n)

        # Calculate confidence interval using t-distribution
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha / 2, df=n - 1)

        ci_lower = mean - t_critical * se
        ci_upper = mean + t_critical * se

        return {
            'mean': mean,
            'std': std,
            'median': np.median(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_folds': n,
            'confidence_level': confidence_level
        }

    def plot_cv_scores(
        self,
        figsize: Tuple[int, int] = (10, 6),
        title: str = 'Cross-Validation Scores',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot cross-validation scores distribution.

        Parameters
        ----------
        figsize : tuple, default=(10, 6)
            Figure size.
        title : str, default='Cross-Validation Scores'
            Plot title.
        save_path : str, optional
            Path to save the figure.

        Returns
        -------
        fig : Figure
            Matplotlib figure object.
        """
        scores = self.get_cv_scores()
        summary = self.get_summary_statistics()

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Box plot
        axes[0].boxplot(scores, vert=True)
        axes[0].scatter([1] * len(scores), scores, alpha=0.5, color='blue')
        axes[0].axhline(y=summary['mean'], color='red', linestyle='--',
                        label=f'Mean: {summary["mean"]:.3f}')
        axes[0].set_ylabel('Score')
        axes[0].set_title('Score Distribution')
        axes[0].legend()

        # Histogram
        axes[1].hist(scores, bins='auto', edgecolor='black', alpha=0.7)
        axes[1].axvline(x=summary['mean'], color='red', linestyle='--',
                        label=f'Mean: {summary["mean"]:.3f}')
        axes[1].axvline(x=summary['ci_lower'], color='green', linestyle=':',
                        label=f'95% CI: [{summary["ci_lower"]:.3f}, {summary["ci_upper"]:.3f}]')
        axes[1].axvline(x=summary['ci_upper'], color='green', linestyle=':')
        axes[1].set_xlabel('Score')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Score Histogram')
        axes[1].legend()

        fig.suptitle(title)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def compare_models(
        self,
        models: Dict[str, BaseEstimator],
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        n_jobs: int = -1
    ) -> pd.DataFrame:
        """
        Compare multiple models using cross-validation.

        Parameters
        ----------
        models : dict
            Dictionary mapping model names to estimators.
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            Target labels.
        n_jobs : int, default=-1
            Number of parallel jobs.

        Returns
        -------
        comparison : DataFrame
            DataFrame with comparison results.
        """
        from sklearn.base import clone

        results = []

        for name, model in models.items():
            cv_results = self.cross_validate(
                clone(model), X, y, n_jobs=n_jobs
            )

            summary = self.get_summary_statistics()

            results.append({
                'model': name,
                'mean_score': summary['mean'],
                'std_score': summary['std'],
                'ci_lower': summary['ci_lower'],
                'ci_upper': summary['ci_upper'],
                'min_score': summary['min'],
                'max_score': summary['max']
            })

        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.sort_values(
            'mean_score', ascending=False
        ).reset_index(drop=True)

        return comparison_df
