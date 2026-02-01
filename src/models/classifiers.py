"""
Batch-aware classification models for HIIT methylation analysis.

This module implements classifiers that can incorporate batch information
as covariates to handle batch effects in DNA methylation data.

Classes:
    BatchAwareClassifier: Wrapper for sklearn classifiers with batch awareness
    HIITClassificationPipeline: Complete pipeline for HIIT classification tasks
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import warnings


@dataclass
class ClassifierConfig:
    """Configuration for classification models.

    Attributes:
        classifier_type: Type of classifier to use. One of
            'logistic_regression', 'random_forest', 'svm', 'gradient_boosting'.
        include_batch: Whether to include batch information as covariates.
        standardize: Whether to standardize features before classification.
        random_state: Random seed for reproducibility.
        max_iter: Maximum iterations for iterative solvers.
        n_estimators: Number of estimators for ensemble methods.
        max_depth: Maximum tree depth for tree-based methods.
        C: Regularization parameter for logistic regression and SVM.
        learning_rate: Learning rate for gradient boosting.
        class_weight: Strategy for handling class imbalance.
    """
    classifier_type: str = "logistic_regression"
    include_batch: bool = True
    standardize: bool = True
    random_state: int = 42
    max_iter: int = 1000
    n_estimators: int = 200
    max_depth: Optional[int] = None
    C: float = 1.0
    learning_rate: float = 0.1
    class_weight: Optional[Union[str, Dict]] = "balanced"

    def build_classifier(self) -> BaseEstimator:
        """Build a sklearn classifier from this configuration.

        Returns:
            A configured sklearn classifier instance.
        """
        if self.classifier_type == "logistic_regression":
            return LogisticRegression(
                C=self.C,
                penalty='l2',
                solver='lbfgs',
                max_iter=self.max_iter,
                class_weight=self.class_weight,
                random_state=self.random_state
            )
        elif self.classifier_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                class_weight=self.class_weight,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.classifier_type == "svm":
            return SVC(
                C=self.C,
                kernel='rbf',
                probability=True,
                class_weight=self.class_weight,
                random_state=self.random_state
            )
        elif self.classifier_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth or 3,
                learning_rate=self.learning_rate,
                random_state=self.random_state
            )
        else:
            raise ValueError(
                f"Unknown classifier type: {self.classifier_type}. "
                f"Choose from: logistic_regression, random_forest, svm, gradient_boosting"
            )


class BatchAwareClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier wrapper that incorporates batch information as covariates.

    This class wraps sklearn classifiers and optionally includes batch
    information during training and prediction to account for batch effects
    in methylation data.

    Parameters
    ----------
    base_classifier : BaseEstimator
        The base sklearn classifier to use. If None, defaults to
        LogisticRegression with L2 regularization.
    include_batch : bool, default=True
        Whether to include batch information as additional features.
    scale_features : bool, default=True
        Whether to standardize features before classification.
    **kwargs : dict
        Additional keyword arguments passed to the base classifier.

    Attributes
    ----------
    classifier_ : BaseEstimator
        The fitted classifier instance.
    scaler_ : StandardScaler
        The fitted scaler for feature standardization.
    batch_encoder_ : LabelEncoder
        Encoder for categorical batch labels.
    feature_names_ : list
        Names of features used in the model.
    classes_ : ndarray
        Unique class labels.

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> clf = BatchAwareClassifier(
    ...     base_classifier=LogisticRegression(),
    ...     include_batch=True
    ... )
    >>> clf.fit(X_train, y_train, batch=batch_train)
    >>> predictions = clf.predict(X_test, batch=batch_test)
    """

    def __init__(
        self,
        base_classifier: Optional[BaseEstimator] = None,
        include_batch: bool = True,
        scale_features: bool = True,
        **kwargs
    ):
        self.base_classifier = base_classifier
        self.include_batch = include_batch
        self.scale_features = scale_features
        self.kwargs = kwargs

        # Initialize internal attributes
        self.classifier_ = None
        self.scaler_ = None
        self.batch_encoder_ = None
        self.feature_names_ = None
        self.classes_ = None
        self._n_original_features = None

    def _get_classifier(self) -> BaseEstimator:
        """Get or create the base classifier instance."""
        if self.base_classifier is not None:
            return clone(self.base_classifier)
        else:
            return LogisticRegression(
                penalty='l2',
                solver='lbfgs',
                max_iter=1000,
                random_state=42,
                **self.kwargs
            )

    def _prepare_features(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        batch: Optional[Union[np.ndarray, pd.Series]] = None,
        fit: bool = False
    ) -> np.ndarray:
        """
        Prepare features by optionally adding batch covariates and scaling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        batch : array-like of shape (n_samples,), optional
            Batch labels for each sample.
        fit : bool, default=False
            Whether to fit the scaler and encoder.

        Returns
        -------
        X_prepared : ndarray
            Prepared feature matrix.
        """
        # Convert to numpy array if DataFrame
        if isinstance(X, pd.DataFrame):
            if fit:
                self.feature_names_ = X.columns.tolist()
            X = X.values

        if fit:
            self._n_original_features = X.shape[1]

        # Add batch information as features if requested
        if self.include_batch and batch is not None:
            batch_array = np.asarray(batch)

            if fit:
                self.batch_encoder_ = LabelEncoder()
                batch_encoded = self.batch_encoder_.fit_transform(batch_array)
            else:
                if self.batch_encoder_ is None:
                    raise ValueError(
                        "Batch encoder not fitted. Call fit() first."
                    )
                batch_encoded = self.batch_encoder_.transform(batch_array)

            # One-hot encode batch (drop first to avoid multicollinearity)
            n_batches = len(self.batch_encoder_.classes_)
            if n_batches > 1:
                batch_onehot = np.zeros((len(batch_encoded), n_batches - 1))
                for i, b in enumerate(batch_encoded):
                    if b > 0:
                        batch_onehot[i, b - 1] = 1
                X = np.hstack([X, batch_onehot])

        # Scale features
        if self.scale_features:
            if fit:
                self.scaler_ = StandardScaler()
                X = self.scaler_.fit_transform(X)
            else:
                if self.scaler_ is None:
                    raise ValueError("Scaler not fitted. Call fit() first.")
                X = self.scaler_.transform(X)

        return X

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        batch: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> 'BatchAwareClassifier':
        """
        Fit the batch-aware classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Target labels.
        batch : array-like of shape (n_samples,), optional
            Batch labels for each sample.

        Returns
        -------
        self : BatchAwareClassifier
            Fitted classifier.
        """
        # Convert y to numpy array
        y_array = np.asarray(y)
        self.classes_ = np.unique(y_array)

        # Prepare features
        X_prepared = self._prepare_features(X, batch, fit=True)

        # Initialize and fit classifier
        self.classifier_ = self._get_classifier()
        self.classifier_.fit(X_prepared, y_array)

        return self

    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        batch: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> np.ndarray:
        """
        Predict class labels for samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        batch : array-like of shape (n_samples,), optional
            Batch labels for each sample.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        if self.classifier_ is None:
            raise ValueError("Classifier not fitted. Call fit() first.")

        X_prepared = self._prepare_features(X, batch, fit=False)
        return self.classifier_.predict(X_prepared)

    def predict_proba(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        batch: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> np.ndarray:
        """
        Predict class probabilities for samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        batch : array-like of shape (n_samples,), optional
            Batch labels for each sample.

        Returns
        -------
        y_proba : ndarray of shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        if self.classifier_ is None:
            raise ValueError("Classifier not fitted. Call fit() first.")

        if not hasattr(self.classifier_, 'predict_proba'):
            raise AttributeError(
                f"{type(self.classifier_).__name__} does not support "
                "predict_proba. Use a classifier with probability support."
            )

        X_prepared = self._prepare_features(X, batch, fit=False)
        return self.classifier_.predict_proba(X_prepared)

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Extract feature importance weights from the fitted classifier.

        Returns
        -------
        importance : dict
            Dictionary mapping feature names to importance scores.

        Notes
        -----
        For LogisticRegression, returns absolute coefficient values.
        For tree-based models, returns feature_importances_.
        For SVM with linear kernel, returns absolute coefficient values.
        """
        if self.classifier_ is None:
            raise ValueError("Classifier not fitted. Call fit() first.")

        # Get importance values based on classifier type
        if hasattr(self.classifier_, 'coef_'):
            # Linear models (LogisticRegression, LinearSVC)
            if self.classifier_.coef_.ndim == 1:
                importance_values = np.abs(self.classifier_.coef_)
            else:
                # Multi-class: average across classes
                importance_values = np.mean(
                    np.abs(self.classifier_.coef_), axis=0
                )
        elif hasattr(self.classifier_, 'feature_importances_'):
            # Tree-based models
            importance_values = self.classifier_.feature_importances_
        else:
            raise AttributeError(
                f"{type(self.classifier_).__name__} does not provide "
                "feature importance information."
            )

        # Only return importance for original features (not batch covariates)
        importance_values = importance_values[:self._n_original_features]

        # Create feature names if not available
        if self.feature_names_ is None:
            feature_names = [f"feature_{i}" for i in range(len(importance_values))]
        else:
            feature_names = self.feature_names_[:len(importance_values)]

        return dict(zip(feature_names, importance_values))


class HIITClassificationPipeline:
    """
    Complete classification pipeline for HIIT methylation analysis.

    This class implements the Triple Analysis Strategy for HIIT studies:
    1. Binary analysis: HIIT responders vs. controls
    2. Multiclass analysis: Classify intervention duration (4W/8W/12W)
    3. Time-series analysis: Trajectory modeling across timepoints

    Parameters
    ----------
    feature_selector : callable, optional
        Feature selection function/object. If None, uses all features.
    data_versions : list of str, default=['original', 'standardized', 'batch_standardized']
        List of data preprocessing versions to compare.
    n_splits : int, default=5
        Number of cross-validation folds.
    n_repeats : int, default=3
        Number of cross-validation repetitions.
    random_state : int, default=42
        Random seed for reproducibility.
    n_jobs : int, default=-1
        Number of parallel jobs. -1 uses all available cores.

    Attributes
    ----------
    classifiers_ : dict
        Dictionary of fitted classifiers.
    results_ : dict
        Dictionary storing analysis results.
    best_model_ : BatchAwareClassifier
        Best performing model based on evaluation metrics.

    Examples
    --------
    >>> pipeline = HIITClassificationPipeline()
    >>> results = pipeline.run_binary_analysis(X, y, batch)
    >>> best_model = pipeline.get_best_model()
    """

    # Default classifiers with their parameter grids
    DEFAULT_CLASSIFIERS = {
        'LogisticRegression': {
            'classifier': LogisticRegression(
                solver='lbfgs',
                max_iter=1000,
                random_state=42
            ),
            'param_grid': {
                'C': [0.01, 0.1, 1.0, 10.0],
                'penalty': ['l2']
            }
        },
        'RandomForest': {
            'classifier': RandomForestClassifier(
                random_state=42,
                n_jobs=-1
            ),
            'param_grid': {
                'n_estimators': [100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5]
            }
        },
        'SVM': {
            'classifier': SVC(
                probability=True,
                random_state=42
            ),
            'param_grid': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
        },
        'GradientBoosting': {
            'classifier': GradientBoostingClassifier(
                random_state=42
            ),
            'param_grid': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5],
                'learning_rate': [0.01, 0.1]
            }
        }
    }

    def __init__(
        self,
        feature_selector: Optional[Any] = None,
        data_versions: Optional[List[str]] = None,
        n_splits: int = 5,
        n_repeats: int = 3,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        self.feature_selector = feature_selector
        self.data_versions = data_versions or [
            'original', 'standardized', 'batch_standardized'
        ]
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.n_jobs = n_jobs

        # Initialize storage
        self.classifiers_ = {}
        self.results_ = {}
        self.best_model_ = None
        self._best_score = -np.inf

    def _select_features(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Apply feature selection if a selector is provided."""
        if self.feature_selector is not None:
            if hasattr(self.feature_selector, 'fit_transform'):
                return self.feature_selector.fit_transform(X, y)
            elif callable(self.feature_selector):
                return self.feature_selector(X, y)
        return X

    def _get_cv(self) -> StratifiedKFold:
        """Get cross-validation splitter."""
        return StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )

    def run_binary_analysis(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        batch: Optional[Union[np.ndarray, pd.Series]] = None,
        classifiers: Optional[Dict[str, Dict]] = None,
        optimize_hyperparameters: bool = True
    ) -> Dict[str, Any]:
        """
        Run binary classification analysis (HIIT vs Control).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            Binary target labels.
        batch : array-like of shape (n_samples,), optional
            Batch labels for batch-aware classification.
        classifiers : dict, optional
            Dictionary of classifiers to use. If None, uses defaults.
        optimize_hyperparameters : bool, default=True
            Whether to perform hyperparameter optimization.

        Returns
        -------
        results : dict
            Dictionary containing:
            - 'predictions': Cross-validated predictions
            - 'probabilities': Cross-validated probability scores
            - 'models': Fitted models per classifier
            - 'best_params': Best hyperparameters per classifier
            - 'cv_scores': Cross-validation scores per classifier
        """
        y_array = np.asarray(y)
        unique_classes = np.unique(y_array)

        if len(unique_classes) != 2:
            raise ValueError(
                f"Binary analysis requires exactly 2 classes, "
                f"found {len(unique_classes)}: {unique_classes}"
            )

        # Apply feature selection
        X_selected = self._select_features(X, y_array)

        # Use provided classifiers or defaults
        clf_configs = classifiers or self.DEFAULT_CLASSIFIERS

        results = {
            'predictions': {},
            'probabilities': {},
            'models': {},
            'best_params': {},
            'cv_scores': {}
        }

        for name, config in clf_configs.items():
            print(f"Training {name}...")

            # Create batch-aware classifier
            batch_clf = BatchAwareClassifier(
                base_classifier=clone(config['classifier']),
                include_batch=(batch is not None)
            )

            if optimize_hyperparameters and 'param_grid' in config:
                # Wrap for GridSearchCV
                # Note: GridSearchCV doesn't support extra fit params easily,
                # so we prepare features first
                X_prepared = batch_clf._prepare_features(X_selected, batch, fit=True)

                grid_search = GridSearchCV(
                    estimator=clone(config['classifier']),
                    param_grid=config['param_grid'],
                    cv=self._get_cv(),
                    scoring='roc_auc',
                    n_jobs=self.n_jobs,
                    refit=True
                )

                grid_search.fit(X_prepared, y_array)

                results['best_params'][name] = grid_search.best_params_
                results['cv_scores'][name] = grid_search.cv_results_['mean_test_score']

                # Update batch classifier with best estimator
                batch_clf.classifier_ = grid_search.best_estimator_
                batch_clf.classes_ = np.unique(y_array)
            else:
                # Fit without hyperparameter optimization
                batch_clf.fit(X_selected, y_array, batch)

            # Store predictions and probabilities
            results['predictions'][name] = batch_clf.predict(X_selected, batch)

            if hasattr(batch_clf.classifier_, 'predict_proba'):
                results['probabilities'][name] = batch_clf.predict_proba(
                    X_selected, batch
                )[:, 1]

            results['models'][name] = batch_clf
            self.classifiers_[f'binary_{name}'] = batch_clf

            # Track best model
            if name in results['cv_scores'] and len(results['cv_scores'][name]) > 0:
                mean_score = np.mean(results['cv_scores'][name])
                if mean_score > self._best_score:
                    self._best_score = mean_score
                    self.best_model_ = batch_clf

        self.results_['binary'] = results
        return results

    def run_multiclass_analysis(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        batch: Optional[Union[np.ndarray, pd.Series]] = None,
        classifiers: Optional[Dict[str, Dict]] = None,
        optimize_hyperparameters: bool = True
    ) -> Dict[str, Any]:
        """
        Run multiclass classification analysis (4W/8W/12W duration).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            Multiclass target labels (e.g., intervention duration).
        batch : array-like of shape (n_samples,), optional
            Batch labels for batch-aware classification.
        classifiers : dict, optional
            Dictionary of classifiers to use. If None, uses defaults.
        optimize_hyperparameters : bool, default=True
            Whether to perform hyperparameter optimization.

        Returns
        -------
        results : dict
            Dictionary containing classification results.
        """
        y_array = np.asarray(y)
        unique_classes = np.unique(y_array)

        if len(unique_classes) < 3:
            warnings.warn(
                f"Multiclass analysis typically expects 3+ classes, "
                f"found {len(unique_classes)}"
            )

        # Apply feature selection
        X_selected = self._select_features(X, y_array)

        # Use provided classifiers or defaults
        clf_configs = classifiers or self.DEFAULT_CLASSIFIERS

        results = {
            'predictions': {},
            'probabilities': {},
            'models': {},
            'best_params': {},
            'cv_scores': {},
            'class_labels': unique_classes
        }

        for name, config in clf_configs.items():
            print(f"Training {name} (multiclass)...")

            # Create batch-aware classifier
            batch_clf = BatchAwareClassifier(
                base_classifier=clone(config['classifier']),
                include_batch=(batch is not None)
            )

            if optimize_hyperparameters and 'param_grid' in config:
                X_prepared = batch_clf._prepare_features(X_selected, batch, fit=True)

                grid_search = GridSearchCV(
                    estimator=clone(config['classifier']),
                    param_grid=config['param_grid'],
                    cv=self._get_cv(),
                    scoring='accuracy',  # Use accuracy for multiclass
                    n_jobs=self.n_jobs,
                    refit=True
                )

                grid_search.fit(X_prepared, y_array)

                results['best_params'][name] = grid_search.best_params_
                results['cv_scores'][name] = grid_search.cv_results_['mean_test_score']

                batch_clf.classifier_ = grid_search.best_estimator_
                batch_clf.classes_ = unique_classes
            else:
                batch_clf.fit(X_selected, y_array, batch)

            results['predictions'][name] = batch_clf.predict(X_selected, batch)

            if hasattr(batch_clf.classifier_, 'predict_proba'):
                results['probabilities'][name] = batch_clf.predict_proba(
                    X_selected, batch
                )

            results['models'][name] = batch_clf
            self.classifiers_[f'multiclass_{name}'] = batch_clf

        self.results_['multiclass'] = results
        return results

    def run_timeseries_analysis(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        timepoints: Union[np.ndarray, pd.Series],
        batch: Optional[Union[np.ndarray, pd.Series]] = None,
        subject_ids: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> Dict[str, Any]:
        """
        Run time-series trajectory analysis across timepoints.

        This analysis models methylation changes over time, useful for
        understanding the temporal dynamics of HIIT-induced epigenetic changes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix with samples from multiple timepoints.
        timepoints : array-like of shape (n_samples,)
            Timepoint labels for each sample (e.g., 'pre', 'post', 'followup').
        batch : array-like of shape (n_samples,), optional
            Batch labels for batch-aware analysis.
        subject_ids : array-like of shape (n_samples,), optional
            Subject identifiers for paired analysis.

        Returns
        -------
        results : dict
            Dictionary containing:
            - 'trajectory_model': Fitted trajectory classifier
            - 'timepoint_predictions': Predictions per timepoint
            - 'transition_analysis': Analysis of state transitions
        """
        timepoints_array = np.asarray(timepoints)
        unique_timepoints = np.unique(timepoints_array)

        results = {
            'timepoints': unique_timepoints.tolist(),
            'trajectory_model': None,
            'timepoint_predictions': {},
            'transition_analysis': {}
        }

        # Train classifier to predict timepoints
        trajectory_clf = BatchAwareClassifier(
            base_classifier=RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state
            ),
            include_batch=(batch is not None)
        )

        # Apply feature selection
        X_selected = self._select_features(X, timepoints_array)
        trajectory_clf.fit(X_selected, timepoints_array, batch)

        results['trajectory_model'] = trajectory_clf
        results['timepoint_predictions']['all'] = trajectory_clf.predict(
            X_selected, batch
        )

        # Analyze transitions between consecutive timepoints
        if subject_ids is not None:
            subject_array = np.asarray(subject_ids)
            unique_subjects = np.unique(subject_array)

            for subject in unique_subjects:
                subject_mask = subject_array == subject
                subject_timepoints = timepoints_array[subject_mask]
                subject_predictions = results['timepoint_predictions']['all'][subject_mask]

                if len(subject_timepoints) > 1:
                    # Sort by timepoint order
                    sorted_indices = np.argsort(subject_timepoints)
                    transitions = []
                    for i in range(len(sorted_indices) - 1):
                        t1 = subject_predictions[sorted_indices[i]]
                        t2 = subject_predictions[sorted_indices[i + 1]]
                        transitions.append((t1, t2))

                    results['transition_analysis'][subject] = transitions

        self.results_['timeseries'] = results
        self.classifiers_['timeseries'] = trajectory_clf

        return results

    def compare_classifiers(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        classifiers: Optional[Dict[str, BaseEstimator]] = None,
        batch: Optional[Union[np.ndarray, pd.Series]] = None,
        scoring: str = 'roc_auc'
    ) -> pd.DataFrame:
        """
        Compare multiple classifiers on the same data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            Target labels.
        classifiers : dict, optional
            Dictionary mapping names to classifier instances.
            If None, uses default classifiers.
        batch : array-like of shape (n_samples,), optional
            Batch labels.
        scoring : str, default='roc_auc'
            Scoring metric for comparison.

        Returns
        -------
        comparison : DataFrame
            DataFrame with classifier comparison results.
        """
        from sklearn.model_selection import cross_val_score

        if classifiers is None:
            classifiers = {
                name: config['classifier']
                for name, config in self.DEFAULT_CLASSIFIERS.items()
            }

        y_array = np.asarray(y)
        X_selected = self._select_features(X, y_array)

        comparison_results = []

        for name, clf in classifiers.items():
            batch_clf = BatchAwareClassifier(
                base_classifier=clone(clf),
                include_batch=(batch is not None)
            )

            # Prepare features for cross-validation
            X_prepared = batch_clf._prepare_features(X_selected, batch, fit=True)

            # Perform cross-validation
            cv_scores = cross_val_score(
                clone(clf),
                X_prepared,
                y_array,
                cv=self._get_cv(),
                scoring=scoring,
                n_jobs=self.n_jobs
            )

            comparison_results.append({
                'classifier': name,
                'mean_score': np.mean(cv_scores),
                'std_score': np.std(cv_scores),
                'min_score': np.min(cv_scores),
                'max_score': np.max(cv_scores),
                'scores': cv_scores.tolist()
            })

        comparison_df = pd.DataFrame(comparison_results)
        comparison_df = comparison_df.sort_values(
            'mean_score', ascending=False
        ).reset_index(drop=True)

        self.results_['classifier_comparison'] = comparison_df

        return comparison_df

    def get_best_model(self) -> Optional[BatchAwareClassifier]:
        """
        Get the best performing model from all analyses.

        Returns
        -------
        best_model : BatchAwareClassifier or None
            The best performing classifier, or None if no models fitted.
        """
        return self.best_model_

    def save_models(self, path: str) -> None:
        """
        Save all fitted models to disk.

        Parameters
        ----------
        path : str
            Directory path to save models.
        """
        import os
        from joblib import dump

        os.makedirs(path, exist_ok=True)

        for name, clf in self.classifiers_.items():
            model_path = os.path.join(path, f"{name}.joblib")
            dump(clf, model_path)
            print(f"Saved {name} to {model_path}")

    def load_models(self, path: str) -> None:
        """
        Load models from disk.

        Parameters
        ----------
        path : str
            Directory path containing saved models.
        """
        import os
        from joblib import load

        for filename in os.listdir(path):
            if filename.endswith('.joblib'):
                name = filename.replace('.joblib', '')
                model_path = os.path.join(path, filename)
                self.classifiers_[name] = load(model_path)
                print(f"Loaded {name} from {model_path}")

    def compare_data_versions(
        self,
        X_versions: Dict[str, Union[np.ndarray, pd.DataFrame]],
        y: Union[np.ndarray, pd.Series],
        batch: Optional[Union[np.ndarray, pd.Series]] = None,
        classifier_name: str = 'LogisticRegression',
        scoring: str = 'roc_auc'
    ) -> Dict[str, Any]:
        """
        Compare model performance across different data preprocessing versions.

        This method enables systematic evaluation of how different preprocessing
        strategies (e.g., original, standardized, batch-corrected) affect
        classification performance.

        Parameters
        ----------
        X_versions : dict
            Dictionary mapping version names to feature matrices.
            Example: {'original': X_orig, 'standardized': X_std, 'batch_corrected': X_batch}
        y : array-like of shape (n_samples,)
            Target labels (same for all versions).
        batch : array-like of shape (n_samples,), optional
            Batch labels for batch-aware classification.
        classifier_name : str, default='LogisticRegression'
            Name of classifier to use from DEFAULT_CLASSIFIERS.
        scoring : str, default='roc_auc'
            Scoring metric for evaluation.

        Returns
        -------
        results : dict
            Dictionary containing:
            - 'version_scores': Cross-validation scores for each version
            - 'version_summary': Summary statistics per version
            - 'best_version': Name of the best performing version
            - 'ranking': Versions ranked by mean performance
        """
        from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

        y_array = np.asarray(y)

        # Get classifier config
        if classifier_name not in self.DEFAULT_CLASSIFIERS:
            raise ValueError(
                f"Unknown classifier: {classifier_name}. "
                f"Available: {list(self.DEFAULT_CLASSIFIERS.keys())}"
            )

        clf_config = self.DEFAULT_CLASSIFIERS[classifier_name]

        results = {
            'version_scores': {},
            'version_summary': {},
            'ranking': [],
            'best_version': None
        }

        version_means = []

        # Create repeated stratified CV
        cv = RepeatedStratifiedKFold(
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            random_state=self.random_state
        )

        for version_name, X in X_versions.items():
            print(f"Evaluating version: {version_name}...")

            # Apply feature selection
            X_selected = self._select_features(X, y_array)

            # Create batch-aware classifier
            batch_clf = BatchAwareClassifier(
                base_classifier=clone(clf_config['classifier']),
                include_batch=(batch is not None)
            )

            # Prepare features
            X_prepared = batch_clf._prepare_features(X_selected, batch, fit=True)

            # Perform cross-validation
            cv_scores = cross_val_score(
                clone(clf_config['classifier']),
                X_prepared,
                y_array,
                cv=cv,
                scoring=scoring,
                n_jobs=self.n_jobs
            )

            results['version_scores'][version_name] = cv_scores

            # Calculate summary statistics
            mean_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)

            results['version_summary'][version_name] = {
                'mean': mean_score,
                'std': std_score,
                'min': np.min(cv_scores),
                'max': np.max(cv_scores),
                'median': np.median(cv_scores)
            }

            version_means.append((version_name, mean_score))

        # Rank versions
        version_means.sort(key=lambda x: x[1], reverse=True)
        results['ranking'] = [v[0] for v in version_means]
        results['best_version'] = version_means[0][0] if version_means else None

        self.results_['data_version_comparison'] = results

        return results

    def get_consensus_predictions(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        batch: Optional[Union[np.ndarray, pd.Series]] = None,
        method: str = 'voting',
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Get consensus predictions from all trained models.

        This method aggregates predictions from multiple trained classifiers
        to produce more robust final predictions through voting or weighted averaging.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix for prediction.
        batch : array-like of shape (n_samples,), optional
            Batch labels for batch-aware prediction.
        method : str, default='voting'
            Consensus method. Options:
            - 'voting': Majority voting for class predictions
            - 'soft_voting': Average probability scores then predict
            - 'weighted': Weighted average based on model performance
        weights : dict, optional
            Dictionary mapping classifier names to weights.
            Required if method='weighted'. If None with 'weighted',
            uses uniform weights.

        Returns
        -------
        results : dict
            Dictionary containing:
            - 'consensus_predictions': Final aggregated predictions
            - 'consensus_probabilities': Aggregated probability scores (if available)
            - 'individual_predictions': Predictions from each model
            - 'agreement_scores': Agreement rate between models
        """
        if not self.classifiers_:
            raise ValueError(
                "No fitted classifiers available. Run analysis methods first."
            )

        results = {
            'consensus_predictions': None,
            'consensus_probabilities': None,
            'individual_predictions': {},
            'agreement_scores': {}
        }

        # Collect predictions from all classifiers
        all_predictions = []
        all_probabilities = []
        classifier_names = []

        for name, clf in self.classifiers_.items():
            try:
                pred = clf.predict(X, batch)
                results['individual_predictions'][name] = pred
                all_predictions.append(pred)
                classifier_names.append(name)

                # Try to get probabilities
                if hasattr(clf, 'predict_proba'):
                    try:
                        prob = clf.predict_proba(X, batch)
                        all_probabilities.append(prob)
                    except Exception:
                        pass
            except Exception as e:
                warnings.warn(f"Could not get predictions from {name}: {e}")

        if not all_predictions:
            raise ValueError("No valid predictions from any classifier.")

        all_predictions = np.array(all_predictions)  # Shape: (n_classifiers, n_samples)
        n_classifiers, n_samples = all_predictions.shape

        # Calculate consensus based on method
        if method == 'voting':
            # Majority voting
            from scipy import stats as sp_stats
            consensus = sp_stats.mode(all_predictions, axis=0, keepdims=False)[0]
            results['consensus_predictions'] = consensus

        elif method == 'soft_voting':
            if all_probabilities:
                # Average probability scores
                all_probs = np.array(all_probabilities)  # (n_classifiers, n_samples, n_classes)
                avg_probs = np.mean(all_probs, axis=0)
                results['consensus_probabilities'] = avg_probs
                results['consensus_predictions'] = np.argmax(avg_probs, axis=1)
            else:
                warnings.warn(
                    "No probability scores available. Falling back to voting."
                )
                from scipy import stats as sp_stats
                consensus = sp_stats.mode(all_predictions, axis=0, keepdims=False)[0]
                results['consensus_predictions'] = consensus

        elif method == 'weighted':
            if weights is None:
                # Use uniform weights
                weights = {name: 1.0 / n_classifiers for name in classifier_names}

            # Normalize weights
            weight_sum = sum(weights.get(name, 0) for name in classifier_names)
            if weight_sum == 0:
                weight_sum = 1.0

            if all_probabilities and len(all_probabilities) == n_classifiers:
                # Weighted probability averaging
                all_probs = np.array(all_probabilities)
                weighted_probs = np.zeros_like(all_probs[0])

                for i, name in enumerate(classifier_names):
                    w = weights.get(name, 0) / weight_sum
                    weighted_probs += w * all_probs[i]

                results['consensus_probabilities'] = weighted_probs
                results['consensus_predictions'] = np.argmax(weighted_probs, axis=1)
            else:
                # Weighted voting (count each prediction with its weight)
                unique_classes = np.unique(all_predictions)
                weighted_votes = np.zeros((n_samples, len(unique_classes)))

                for i, name in enumerate(classifier_names):
                    w = weights.get(name, 0) / weight_sum
                    for j, cls in enumerate(unique_classes):
                        weighted_votes[:, j] += w * (all_predictions[i] == cls)

                results['consensus_predictions'] = unique_classes[
                    np.argmax(weighted_votes, axis=1)
                ]
        else:
            raise ValueError(
                f"Unknown consensus method: {method}. "
                f"Choose from: 'voting', 'soft_voting', 'weighted'"
            )

        # Calculate agreement scores
        consensus = results['consensus_predictions']
        for name in classifier_names:
            pred = results['individual_predictions'][name]
            agreement = np.mean(pred == consensus)
            results['agreement_scores'][name] = agreement

        return results
