"""
Machine learning-based feature selection methods.

This module provides ML-based feature selection methods including
LASSO (L1 regularization), Elastic Net (L1+L2), and Random Forest
importance-based selection.

These methods complement statistical approaches by capturing
non-linear relationships and feature interactions.

Example:
    >>> from src.features.ml_selection import (
    ...     LassoFeatureSelector,
    ...     ElasticNetFeatureSelector,
    ...     RandomForestFeatureSelector
    ... )
    >>>
    >>> # LASSO selection with cross-validation
    >>> lasso = LassoFeatureSelector(cv=5)
    >>> lasso.fit(data, labels)
    >>> features = lasso.get_selected_features()
"""

from typing import List, Optional, Dict, Any, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, ElasticNetCV, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import logging

# Configure module logger
logger = logging.getLogger(__name__)


class LassoFeatureSelector:
    """
    LASSO-based feature selector using L1 regularization.

    Uses LASSO regression with cross-validation to automatically
    select the optimal regularization strength and identify features
    with non-zero coefficients.

    L1 regularization encourages sparsity, effectively performing
    embedded feature selection by shrinking irrelevant feature
    coefficients to exactly zero.

    Attributes:
        cv: Number of cross-validation folds.
        n_alphas: Number of alpha values to try.
        max_iter: Maximum iterations for optimization.
        n_jobs: Number of parallel jobs.
        random_state: Random seed for reproducibility.
        model_: Fitted LassoCV model.
        results_: Dictionary with selection results.

    Example:
        >>> from src.features.ml_selection import LassoFeatureSelector
        >>>
        >>> # Initialize with cross-validation
        >>> selector = LassoFeatureSelector(cv=5, n_jobs=4)
        >>>
        >>> # Fit and select features
        >>> selector.fit(methylation_data.T, labels)  # samples x features
        >>> features = selector.get_selected_features()
        >>> print(f"Selected {len(features)} features with non-zero coefficients")
    """

    def __init__(
        self,
        cv: int = 5,
        n_alphas: int = 100,
        max_iter: int = 10000,
        n_jobs: int = -1,
        random_state: int = 42
    ):
        """
        Initialize LassoFeatureSelector.

        Args:
            cv: Number of cross-validation folds for alpha selection.
            n_alphas: Number of alpha values to test.
            max_iter: Maximum iterations for convergence.
            n_jobs: Number of parallel jobs (-1 uses all cores).
            random_state: Random seed for reproducibility.

        Example:
            >>> selector = LassoFeatureSelector(cv=5, n_alphas=100)
        """
        self.cv = cv
        self.n_alphas = n_alphas
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.model_: Optional[LassoCV] = None
        self.scaler_: Optional[StandardScaler] = None
        self.feature_names_: Optional[List[str]] = None
        self.results_: Dict[str, Any] = {}
        self._is_fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: np.ndarray
    ) -> 'LassoFeatureSelector':
        """
        Fit LASSO model with cross-validation.

        Args:
            X: Feature matrix (samples x features). If DataFrame,
               column names are preserved as feature names.
            y: Target labels (continuous or binary).

        Returns:
            Self for method chaining.

        Example:
            >>> selector = LassoFeatureSelector()
            >>> selector.fit(methylation_data.T, labels)
        """
        # Extract feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X_values = X.values
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]
            X_values = X

        y = np.asarray(y)

        logger.info("Fitting LASSO with %d samples and %d features",
                   X_values.shape[0], X_values.shape[1])

        # Standardize features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_values)

        # Fit LassoCV
        self.model_ = LassoCV(
            cv=self.cv,
            n_alphas=self.n_alphas,
            max_iter=self.max_iter,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )
        self.model_.fit(X_scaled, y)

        # Store results
        non_zero_mask = self.model_.coef_ != 0
        self.results_ = {
            'optimal_alpha': self.model_.alpha_,
            'n_selected': non_zero_mask.sum(),
            'coefficients': pd.Series(
                self.model_.coef_,
                index=self.feature_names_
            ),
            'selected_mask': non_zero_mask
        }

        self._is_fitted = True

        logger.info("LASSO fit complete. Optimal alpha=%.6f, selected %d features",
                   self.model_.alpha_, self.results_['n_selected'])

        return self

    def get_selected_features(
        self,
        min_coef: float = 0.0
    ) -> List[str]:
        """
        Get features with non-zero coefficients.

        Args:
            min_coef: Minimum absolute coefficient value for selection.
                     Default 0.0 selects all non-zero coefficients.

        Returns:
            List of selected feature names.

        Raises:
            ValueError: If model has not been fitted.

        Example:
            >>> selector = LassoFeatureSelector()
            >>> selector.fit(X, y)
            >>> features = selector.get_selected_features(min_coef=0.01)
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        coefficients = self.results_['coefficients']
        mask = np.abs(coefficients) > min_coef

        selected = coefficients[mask].index.tolist()

        logger.info("Selected %d features with |coef| > %.4f",
                   len(selected), min_coef)

        return selected

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance based on coefficient magnitude.

        Returns:
            DataFrame with features ranked by absolute coefficient.

        Example:
            >>> selector = LassoFeatureSelector()
            >>> selector.fit(X, y)
            >>> importance = selector.get_feature_importance()
            >>> print(importance.head(10))
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        coefficients = self.results_['coefficients']

        importance_df = pd.DataFrame({
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        })
        importance_df = importance_df.sort_values(
            'abs_coefficient', ascending=False
        )

        return importance_df


class ElasticNetFeatureSelector:
    """
    Elastic Net feature selector using combined L1/L2 regularization.

    Elastic Net combines LASSO (L1) and Ridge (L2) regularization,
    providing a balance between sparsity and handling correlated features.
    This is particularly useful for DNA methylation data where nearby
    CpG sites often show correlation.

    Attributes:
        cv: Number of cross-validation folds.
        l1_ratio: Mixing parameter (0=Ridge, 1=LASSO).
        n_alphas: Number of alpha values to try.
        n_jobs: Number of parallel jobs.
        model_: Fitted ElasticNetCV model.
        results_: Dictionary with selection results.

    Example:
        >>> from src.features.ml_selection import ElasticNetFeatureSelector
        >>>
        >>> # Initialize with L1/L2 ratio optimization
        >>> selector = ElasticNetFeatureSelector(
        ...     cv=5,
        ...     l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99]
        ... )
        >>>
        >>> selector.fit(methylation_data.T, labels)
        >>> features = selector.get_selected_features()
    """

    def __init__(
        self,
        cv: int = 5,
        l1_ratio: Union[float, List[float]] = 0.5,
        n_alphas: int = 100,
        max_iter: int = 10000,
        n_jobs: int = -1,
        random_state: int = 42
    ):
        """
        Initialize ElasticNetFeatureSelector.

        Args:
            cv: Number of cross-validation folds.
            l1_ratio: L1/L2 mixing parameter(s). Can be single value or
                     list for cross-validation selection.
            n_alphas: Number of alpha values to test.
            max_iter: Maximum iterations for convergence.
            n_jobs: Number of parallel jobs.
            random_state: Random seed for reproducibility.

        Example:
            >>> # With fixed L1 ratio
            >>> selector = ElasticNetFeatureSelector(l1_ratio=0.7)
            >>>
            >>> # With L1 ratio selection via CV
            >>> selector = ElasticNetFeatureSelector(
            ...     l1_ratio=[0.1, 0.5, 0.9, 0.99]
            ... )
        """
        self.cv = cv
        self.l1_ratio = l1_ratio
        self.n_alphas = n_alphas
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.model_: Optional[ElasticNetCV] = None
        self.scaler_: Optional[StandardScaler] = None
        self.feature_names_: Optional[List[str]] = None
        self.results_: Dict[str, Any] = {}
        self._is_fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: np.ndarray
    ) -> 'ElasticNetFeatureSelector':
        """
        Fit Elastic Net model with cross-validation.

        Args:
            X: Feature matrix (samples x features).
            y: Target labels.

        Returns:
            Self for method chaining.

        Example:
            >>> selector = ElasticNetFeatureSelector()
            >>> selector.fit(methylation_data.T, labels)
        """
        # Extract feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X_values = X.values
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]
            X_values = X

        y = np.asarray(y)

        logger.info("Fitting Elastic Net with %d samples and %d features",
                   X_values.shape[0], X_values.shape[1])

        # Standardize features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_values)

        # Fit ElasticNetCV
        self.model_ = ElasticNetCV(
            cv=self.cv,
            l1_ratio=self.l1_ratio,
            n_alphas=self.n_alphas,
            max_iter=self.max_iter,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )
        self.model_.fit(X_scaled, y)

        # Store results
        non_zero_mask = self.model_.coef_ != 0
        self.results_ = {
            'optimal_alpha': self.model_.alpha_,
            'optimal_l1_ratio': self.model_.l1_ratio_,
            'n_selected': non_zero_mask.sum(),
            'coefficients': pd.Series(
                self.model_.coef_,
                index=self.feature_names_
            ),
            'selected_mask': non_zero_mask
        }

        self._is_fitted = True

        logger.info(
            "Elastic Net fit complete. alpha=%.6f, l1_ratio=%.2f, selected %d features",
            self.model_.alpha_, self.model_.l1_ratio_, self.results_['n_selected']
        )

        return self

    def get_selected_features(
        self,
        min_coef: float = 0.0
    ) -> List[str]:
        """
        Get features with non-zero coefficients.

        Args:
            min_coef: Minimum absolute coefficient value.

        Returns:
            List of selected feature names.

        Example:
            >>> selector = ElasticNetFeatureSelector()
            >>> selector.fit(X, y)
            >>> features = selector.get_selected_features()
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        coefficients = self.results_['coefficients']
        mask = np.abs(coefficients) > min_coef

        selected = coefficients[mask].index.tolist()

        logger.info("Selected %d features with |coef| > %.4f",
                   len(selected), min_coef)

        return selected

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance based on coefficient magnitude.

        Returns:
            DataFrame with features ranked by absolute coefficient.
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        coefficients = self.results_['coefficients']

        importance_df = pd.DataFrame({
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        })
        importance_df = importance_df.sort_values(
            'abs_coefficient', ascending=False
        )

        return importance_df


class RandomForestFeatureSelector:
    """
    Random Forest-based feature selector using importance scores.

    Uses Random Forest classifier to compute feature importance
    based on mean decrease in impurity (Gini importance) or
    permutation importance.

    Particularly effective for capturing non-linear relationships
    and feature interactions in classification tasks.

    Attributes:
        n_estimators: Number of trees in the forest.
        max_depth: Maximum tree depth.
        n_jobs: Number of parallel jobs.
        random_state: Random seed.
        model_: Fitted RandomForestClassifier.
        results_: Dictionary with selection results.

    Example:
        >>> from src.features.ml_selection import RandomForestFeatureSelector
        >>>
        >>> selector = RandomForestFeatureSelector(n_estimators=500)
        >>> selector.fit(methylation_data.T, labels)
        >>> features = selector.get_top_features(n_features=100)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 5,
        n_jobs: int = -1,
        random_state: int = 42
    ):
        """
        Initialize RandomForestFeatureSelector.

        Args:
            n_estimators: Number of trees in the forest.
            max_depth: Maximum depth of trees. None for unlimited.
            min_samples_leaf: Minimum samples required in a leaf.
            n_jobs: Number of parallel jobs.
            random_state: Random seed for reproducibility.

        Example:
            >>> selector = RandomForestFeatureSelector(
            ...     n_estimators=500,
            ...     max_depth=10
            ... )
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.model_: Optional[RandomForestClassifier] = None
        self.feature_names_: Optional[List[str]] = None
        self.results_: Dict[str, Any] = {}
        self._is_fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: np.ndarray
    ) -> 'RandomForestFeatureSelector':
        """
        Fit Random Forest model.

        Args:
            X: Feature matrix (samples x features).
            y: Target labels (categorical for classification).

        Returns:
            Self for method chaining.

        Example:
            >>> selector = RandomForestFeatureSelector()
            >>> selector.fit(methylation_data.T, labels)
        """
        # Extract feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X_values = X.values
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]
            X_values = X

        y = np.asarray(y)

        logger.info("Fitting Random Forest with %d samples and %d features",
                   X_values.shape[0], X_values.shape[1])

        # Fit Random Forest
        self.model_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )
        self.model_.fit(X_values, y)

        # Store results
        importances = self.model_.feature_importances_
        self.results_ = {
            'importances': pd.Series(
                importances,
                index=self.feature_names_
            ),
            'oob_score': self.model_.oob_score_ if hasattr(self.model_, 'oob_score_') else None
        }

        self._is_fitted = True

        logger.info("Random Forest fit complete. Mean importance: %.6f",
                   importances.mean())

        return self

    def get_top_features(
        self,
        n_features: int = 100
    ) -> List[str]:
        """
        Get top N features by importance.

        Args:
            n_features: Number of top features to return.

        Returns:
            List of top feature names.

        Example:
            >>> selector = RandomForestFeatureSelector()
            >>> selector.fit(X, y)
            >>> top_features = selector.get_top_features(100)
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        importances = self.results_['importances']
        sorted_importances = importances.sort_values(ascending=False)

        top_features = sorted_importances.head(n_features).index.tolist()

        logger.info("Selected top %d features by importance", len(top_features))

        return top_features

    def get_selected_features(
        self,
        importance_threshold: Optional[float] = None,
        percentile: Optional[float] = None
    ) -> List[str]:
        """
        Get features meeting importance threshold.

        Args:
            importance_threshold: Minimum importance value.
            percentile: Select top percentile of features (0-100).
                       If both provided, importance_threshold takes precedence.

        Returns:
            List of selected feature names.

        Example:
            >>> selector = RandomForestFeatureSelector()
            >>> selector.fit(X, y)
            >>>
            >>> # By threshold
            >>> features = selector.get_selected_features(importance_threshold=0.001)
            >>>
            >>> # By percentile
            >>> features = selector.get_selected_features(percentile=10)  # Top 10%
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        importances = self.results_['importances']

        if importance_threshold is not None:
            mask = importances >= importance_threshold
            selected = importances[mask].index.tolist()

        elif percentile is not None:
            threshold = np.percentile(importances, 100 - percentile)
            mask = importances >= threshold
            selected = importances[mask].index.tolist()

        else:
            # Default: features with above-mean importance
            threshold = importances.mean()
            mask = importances >= threshold
            selected = importances[mask].index.tolist()

        logger.info("Selected %d features", len(selected))

        return selected

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get full feature importance DataFrame.

        Returns:
            DataFrame with importance scores sorted by value.

        Example:
            >>> selector = RandomForestFeatureSelector()
            >>> selector.fit(X, y)
            >>> importance = selector.get_feature_importance()
            >>> print(importance.head(20))
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        importance_df = pd.DataFrame({
            'importance': self.results_['importances']
        })
        importance_df = importance_df.sort_values(
            'importance', ascending=False
        )
        importance_df['cumulative_importance'] = (
            importance_df['importance'].cumsum() /
            importance_df['importance'].sum()
        )

        return importance_df

    def compute_permutation_importance(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: np.ndarray,
        n_repeats: int = 10
    ) -> pd.DataFrame:
        """
        Compute permutation importance for more robust estimates.

        Permutation importance measures the decrease in model performance
        when a feature's values are randomly shuffled.

        Args:
            X: Feature matrix (samples x features).
            y: Target labels.
            n_repeats: Number of permutation repeats.

        Returns:
            DataFrame with permutation importance statistics.

        Example:
            >>> selector = RandomForestFeatureSelector()
            >>> selector.fit(X, y)
            >>> perm_importance = selector.compute_permutation_importance(X, y)
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        from sklearn.inspection import permutation_importance

        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X

        result = permutation_importance(
            self.model_, X_values, y,
            n_repeats=n_repeats,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )

        perm_df = pd.DataFrame({
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std
        }, index=self.feature_names_)

        perm_df = perm_df.sort_values('importance_mean', ascending=False)

        logger.info("Computed permutation importance with %d repeats", n_repeats)

        return perm_df
