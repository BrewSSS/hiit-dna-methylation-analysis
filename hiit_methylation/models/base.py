"""
Base class for HIIT DNA methylation analysis models.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report


class BaseHIITModel(BaseEstimator, ABC):
    """
    Base class for HIIT DNA methylation analysis models.
    """
    
    def __init__(
        self,
        feature_selection: Optional[str] = None,
        n_features: Optional[int] = None,
        cv_folds: int = 5,
        random_state: Optional[int] = 42
    ):
        """
        Initialize base HIIT model.
        
        Parameters:
        -----------
        feature_selection : str, optional
            Feature selection method ('variance', 'univariate', 'lasso')
        n_features : int, optional
            Number of features to select
        cv_folds : int
            Number of cross-validation folds
        random_state : int, optional
            Random seed for reproducibility
        """
        self.feature_selection = feature_selection
        self.n_features = n_features
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        self.model_ = None
        self.feature_selector_ = None
        self.selected_features_ = None
        self.is_fitted_ = False
    
    @abstractmethod
    def _create_model(self) -> BaseEstimator:
        """Create the underlying scikit-learn model."""
        pass
    
    def _select_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Apply feature selection if specified.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
            
        Returns:
        --------
        pd.DataFrame
            Selected features
        """
        if self.feature_selection is None:
            self.selected_features_ = X.columns.tolist()
            return X
        
        from sklearn.feature_selection import (
            VarianceThreshold, 
            SelectKBest, 
            f_classif, 
            f_regression
        )
        from sklearn.linear_model import LassoCV
        
        if self.feature_selection == 'variance':
            selector = VarianceThreshold()
            X_selected = selector.fit_transform(X)
            selected_mask = selector.get_support()
            
        elif self.feature_selection == 'univariate':
            # Choose scoring function based on task type
            if hasattr(self, '_estimator_type') and self._estimator_type == 'classifier':
                score_func = f_classif
            else:
                score_func = f_regression
                
            n_features = self.n_features or min(1000, X.shape[1] // 2)
            selector = SelectKBest(score_func=score_func, k=n_features)
            X_selected = selector.fit_transform(X, y)
            selected_mask = selector.get_support()
            
        elif self.feature_selection == 'lasso':
            # Use Lasso for feature selection
            lasso = LassoCV(cv=self.cv_folds, random_state=self.random_state)
            lasso.fit(X, y)
            selected_mask = np.abs(lasso.coef_) > 0
            X_selected = X.loc[:, selected_mask]
            
        else:
            raise ValueError(f"Unknown feature selection method: {self.feature_selection}")
        
        self.feature_selector_ = selector if self.feature_selection != 'lasso' else lasso
        self.selected_features_ = X.columns[selected_mask].tolist()
        
        print(f"Selected {len(self.selected_features_)} features from {X.shape[1]}")
        
        return pd.DataFrame(X_selected, index=X.index, columns=self.selected_features_)
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseHIITModel':
        """
        Fit the HIIT methylation model.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Methylation data (samples x CpG sites)
        y : pd.Series
            Target variable
            
        Returns:
        --------
        BaseHIITModel
            Fitted model
        """
        # Feature selection
        X_selected = self._select_features(X, y)
        
        # Create and fit model
        self.model_ = self._create_model()
        self.model_.fit(X_selected, y)
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the fitted model.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Methylation data
            
        Returns:
        --------
        np.ndarray
            Predictions
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        # Apply same feature selection
        if self.feature_selection:
            X_selected = X[self.selected_features_]
        else:
            X_selected = X
            
        return self.model_.predict(X_selected)
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Perform cross-validation on the model.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Methylation data
        y : pd.Series
            Target variable
            
        Returns:
        --------
        Dict[str, float]
            Cross-validation scores
        """
        # Feature selection
        X_selected = self._select_features(X, y)
        
        # Create model
        model = self._create_model()
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, X_selected, y, 
            cv=self.cv_folds, 
            scoring=None  # Use default scoring
        )
        
        return {
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'scores': cv_scores
        }
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """
        Get feature importance if available.
        
        Returns:
        --------
        pd.Series, optional
            Feature importance scores
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted first")
        
        if hasattr(self.model_, 'feature_importances_'):
            return pd.Series(
                self.model_.feature_importances_,
                index=self.selected_features_
            ).sort_values(ascending=False)
        elif hasattr(self.model_, 'coef_'):
            return pd.Series(
                np.abs(self.model_.coef_).flatten(),
                index=self.selected_features_
            ).sort_values(ascending=False)
        else:
            return None