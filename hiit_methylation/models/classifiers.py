"""
Classification models for HIIT DNA methylation analysis.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

from .base import BaseHIITModel


class HIITMethylationClassifier(BaseHIITModel):
    """
    Classification model for HIIT intervention response prediction.
    
    This model can predict:
    - HIIT vs Control group classification
    - Pre vs Post intervention timepoint
    - Response vs Non-response to HIIT
    """
    
    def __init__(
        self,
        model_type: str = 'random_forest',
        hyperparameter_tuning: bool = False,
        feature_selection: Optional[str] = 'univariate',
        n_features: Optional[int] = 1000,
        cv_folds: int = 5,
        random_state: Optional[int] = 42,
        **model_params
    ):
        """
        Initialize HIIT methylation classifier.
        
        Parameters:
        -----------
        model_type : str
            Type of classifier ('random_forest', 'svm', 'logistic')
        hyperparameter_tuning : bool
            Whether to perform hyperparameter tuning
        feature_selection : str, optional
            Feature selection method
        n_features : int, optional
            Number of features to select
        cv_folds : int
            Number of CV folds
        random_state : int, optional
            Random seed
        **model_params
            Additional parameters for the classifier
        """
        super().__init__(feature_selection, n_features, cv_folds, random_state)
        
        self.model_type = model_type
        self.hyperparameter_tuning = hyperparameter_tuning
        self.model_params = model_params
        self._estimator_type = 'classifier'
        
        self.best_params_ = None
        self.cv_results_ = None
    
    def _create_model(self):
        """Create the underlying classifier."""
        
        if self.model_type == 'random_forest':
            default_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': self.random_state
            }
            default_params.update(self.model_params)
            model = RandomForestClassifier(**default_params)
            
        elif self.model_type == 'svm':
            default_params = {
                'kernel': 'rbf',
                'C': 1.0,
                'gamma': 'scale',
                'random_state': self.random_state
            }
            default_params.update(self.model_params)
            model = SVC(**default_params, probability=True)
            
        elif self.model_type == 'logistic':
            default_params = {
                'C': 1.0,
                'penalty': 'l2',
                'max_iter': 1000,
                'random_state': self.random_state
            }
            default_params.update(self.model_params)
            model = LogisticRegression(**default_params)
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Hyperparameter tuning
        if self.hyperparameter_tuning:
            model = self._tune_hyperparameters(model)
            
        return model
    
    def _tune_hyperparameters(self, model):
        """Perform hyperparameter tuning."""
        
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'svm': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'poly', 'linear']
            },
            'logistic': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'lbfgs', 'saga']
            }
        }
        
        param_grid = param_grids.get(self.model_type, {})
        
        if param_grid:
            grid_search = GridSearchCV(
                model, 
                param_grid, 
                cv=self.cv_folds,
                scoring='roc_auc',
                n_jobs=-1,
                random_state=self.random_state
            )
            return grid_search
        
        return model
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'HIITMethylationClassifier':
        """Fit the classifier."""
        
        result = super().fit(X, y)
        
        # Store best parameters if hyperparameter tuning was used
        if self.hyperparameter_tuning and hasattr(self.model_, 'best_params_'):
            self.best_params_ = self.model_.best_params_
            self.cv_results_ = self.model_.cv_results_
            
        return result
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Methylation data
            
        Returns:
        --------
        np.ndarray
            Class probabilities
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        # Apply same feature selection
        if self.feature_selection:
            X_selected = X[self.selected_features_]
        else:
            X_selected = X
            
        return self.model_.predict_proba(X_selected)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate the classifier performance.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Test methylation data
        y : pd.Series
            True labels
            
        Returns:
        --------
        Dict[str, float]
            Performance metrics
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
        }
        
        # ROC AUC for binary classification
        if len(np.unique(y)) == 2:
            metrics['roc_auc'] = roc_auc_score(y, y_proba[:, 1])
        
        return metrics
    
    def get_classification_report(self, X: pd.DataFrame, y: pd.Series) -> str:
        """Get detailed classification report."""
        y_pred = self.predict(X)
        return classification_report(y, y_pred)
    
    def predict_hiit_response(
        self, 
        pre_methylation: pd.DataFrame,
        post_methylation: pd.DataFrame
    ) -> np.ndarray:
        """
        Predict HIIT response based on pre/post methylation changes.
        
        Parameters:
        -----------
        pre_methylation : pd.DataFrame
            Pre-intervention methylation data
        post_methylation : pd.DataFrame
            Post-intervention methylation data
            
        Returns:
        --------
        np.ndarray
            Predicted response (1 = responder, 0 = non-responder)
        """
        # Calculate methylation changes
        delta_methylation = post_methylation - pre_methylation
        
        # Use the fitted model to predict response
        return self.predict(delta_methylation)