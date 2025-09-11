"""
Regression models for HIIT DNA methylation analysis.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from .base import BaseHIITModel


class HIITMethylationRegressor(BaseHIITModel):
    """
    Regression model for HIIT methylation analysis.
    
    This model can predict:
    - Continuous methylation changes after HIIT
    - Fitness improvement scores
    - Physiological response measures
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
        Initialize HIIT methylation regressor.
        
        Parameters:
        -----------
        model_type : str
            Type of regressor ('random_forest', 'svm', 'elastic_net', 'ridge')
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
            Additional parameters for the regressor
        """
        super().__init__(feature_selection, n_features, cv_folds, random_state)
        
        self.model_type = model_type
        self.hyperparameter_tuning = hyperparameter_tuning
        self.model_params = model_params
        self._estimator_type = 'regressor'
        
        self.best_params_ = None
        self.cv_results_ = None
    
    def _create_model(self):
        """Create the underlying regressor."""
        
        if self.model_type == 'random_forest':
            default_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': self.random_state
            }
            default_params.update(self.model_params)
            model = RandomForestRegressor(**default_params)
            
        elif self.model_type == 'svm':
            default_params = {
                'kernel': 'rbf',
                'C': 1.0,
                'gamma': 'scale',
                'epsilon': 0.1
            }
            default_params.update(self.model_params)
            model = SVR(**default_params)
            
        elif self.model_type == 'elastic_net':
            default_params = {
                'alpha': 1.0,
                'l1_ratio': 0.5,
                'max_iter': 1000,
                'random_state': self.random_state
            }
            default_params.update(self.model_params)
            model = ElasticNet(**default_params)
            
        elif self.model_type == 'ridge':
            default_params = {
                'alpha': 1.0,
                'random_state': self.random_state
            }
            default_params.update(self.model_params)
            model = Ridge(**default_params)
            
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
                'epsilon': [0.01, 0.1, 0.2, 0.5]
            },
            'elastic_net': {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            },
            'ridge': {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
            }
        }
        
        param_grid = param_grids.get(self.model_type, {})
        
        if param_grid:
            grid_search = GridSearchCV(
                model, 
                param_grid, 
                cv=self.cv_folds,
                scoring='r2',
                n_jobs=-1
            )
            return grid_search
        
        return model
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'HIITMethylationRegressor':
        """Fit the regressor."""
        
        result = super().fit(X, y)
        
        # Store best parameters if hyperparameter tuning was used
        if self.hyperparameter_tuning and hasattr(self.model_, 'best_params_'):
            self.best_params_ = self.model_.best_params_
            self.cv_results_ = self.model_.cv_results_
            
        return result
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate the regressor performance.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Test methylation data
        y : pd.Series
            True values
            
        Returns:
        --------
        Dict[str, float]
            Performance metrics
        """
        y_pred = self.predict(X)
        
        return {
            'r2_score': r2_score(y, y_pred),
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred)
        }
    
    def predict_methylation_change(
        self,
        baseline_methylation: pd.DataFrame,
        clinical_features: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """
        Predict methylation changes based on baseline data and clinical features.
        
        Parameters:
        -----------
        baseline_methylation : pd.DataFrame
            Baseline methylation data
        clinical_features : pd.DataFrame, optional
            Clinical/demographic features (age, BMI, etc.)
            
        Returns:
        --------
        np.ndarray
            Predicted methylation changes
        """
        
        # Combine methylation and clinical features if provided
        if clinical_features is not None:
            # Ensure same sample order
            common_samples = baseline_methylation.index.intersection(clinical_features.index)
            X_combined = pd.concat([
                baseline_methylation.loc[common_samples],
                clinical_features.loc[common_samples]
            ], axis=1)
        else:
            X_combined = baseline_methylation
        
        return self.predict(X_combined)
    
    def predict_fitness_improvement(
        self,
        methylation_changes: pd.DataFrame,
        baseline_fitness: Optional[pd.Series] = None
    ) -> np.ndarray:
        """
        Predict fitness improvement based on methylation changes.
        
        Parameters:
        -----------
        methylation_changes : pd.DataFrame
            Methylation changes (post - pre)
        baseline_fitness : pd.Series, optional
            Baseline fitness scores
            
        Returns:
        --------
        np.ndarray
            Predicted fitness improvement
        """
        
        if baseline_fitness is not None:
            # Include baseline fitness as a feature
            X_combined = methylation_changes.copy()
            X_combined['baseline_fitness'] = baseline_fitness
            return self.predict(X_combined)
        else:
            return self.predict(methylation_changes)
    
    def get_top_predictive_cpg_sites(self, top_n: int = 20) -> pd.Series:
        """
        Get the most predictive CpG sites.
        
        Parameters:
        -----------
        top_n : int
            Number of top CpG sites to return
            
        Returns:
        --------
        pd.Series
            Top predictive CpG sites with their importance scores
        """
        importance = self.get_feature_importance()
        if importance is not None:
            return importance.head(top_n)
        else:
            raise ValueError("Feature importance not available for this model type")