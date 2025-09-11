"""
Basic functionality tests for HIIT methylation analysis package.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add the package to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hiit_methylation.data import create_sample_data, preprocess_methylation_data
from hiit_methylation.models import HIITMethylationClassifier, HIITMethylationRegressor
from hiit_methylation.utils import methylation_metrics
from hiit_methylation.utils.metrics import validate_methylation_data


class TestBasicFunctionality(unittest.TestCase):
    """Test basic functionality of the HIIT methylation package."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Generate small test dataset
        self.methylation_data, self.metadata = create_sample_data(
            n_samples=20,
            n_cpg_sites=100,
            hiit_effect_size=0.2,
            random_state=42
        )
    
    def test_data_generation(self):
        """Test synthetic data generation."""
        self.assertEqual(self.methylation_data.shape, (20, 100))
        self.assertEqual(len(self.metadata), 20)
        
        # Check required columns
        self.assertIn('hiit_group', self.metadata.columns)
        self.assertIn('time_point', self.metadata.columns)
        
        # Check value ranges
        self.assertTrue((self.methylation_data >= 0).all().all())
        self.assertTrue((self.methylation_data <= 1).all().all())
    
    def test_preprocessing(self):
        """Test data preprocessing."""
        processed_data = preprocess_methylation_data(
            self.methylation_data,
            normalize_method='logit',
            filter_variance=True
        )
        
        # Should have same number of samples
        self.assertEqual(processed_data.shape[0], self.methylation_data.shape[0])
        
        # Should have processed successfully
        self.assertFalse(processed_data.isnull().any().any())
    
    def test_classification_model(self):
        """Test classification model."""
        # Prepare data
        processed_data = preprocess_methylation_data(self.methylation_data)
        y = (self.metadata['hiit_group'] == 'HIIT').astype(int)
        
        # Create and train classifier
        classifier = HIITMethylationClassifier(
            model_type='random_forest',
            feature_selection='univariate',
            n_features=20,
            random_state=42
        )
        
        classifier.fit(processed_data, y)
        
        # Test predictions
        predictions = classifier.predict(processed_data)
        probabilities = classifier.predict_proba(processed_data)
        
        self.assertEqual(len(predictions), len(y))
        self.assertEqual(probabilities.shape, (len(y), 2))
        
        # Test feature importance
        importance = classifier.get_feature_importance()
        self.assertIsInstance(importance, pd.Series)
        self.assertEqual(len(importance), 20)  # n_features
    
    def test_regression_model(self):
        """Test regression model."""
        # Prepare data
        processed_data = preprocess_methylation_data(self.methylation_data)
        y = np.random.normal(0, 0.1, len(processed_data))  # Synthetic target
        
        # Create and train regressor
        regressor = HIITMethylationRegressor(
            model_type='random_forest',
            feature_selection='univariate',
            n_features=20,
            random_state=42
        )
        
        regressor.fit(processed_data, y)
        
        # Test predictions
        predictions = regressor.predict(processed_data)
        
        self.assertEqual(len(predictions), len(y))
        
        # Test feature importance
        importance = regressor.get_feature_importance()
        self.assertIsInstance(importance, pd.Series)
    
    def test_metrics(self):
        """Test evaluation metrics."""
        # Classification metrics
        y_true_class = np.array([0, 1, 1, 0, 1])
        y_pred_class = np.array([0, 1, 0, 0, 1])
        
        class_metrics = methylation_metrics(
            y_true_class, y_pred_class, task_type='classification'
        )
        
        self.assertIn('accuracy', class_metrics)
        self.assertIn('f1_score', class_metrics)
        
        # Regression metrics
        y_true_reg = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred_reg = np.array([1.1, 2.1, 2.9, 3.9, 5.1])
        
        reg_metrics = methylation_metrics(
            y_true_reg, y_pred_reg, task_type='regression'
        )
        
        self.assertIn('r2_score', reg_metrics)
        self.assertIn('mse', reg_metrics)
        self.assertIn('rmse', reg_metrics)
    
    def test_data_validation(self):
        """Test data validation functions."""
        validation_results = validate_methylation_data(self.methylation_data)
        
        self.assertIn('shape', validation_results)
        self.assertIn('missing_values', validation_results)
        self.assertIn('values_in_range', validation_results)
        
        # Should pass validation for synthetic data
        self.assertTrue(validation_results['values_in_range'])
        self.assertEqual(validation_results['shape'], self.methylation_data.shape)


class TestPackageIntegration(unittest.TestCase):
    """Test package integration and imports."""
    
    def test_imports(self):
        """Test that all main components can be imported."""
        try:
            from hiit_methylation import (
                load_methylation_data,
                HIITMethylationClassifier,
                HIITMethylationRegressor,
                plot_methylation_patterns,
                methylation_metrics
            )
        except ImportError as e:
            self.fail(f"Import failed: {e}")
    
    def test_package_metadata(self):
        """Test package metadata."""
        import hiit_methylation
        
        self.assertTrue(hasattr(hiit_methylation, '__version__'))
        self.assertTrue(hasattr(hiit_methylation, '__author__'))


if __name__ == '__main__':
    unittest.main()