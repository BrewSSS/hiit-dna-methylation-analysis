"""Tests for batch-aware classifiers."""
import pytest
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.models.classifiers import BatchAwareClassifier, ClassifierConfig


class TestClassifierConfig:
    """Tests for ClassifierConfig."""

    def test_default_config(self):
        """Default config should have reasonable values."""
        config = ClassifierConfig()
        assert config.classifier_type in [
            "logistic_regression", "random_forest", "svm"
        ]
        assert config.random_state == 42
        assert isinstance(config.include_batch, bool)

    def test_build_classifier(self):
        """Config should build a valid sklearn classifier."""
        config = ClassifierConfig(classifier_type="logistic_regression")
        clf = config.build_classifier()
        assert hasattr(clf, "fit")
        assert hasattr(clf, "predict")


class TestBatchAwareClassifier:
    """Tests for BatchAwareClassifier."""

    def test_fit_predict(self, mock_methylation_data, mock_binary_labels):
        """Classifier should fit and predict."""
        le = LabelEncoder()
        y = le.fit_transform(mock_binary_labels)
        clf = BatchAwareClassifier(base_classifier="logistic_regression")
        clf.fit(mock_methylation_data, y)
        preds = clf.predict(mock_methylation_data)
        assert len(preds) == len(y)
        assert set(preds).issubset({0, 1})

    def test_with_batch_covariates(
        self, mock_methylation_data, mock_binary_labels, mock_batch_info
    ):
        """Classifier should work with batch covariates."""
        le = LabelEncoder()
        y = le.fit_transform(mock_binary_labels)
        clf = BatchAwareClassifier(
            base_classifier="logistic_regression", include_batch=True
        )
        clf.fit(mock_methylation_data, y, batch=mock_batch_info)
        preds = clf.predict(mock_methylation_data, batch=mock_batch_info)
        assert len(preds) == len(y)

    def test_predict_proba(self, mock_methylation_data, mock_binary_labels):
        """Should return probability estimates."""
        le = LabelEncoder()
        y = le.fit_transform(mock_binary_labels)
        clf = BatchAwareClassifier(base_classifier="logistic_regression")
        clf.fit(mock_methylation_data, y)
        proba = clf.predict_proba(mock_methylation_data)
        assert proba.shape[0] == len(y)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_feature_importance(self, mock_methylation_data, mock_binary_labels):
        """Should extract feature importance after fitting."""
        le = LabelEncoder()
        y = le.fit_transform(mock_binary_labels)
        clf = BatchAwareClassifier(base_classifier="random_forest")
        clf.fit(mock_methylation_data, y)
        importance = clf.get_feature_importance()
        assert importance is not None
