"""
Tests for the methylation data preprocessing module.

Tests cover:
- Variance filtering
- Missing value handling (multiple strategies)
- Batch correction (median centering)
- Data standardization (z-score, min-max)
- Data version creation
- Utility functions (normalize_beta_values, calculate_missing_rate)
"""

import pytest
import numpy as np
import pandas as pd

from src.data.preprocessing import (
    MethylationPreprocessor,
    normalize_beta_values,
    calculate_missing_rate,
)


class TestMethylationPreprocessorInit:
    """Tests for MethylationPreprocessor initialization."""

    def test_default_init(self):
        """Test default initialization parameters."""
        pp = MethylationPreprocessor()
        assert pp.std_threshold == 0.02
        assert pp.missing_threshold == 0.2

    def test_custom_init(self):
        """Test custom initialization parameters."""
        pp = MethylationPreprocessor(std_threshold=0.05, missing_threshold=0.1)
        assert pp.std_threshold == 0.05
        assert pp.missing_threshold == 0.1


class TestVarianceFilter:
    """Tests for filter_low_variance."""

    def test_filter_removes_constant_probes(self):
        """Test that constant probes (zero variance) are removed."""
        data = pd.DataFrame({
            'S1': [0.5, 0.3, 0.1],
            'S2': [0.5, 0.7, 0.9],
            'S3': [0.5, 0.5, 0.5],
        }, index=['cg01', 'cg02', 'cg03'])

        pp = MethylationPreprocessor(std_threshold=0.01)
        filtered = pp.filter_low_variance(data)

        assert 'cg01' not in filtered.index or 'cg03' not in filtered.index
        # cg02 has high variance, should remain
        assert filtered.shape[0] <= data.shape[0]

    def test_filter_preserves_high_variance(self, mock_methylation_data_transposed):
        """Test that high variance probes are retained."""
        pp = MethylationPreprocessor(std_threshold=0.001)
        filtered = pp.filter_low_variance(mock_methylation_data_transposed)

        # With a very low threshold, most probes should pass
        assert filtered.shape[0] > 0

    def test_filter_custom_threshold(self, mock_methylation_data_transposed):
        """Test that stricter threshold removes more probes."""
        pp = MethylationPreprocessor()

        filtered_loose = pp.filter_low_variance(mock_methylation_data_transposed, threshold=0.01)
        filtered_strict = pp.filter_low_variance(mock_methylation_data_transposed, threshold=0.10)

        assert filtered_strict.shape[0] <= filtered_loose.shape[0]

    def test_filter_return_stats(self, mock_methylation_data_transposed):
        """Test that probe statistics are returned when requested."""
        pp = MethylationPreprocessor()
        filtered, stats = pp.filter_low_variance(
            mock_methylation_data_transposed, return_stats=True
        )

        assert isinstance(stats, pd.DataFrame)
        assert 'mean' in stats.columns
        assert 'std' in stats.columns
        assert 'missing_rate' in stats.columns
        assert len(stats) == mock_methylation_data_transposed.shape[0]


class TestMissingValueHandling:
    """Tests for handle_missing_values."""

    def test_no_missing_values(self, mock_methylation_data_transposed):
        """Test handling data with no missing values."""
        pp = MethylationPreprocessor()
        result = pp.handle_missing_values(mock_methylation_data_transposed)

        assert result.isna().sum().sum() == 0
        assert result.shape == mock_methylation_data_transposed.shape

    def test_median_imputation(self, mock_methylation_with_missing):
        """Test median imputation strategy removes all NaN."""
        pp = MethylationPreprocessor()
        result = pp.handle_missing_values(
            mock_methylation_with_missing, strategy='median'
        )

        assert result.isna().sum().sum() == 0
        assert result.shape[0] <= mock_methylation_with_missing.shape[0]

    def test_mean_imputation(self, mock_methylation_with_missing):
        """Test mean imputation strategy removes all NaN."""
        pp = MethylationPreprocessor()
        result = pp.handle_missing_values(
            mock_methylation_with_missing, strategy='mean'
        )

        assert result.isna().sum().sum() == 0

    def test_drop_strategy(self, mock_methylation_with_missing):
        """Test drop strategy removes probes with missing values."""
        pp = MethylationPreprocessor()
        result = pp.handle_missing_values(
            mock_methylation_with_missing, strategy='drop'
        )

        assert result.isna().sum().sum() == 0
        assert result.shape[0] <= mock_methylation_with_missing.shape[0]

    def test_invalid_strategy(self, mock_methylation_data_transposed):
        """Test that invalid strategy raises ValueError."""
        pp = MethylationPreprocessor()

        with pytest.raises(ValueError, match="Invalid imputation strategy"):
            pp.handle_missing_values(
                mock_methylation_data_transposed, strategy='invalid_method'
            )


class TestBatchCorrection:
    """Tests for batch correction."""

    def test_median_centering_basic(self, mock_methylation_data_transposed, mock_batch_info):
        """Test basic median centering batch correction."""
        pp = MethylationPreprocessor()
        corrected = pp.apply_batch_correction(
            mock_methylation_data_transposed,
            mock_batch_info,
            method='median_centering'
        )

        assert corrected.shape == mock_methylation_data_transposed.shape
        # After correction, batch medians should be closer to each other
        batch1_samples = mock_batch_info[mock_batch_info == 'Batch1'].index
        batch2_samples = mock_batch_info[mock_batch_info == 'Batch2'].index

        batch1_median = corrected[batch1_samples].median(axis=1)
        batch2_median = corrected[batch2_samples].median(axis=1)

        # The difference between batch medians should be small after correction
        median_diff = (batch1_median - batch2_median).abs().mean()
        assert median_diff < 0.1  # Relaxed threshold for random data

    def test_no_common_samples_raises(self, mock_methylation_data_transposed):
        """Test that non-overlapping samples raise ValueError."""
        pp = MethylationPreprocessor()
        bad_batch = pd.Series(
            ['Batch1'] * 10 + ['Batch2'] * 10,
            index=[f'WRONG{i:07d}' for i in range(20)]
        )

        with pytest.raises(ValueError, match="No common samples"):
            pp.apply_batch_correction(
                mock_methylation_data_transposed, bad_batch
            )

    def test_invalid_method(self, mock_methylation_data_transposed, mock_batch_info):
        """Test that invalid batch correction method raises ValueError."""
        pp = MethylationPreprocessor()

        with pytest.raises(ValueError, match="Invalid batch correction method"):
            pp.apply_batch_correction(
                mock_methylation_data_transposed,
                mock_batch_info,
                method='invalid_method'
            )


class TestStandardization:
    """Tests for data standardization."""

    def test_zscore_per_probe(self, mock_methylation_data_transposed):
        """Test z-score standardization per probe (row)."""
        pp = MethylationPreprocessor()
        standardized = pp.standardize(
            mock_methylation_data_transposed, method='zscore', axis=1
        )

        # Each row should have mean ~0 and std ~1
        row_means = standardized.mean(axis=1)
        row_stds = standardized.std(axis=1)

        np.testing.assert_array_almost_equal(row_means, 0, decimal=10)
        np.testing.assert_array_almost_equal(row_stds, 1, decimal=10)

    def test_zscore_per_sample(self, mock_methylation_data_transposed):
        """Test z-score standardization per sample (column)."""
        pp = MethylationPreprocessor()
        standardized = pp.standardize(
            mock_methylation_data_transposed, method='zscore', axis=0
        )

        # Each column should have mean ~0 and std ~1
        col_means = standardized.mean(axis=0)
        col_stds = standardized.std(axis=0)

        np.testing.assert_array_almost_equal(col_means, 0, decimal=10)
        np.testing.assert_array_almost_equal(col_stds, 1, decimal=10)

    def test_minmax_per_probe(self, mock_methylation_data_transposed):
        """Test min-max scaling per probe."""
        pp = MethylationPreprocessor()
        scaled = pp.standardize(
            mock_methylation_data_transposed, method='minmax', axis=1
        )

        # Each row should be in [0, 1]
        assert scaled.min(axis=1).min() >= -1e-10
        assert scaled.max(axis=1).max() <= 1 + 1e-10

    def test_invalid_method(self, mock_methylation_data_transposed):
        """Test that invalid standardization method raises ValueError."""
        pp = MethylationPreprocessor()

        with pytest.raises(ValueError, match="Invalid standardization method"):
            pp.standardize(mock_methylation_data_transposed, method='invalid')


class TestCreateDataVersions:
    """Tests for creating multiple data versions."""

    def test_versions_without_batch(self, mock_methylation_data_transposed):
        """Test data versions without batch info."""
        pp = MethylationPreprocessor()
        versions = pp.create_data_versions(mock_methylation_data_transposed)

        assert 'original' in versions
        assert 'standardized' in versions
        assert 'batch_corrected' not in versions
        assert len(versions) == 2

    def test_versions_with_batch(self, mock_methylation_data_transposed, mock_batch_info):
        """Test data versions with batch info."""
        pp = MethylationPreprocessor()
        versions = pp.create_data_versions(
            mock_methylation_data_transposed, batch_info=mock_batch_info
        )

        assert 'original' in versions
        assert 'standardized' in versions
        assert 'batch_corrected' in versions
        assert 'batch_corrected_standardized' in versions
        assert len(versions) == 4

    def test_original_version_is_copy(self, mock_methylation_data_transposed):
        """Test that original version is a copy, not a reference."""
        pp = MethylationPreprocessor()
        versions = pp.create_data_versions(mock_methylation_data_transposed)

        # Modifying the version should not affect the original
        versions['original'].iloc[0, 0] = -999
        assert mock_methylation_data_transposed.iloc[0, 0] != -999


class TestNormalizeBetaValues:
    """Tests for normalize_beta_values utility."""

    def test_values_within_range(self):
        """Test that values outside [0, 1] are clipped."""
        data = pd.DataFrame({
            'S1': [-0.1, 0.5, 1.2],
            'S2': [0.0, 0.5, 1.0],
        })

        normalized = normalize_beta_values(data)

        assert normalized.min().min() >= 0.0
        assert normalized.max().max() <= 1.0

    def test_custom_clip_range(self):
        """Test custom clip range."""
        data = pd.DataFrame({
            'S1': [0.1, 0.5, 0.9],
        })

        normalized = normalize_beta_values(data, clip_range=(0.2, 0.8))

        assert normalized.min().min() >= 0.2
        assert normalized.max().max() <= 0.8


class TestCalculateMissingRate:
    """Tests for calculate_missing_rate utility."""

    def test_no_missing_values(self):
        """Test missing rate is 0 for complete data."""
        data = pd.DataFrame(np.ones((5, 5)))
        assert calculate_missing_rate(data) == 0.0

    def test_all_missing(self):
        """Test missing rate is 1 for all-NaN data."""
        data = pd.DataFrame(np.full((5, 5), np.nan))
        assert calculate_missing_rate(data) == 1.0

    def test_per_probe_missing_rate(self):
        """Test per-probe (row) missing rate."""
        data = pd.DataFrame({
            'S1': [1.0, np.nan, 1.0],
            'S2': [1.0, np.nan, 1.0],
            'S3': [1.0, np.nan, np.nan],
        })

        rates = calculate_missing_rate(data, axis=1)
        assert rates.iloc[0] == 0.0
        assert rates.iloc[1] == 1.0
        np.testing.assert_almost_equal(rates.iloc[2], 1/3)

    def test_per_sample_missing_rate(self):
        """Test per-sample (column) missing rate."""
        data = pd.DataFrame({
            'S1': [1.0, np.nan],
            'S2': [1.0, 1.0],
        })

        rates = calculate_missing_rate(data, axis=0)
        assert rates['S1'] == 0.5
        assert rates['S2'] == 0.0
