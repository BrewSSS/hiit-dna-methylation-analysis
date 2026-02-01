"""Shared pytest fixtures for HIIT methylation tests."""
import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def mock_methylation_data():
    """Create mock methylation beta-value data (20 samples x 100 CpGs)."""
    np.random.seed(42)
    n_samples, n_features = 20, 100
    data = pd.DataFrame(
        np.random.beta(2, 5, size=(n_samples, n_features)),
        columns=[f"cg{i:08d}" for i in range(n_features)],
        index=[f"GSM{i:07d}" for i in range(n_samples)],
    )
    return data


@pytest.fixture
def mock_binary_labels():
    """Create mock binary labels (HIIT vs Control)."""
    return pd.Series(
        ["HIIT"] * 12 + ["Control"] * 8,
        index=[f"GSM{i:07d}" for i in range(20)],
        name="group",
    )


@pytest.fixture
def mock_multiclass_labels():
    """Create mock multiclass labels (4W/8W/12W/Control)."""
    return pd.Series(
        ["4W"] * 4 + ["8W"] * 4 + ["12W"] * 4 + ["Control"] * 8,
        index=[f"GSM{i:07d}" for i in range(20)],
        name="duration",
    )


@pytest.fixture
def mock_batch_info():
    """Create mock batch information."""
    return pd.Series(
        ["Batch1"] * 10 + ["Batch2"] * 10,
        index=[f"GSM{i:07d}" for i in range(20)],
        name="batch",
    )


@pytest.fixture
def mock_timepoints():
    """Create mock timepoint information."""
    return pd.Series(
        [0, 4, 8, 12] * 5,
        index=[f"GSM{i:07d}" for i in range(20)],
        name="timepoint",
    )


@pytest.fixture
def mock_gene_list():
    """Create mock gene list for enrichment testing."""
    return ["BRCA1", "TP53", "EGFR", "VEGFA", "TNF", "IL6", "MAPK1",
            "AKT1", "PIK3CA", "MTOR", "PPARG", "NR3C1", "HDAC1"]


@pytest.fixture
def mock_pvalues():
    """Create mock p-values array."""
    np.random.seed(42)
    return np.concatenate([
        np.random.uniform(0.001, 0.01, 10),
        np.random.uniform(0.01, 0.05, 20),
        np.random.uniform(0.05, 1.0, 70),
    ])


@pytest.fixture
def mock_effect_sizes():
    """Create mock effect sizes array."""
    np.random.seed(42)
    return np.random.normal(0, 0.5, 100)
