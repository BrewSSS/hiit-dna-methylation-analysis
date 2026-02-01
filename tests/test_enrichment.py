"""Tests for enrichment analysis module."""
import pytest
import numpy as np
from scipy import stats
from src.enrichment.analysis import (
    FallbackEnrichmentStrategy,
    EnrichmentConfig,
    EnrichmentResult,
)


class TestEnrichmentConfig:
    """Tests for EnrichmentConfig."""

    def test_default_values(self):
        """Default config should have reasonable values."""
        config = EnrichmentConfig()
        assert config.pvalue_cutoff == 0.05
        assert config.min_gene_set_size > 0
        assert config.max_gene_set_size > config.min_gene_set_size
        assert config.background_size > 0


class TestEnrichmentResult:
    """Tests for EnrichmentResult."""

    def test_creation(self):
        """Should create a result container."""
        result = EnrichmentResult(
            term_id="GO:0006915",
            term_name="apoptotic process",
            pvalue=0.001,
            qvalue=0.01,
            overlap_count=5,
            gene_set_size=100,
            overlap_genes=["TP53", "BCL2", "BAX", "CASP3", "CASP9"],
            enrichment_score=2.5,
            category="GO_BP",
        )
        assert result.term_id == "GO:0006915"
        assert len(result.overlap_genes) == 5


class TestFallbackEnrichmentStrategy:
    """Tests for FallbackEnrichmentStrategy."""

    def test_initialization(self):
        """Should initialize with default config."""
        strategy = FallbackEnrichmentStrategy()
        assert strategy.config is not None

    def test_hypergeometric_test(self):
        """Hypergeometric test should return valid p-value."""
        strategy = FallbackEnrichmentStrategy()
        query = {"A", "B", "C", "D", "E"}
        gene_set = {"A", "B", "C", "X", "Y", "Z", "W", "V", "U", "T"}
        background = 1000
        pval = strategy.hypergeometric_test(query, gene_set, background)
        assert 0 <= pval <= 1

    def test_hypergeometric_no_overlap(self):
        """No overlap should give high p-value."""
        strategy = FallbackEnrichmentStrategy()
        query = {"A", "B", "C"}
        gene_set = {"X", "Y", "Z"}
        pval = strategy.hypergeometric_test(query, gene_set, 1000)
        assert pval > 0.5

    def test_hypergeometric_full_overlap(self):
        """Full overlap should give low p-value."""
        strategy = FallbackEnrichmentStrategy()
        query = {"A", "B", "C", "D", "E"}
        gene_set = {"A", "B", "C", "D", "E", "F", "G"}
        pval = strategy.hypergeometric_test(query, gene_set, 20000)
        assert pval < 0.05

    def test_multiple_testing_correction(self):
        """Correction should adjust p-values upward."""
        strategy = FallbackEnrichmentStrategy()
        pvalues = np.array([0.01, 0.03, 0.05, 0.10])
        corrected = strategy.multiple_testing_correction(pvalues, method="fdr_bh")
        assert len(corrected) == len(pvalues)
        assert all(c >= p for c, p in zip(corrected, pvalues))
