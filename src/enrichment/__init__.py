"""
Enrichment analysis module with fallback methodology.

This module provides robust enrichment analysis capabilities for DNA methylation
biomarker studies. It implements a fallback methodology that automatically switches
to MSigDB gene sets with hypergeometric testing when standard annotation databases
(such as org.Hs.eg.db via R/clusterProfiler) are unavailable.

Key Components:
    - FallbackEnrichmentStrategy: Core innovation implementing automatic fallback
    - EnrichmentAnalyzer: Complete enrichment pipeline for methylation biomarkers
    - GOAnnotator: Gene Ontology annotation interface
    - KEGGPathwayMapper: KEGG pathway mapping interface
    - MSigDBLoader: MSigDB gene set loader for fallback enrichment

Example Usage:
    >>> from src.enrichment import EnrichmentAnalyzer, FallbackEnrichmentStrategy
    >>>
    >>> # Initialize analyzer
    >>> analyzer = EnrichmentAnalyzer()
    >>>
    >>> # Map CpG sites to genes and run enrichment
    >>> results = analyzer.run_comprehensive_analysis(cpg_sites)
    >>>
    >>> # Or use fallback strategy directly
    >>> strategy = FallbackEnrichmentStrategy()
    >>> enrichment_results = strategy.run_analysis(gene_list)
"""

from .analysis import (
    EnrichmentAnalyzer,
    EnrichmentConfig,
    EnrichmentResult,
    FallbackEnrichmentStrategy,
    run_enrichment_analysis
)
from .databases import (
    GOAnnotator,
    KEGGPathwayMapper,
    MSigDBLoader,
    EPICAnnotationMapper
)

__all__ = [
    'EnrichmentAnalyzer',
    'EnrichmentConfig',
    'EnrichmentResult',
    'FallbackEnrichmentStrategy',
    'run_enrichment_analysis',
    'GOAnnotator',
    'KEGGPathwayMapper',
    'MSigDBLoader',
    'EPICAnnotationMapper'
]

__version__ = '1.0.0'
__author__ = 'HIIT-Methylation-Classification Team'
