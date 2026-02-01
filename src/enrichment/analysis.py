"""
Enrichment analysis with fallback methodology.

This module implements the core innovation of the enrichment analysis pipeline:
a robust fallback strategy that automatically switches to MSigDB-based
hypergeometric testing when standard R-based tools are unavailable.

Classes:
    EnrichmentConfig: Configuration dataclass for enrichment analysis
    EnrichmentResult: Container dataclass for enrichment results
    FallbackEnrichmentStrategy: Core fallback methodology implementation
    EnrichmentAnalyzer: Complete enrichment pipeline for methylation biomarkers

Functions:
    run_enrichment_analysis: Convenience function for quick enrichment analysis
"""

import json
import logging
import os
import urllib.request
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .databases import GOAnnotator, KEGGPathwayMapper, MSigDBLoader

logger = logging.getLogger(__name__)


@dataclass
class EnrichmentConfig:
    """
    Configuration for enrichment analysis.

    Attributes:
        primary_method: Primary enrichment method to use ('clusterprofiler', 'gprofiler', 'enrichr')
        fallback_method: Fallback method when primary fails ('msigdb_hypergeometric', 'enrichr_api')
        pvalue_cutoff: P-value threshold for significance (default: 0.05)
        qvalue_cutoff: Adjusted p-value (FDR) threshold (default: 0.1)
        min_gene_set_size: Minimum gene set size to consider (default: 10)
        max_gene_set_size: Maximum gene set size to consider (default: 500)
        background_size: Size of background gene universe (default: 20000)
        correction_method: Multiple testing correction method (default: 'fdr_bh')

    Example:
        >>> config = EnrichmentConfig(
        ...     pvalue_cutoff=0.01,
        ...     qvalue_cutoff=0.05,
        ...     min_gene_set_size=15
        ... )
        >>> analyzer = EnrichmentAnalyzer(config=config)
    """
    primary_method: str = "gprofiler"
    fallback_method: str = "msigdb_hypergeometric"
    pvalue_cutoff: float = 0.05
    qvalue_cutoff: float = 0.1
    min_gene_set_size: int = 10
    max_gene_set_size: int = 500
    background_size: int = 20000
    correction_method: str = "fdr_bh"


@dataclass
class EnrichmentResult:
    """
    Container for a single enrichment analysis result.

    Attributes:
        term_id: Identifier for the enriched term (GO ID, pathway ID, etc.)
        term_name: Human-readable name of the enriched term
        pvalue: Raw p-value from statistical test
        qvalue: Adjusted p-value after multiple testing correction
        overlap_count: Number of query genes overlapping with gene set
        gene_set_size: Total size of the gene set
        overlap_genes: List of overlapping gene symbols
        enrichment_score: Fold enrichment or similar score
        category: Source category (GO_BP, GO_MF, GO_CC, KEGG, REACTOME, etc.)

    Example:
        >>> result = EnrichmentResult(
        ...     term_id='GO:0006915',
        ...     term_name='apoptotic process',
        ...     pvalue=0.001,
        ...     qvalue=0.01,
        ...     overlap_count=5,
        ...     gene_set_size=100,
        ...     overlap_genes=['TP53', 'BCL2', 'BAX', 'CASP3', 'CASP9'],
        ...     enrichment_score=3.5,
        ...     category='GO_BP'
        ... )
    """
    term_id: str
    term_name: str
    pvalue: float
    qvalue: float
    overlap_count: int
    gene_set_size: int
    overlap_genes: List[str]
    enrichment_score: float
    category: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            'term_id': self.term_id,
            'term_name': self.term_name,
            'p_value': self.pvalue,
            'p_adjusted': self.qvalue,
            'overlap_count': self.overlap_count,
            'gene_set_size': self.gene_set_size,
            'overlap_genes': ';'.join(self.overlap_genes),
            'enrichment_score': self.enrichment_score,
            'category': self.category
        }


class FallbackEnrichmentStrategy:
    """
    Fallback Enrichment Methodology.

    Implements a robust enrichment analysis strategy that automatically
    falls back to MSigDB gene sets with hypergeometric testing when
    standard annotation databases (org.Hs.eg.db via R/clusterProfiler)
    are unavailable.

    This is the core innovation of the enrichment module, enabling
    enrichment analysis in pure Python environments without requiring
    R dependencies.

    Attributes:
        primary_method: Primary enrichment method ('clusterprofiler' or 'gprofiler')
        fallback_method: Fallback method ('msigdb_hypergeometric')
        msigdb_loader: MSigDB gene set loader instance
        method_used: Records which method was actually used

    Example:
        >>> strategy = FallbackEnrichmentStrategy()
        >>> results = strategy.run_analysis(
        ...     gene_list=['TP53', 'BRCA1', 'EGFR'],
        ...     background_size=20000
        ... )
        >>> print(f"Method used: {strategy.get_method_used()}")
    """

    # Supported primary methods
    PRIMARY_METHODS = ['clusterprofiler', 'gprofiler', 'enrichr']

    # Supported fallback methods
    FALLBACK_METHODS = ['msigdb_hypergeometric', 'enrichr_api']

    def __init__(self,
                 primary_method: str = 'clusterprofiler',
                 fallback_method: str = 'msigdb_hypergeometric',
                 msigdb_path: Optional[str] = None):
        """
        Initialize the fallback enrichment strategy.

        Args:
            primary_method: Primary enrichment method to attempt first.
                           Options: 'clusterprofiler', 'gprofiler', 'enrichr'
            fallback_method: Fallback method when primary fails.
                            Options: 'msigdb_hypergeometric', 'enrichr_api'
            msigdb_path: Path to directory containing MSigDB GMT files.
                        Required for msigdb_hypergeometric fallback.
        """
        if primary_method not in self.PRIMARY_METHODS:
            raise ValueError(f"Unknown primary method: {primary_method}. "
                           f"Choose from {self.PRIMARY_METHODS}")

        if fallback_method not in self.FALLBACK_METHODS:
            raise ValueError(f"Unknown fallback method: {fallback_method}. "
                           f"Choose from {self.FALLBACK_METHODS}")

        self.primary_method = primary_method
        self.fallback_method = fallback_method
        self.msigdb_path = msigdb_path
        self.msigdb_loader: Optional[MSigDBLoader] = None
        self._method_used: Optional[str] = None
        self._rpy2_available: Optional[bool] = None

        # Initialize MSigDB loader if path provided
        if msigdb_path:
            self.msigdb_loader = MSigDBLoader(gmt_dir=msigdb_path)

    def run_analysis(self,
                     gene_list: List[str],
                     background_size: int = 20000,
                     ontologies: List[str] = ['BP', 'MF', 'CC'],
                     pvalue_cutoff: float = 0.05,
                     qvalue_cutoff: float = 0.1,
                     min_gene_set_size: int = 10,
                     max_gene_set_size: int = 500) -> Dict[str, Any]:
        """
        Run enrichment analysis with automatic fallback.

        Attempts the primary method first. If it fails (e.g., due to missing
        R dependencies), automatically falls back to the fallback method.

        Args:
            gene_list: List of gene symbols to analyze
            background_size: Size of background gene universe (default: 20000)
            ontologies: GO ontologies to include (default: ['BP', 'MF', 'CC'])
            pvalue_cutoff: P-value threshold for significance (default: 0.05)
            qvalue_cutoff: Adjusted p-value threshold (default: 0.1)
            min_gene_set_size: Minimum gene set size to consider (default: 10)
            max_gene_set_size: Maximum gene set size to consider (default: 500)

        Returns:
            Dictionary containing:
                - 'results': DataFrame with enrichment results
                - 'method_used': Which method was used
                - 'parameters': Analysis parameters
                - 'summary': Summary statistics
        """
        # Validate input
        if not gene_list:
            raise ValueError("gene_list cannot be empty")

        gene_list = list(set(gene_list))  # Remove duplicates
        logger.info(f"Running enrichment analysis for {len(gene_list)} genes")

        # Try primary method
        try:
            results = self._try_primary_method(gene_list, ontologies, pvalue_cutoff)
            if results is not None and len(results) > 0:
                self._method_used = self.primary_method
                return self._format_results(results, gene_list, pvalue_cutoff)
        except Exception as e:
            logger.warning(f"Primary method ({self.primary_method}) failed: {e}")
            logger.info(f"Falling back to {self.fallback_method}")

        # Run fallback method
        results = self._run_fallback_method(
            gene_list=gene_list,
            background_size=background_size,
            pvalue_cutoff=pvalue_cutoff,
            qvalue_cutoff=qvalue_cutoff,
            min_size=min_gene_set_size,
            max_size=max_gene_set_size
        )

        self._method_used = self.fallback_method
        return self._format_results(results, gene_list, pvalue_cutoff)

    def _try_primary_method(self,
                            gene_list: List[str],
                            ontologies: List[str],
                            pvalue_cutoff: float) -> Optional[pd.DataFrame]:
        """
        Attempt to run the primary enrichment method.

        Args:
            gene_list: List of gene symbols
            ontologies: GO ontologies to query
            pvalue_cutoff: P-value threshold

        Returns:
            DataFrame with results, or None if method fails
        """
        if self.primary_method == 'clusterprofiler':
            return self._try_clusterprofiler(gene_list, ontologies, pvalue_cutoff)
        elif self.primary_method == 'gprofiler':
            return self._try_gprofiler(gene_list, ontologies, pvalue_cutoff)
        elif self.primary_method == 'enrichr':
            return self._try_enrichr_primary(gene_list)
        else:
            return None

    def _try_clusterprofiler(self,
                             gene_list: List[str],
                             ontologies: List[str],
                             pvalue_cutoff: float) -> Optional[pd.DataFrame]:
        """
        Attempt clusterProfiler analysis via rpy2.

        Args:
            gene_list: List of gene symbols
            ontologies: GO ontologies
            pvalue_cutoff: P-value threshold

        Returns:
            DataFrame with results, or None if rpy2 unavailable
        """
        # Check if rpy2 is available
        if self._rpy2_available is None:
            try:
                import rpy2.robjects as ro
                from rpy2.robjects import pandas2ri
                from rpy2.robjects.packages import importr
                self._rpy2_available = True
            except ImportError:
                self._rpy2_available = False
                logger.info("rpy2 not available, will use fallback method")

        if not self._rpy2_available:
            return None

        try:
            import rpy2.robjects as ro
            from rpy2.robjects import pandas2ri
            from rpy2.robjects.packages import importr

            pandas2ri.activate()

            # Import R packages
            cluster_profiler = importr('clusterProfiler')
            org_db = importr('org.Hs.eg.db')

            # Convert gene list to R vector
            gene_vector = ro.StrVector(gene_list)

            all_results = []

            for ont in ontologies:
                # Run enrichGO
                result = cluster_profiler.enrichGO(
                    gene=gene_vector,
                    OrgDb=org_db,
                    keyType='SYMBOL',
                    ont=ont,
                    pvalueCutoff=pvalue_cutoff,
                    pAdjustMethod='BH'
                )

                # Convert to DataFrame
                with (ro.default_converter + pandas2ri.converter).context():
                    result_df = ro.conversion.get_conversion().rpy2py(
                        ro.r['as.data.frame'](result)
                    )

                if len(result_df) > 0:
                    result_df['ontology'] = ont
                    all_results.append(result_df)

            if all_results:
                return pd.concat(all_results, ignore_index=True)
            return None

        except Exception as e:
            logger.warning(f"clusterProfiler failed: {e}")
            return None

    def _try_gprofiler(self,
                       gene_list: List[str],
                       ontologies: List[str],
                       pvalue_cutoff: float) -> Optional[pd.DataFrame]:
        """
        Attempt g:Profiler analysis via API.

        Args:
            gene_list: List of gene symbols
            ontologies: GO ontologies
            pvalue_cutoff: P-value threshold

        Returns:
            DataFrame with results, or None if API fails
        """
        url = 'https://biit.cs.ut.ee/gprofiler/api/gost/profile/'

        # Map ontologies to g:Profiler sources
        source_map = {'BP': 'GO:BP', 'MF': 'GO:MF', 'CC': 'GO:CC'}
        sources = [source_map.get(ont, ont) for ont in ontologies]

        data = {
            'organism': 'hsapiens',
            'query': gene_list,
            'sources': sources,
            'user_threshold': pvalue_cutoff,
            'all_results': False,
            'ordered': False,
            'combined': False,
            'measure_underrepresentation': False,
            'domain_scope': 'annotated',
            'significance_threshold_method': 'g_SCS'
        }

        try:
            json_data = json.dumps(data).encode('utf-8')
            req = urllib.request.Request(
                url,
                data=json_data,
                headers={'Content-Type': 'application/json'}
            )

            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read().decode('utf-8'))

            if 'result' in result and result['result']:
                terms = result['result']
                df = pd.DataFrame([{
                    'term_id': t['native'],
                    'term_name': t['name'],
                    'source': t['source'],
                    'p_value': t['p_value'],
                    'term_size': t['term_size'],
                    'query_size': t['query_size'],
                    'intersection_size': t['intersection_size'],
                    'precision': t['intersection_size'] / t['query_size'] if t['query_size'] > 0 else 0,
                    'recall': t['intersection_size'] / t['term_size'] if t['term_size'] > 0 else 0
                } for t in terms])

                return df

            return None

        except Exception as e:
            logger.warning(f"g:Profiler API failed: {e}")
            return None

    def _try_enrichr_primary(self, gene_list: List[str]) -> Optional[pd.DataFrame]:
        """
        Attempt Enrichr analysis via API as primary method.

        Args:
            gene_list: List of gene symbols

        Returns:
            DataFrame with results, or None if API fails
        """
        return self._run_enrichr_api(gene_list)

    def _run_fallback_method(self,
                             gene_list: List[str],
                             background_size: int,
                             pvalue_cutoff: float,
                             qvalue_cutoff: float,
                             min_size: int,
                             max_size: int) -> pd.DataFrame:
        """
        Run the fallback enrichment method.

        Args:
            gene_list: List of gene symbols
            background_size: Background gene universe size
            pvalue_cutoff: P-value threshold
            qvalue_cutoff: Adjusted p-value threshold
            min_size: Minimum gene set size
            max_size: Maximum gene set size

        Returns:
            DataFrame with enrichment results
        """
        if self.fallback_method == 'msigdb_hypergeometric':
            return self._run_msigdb_hypergeometric(
                gene_list=gene_list,
                background_size=background_size,
                pvalue_cutoff=pvalue_cutoff,
                qvalue_cutoff=qvalue_cutoff,
                min_size=min_size,
                max_size=max_size
            )
        elif self.fallback_method == 'enrichr_api':
            result = self._run_enrichr_api(gene_list)
            if result is not None:
                return result
            # If Enrichr fails, try MSigDB as last resort
            logger.warning("Enrichr API failed, trying MSigDB hypergeometric")
            return self._run_msigdb_hypergeometric(
                gene_list=gene_list,
                background_size=background_size,
                pvalue_cutoff=pvalue_cutoff,
                qvalue_cutoff=qvalue_cutoff,
                min_size=min_size,
                max_size=max_size
            )
        else:
            raise ValueError(f"Unknown fallback method: {self.fallback_method}")

    def _run_msigdb_hypergeometric(self,
                                   gene_list: List[str],
                                   background_size: int,
                                   pvalue_cutoff: float,
                                   qvalue_cutoff: float,
                                   min_size: int,
                                   max_size: int) -> pd.DataFrame:
        """
        Run MSigDB-based hypergeometric enrichment analysis.

        This is the core fallback implementation using scipy's hypergeometric
        distribution for statistical testing.

        Args:
            gene_list: List of gene symbols
            background_size: Background gene universe size
            pvalue_cutoff: P-value threshold
            qvalue_cutoff: Adjusted p-value threshold
            min_size: Minimum gene set size
            max_size: Maximum gene set size

        Returns:
            DataFrame with enrichment results
        """
        # Ensure MSigDB loader is initialized
        if self.msigdb_loader is None:
            if self.msigdb_path:
                self.msigdb_loader = MSigDBLoader(gmt_dir=self.msigdb_path)
                self.msigdb_loader.load_gene_sets()
            else:
                # Use Enrichr API as backup if no MSigDB path
                logger.warning("No MSigDB path provided, using Enrichr API")
                return self._run_enrichr_api(gene_list) or pd.DataFrame()

        query_genes = set(gene_list)
        results = []

        for set_name, gene_set in self.msigdb_loader.gene_sets.items():
            set_size = len(gene_set)

            # Filter by size
            if set_size < min_size or set_size > max_size:
                continue

            # Calculate overlap
            overlap = query_genes.intersection(gene_set)
            overlap_size = len(overlap)

            if overlap_size == 0:
                continue

            # Hypergeometric test
            p_value = self.hypergeometric_test(
                query_genes=query_genes,
                gene_set=gene_set,
                background_size=background_size
            )

            # Calculate fold enrichment
            expected = (len(query_genes) * set_size) / background_size
            fold_enrichment = overlap_size / expected if expected > 0 else 0

            results.append({
                'term_id': set_name,
                'term_name': set_name.replace('_', ' '),
                'source': 'MSigDB',
                'p_value': p_value,
                'term_size': set_size,
                'query_size': len(query_genes),
                'intersection_size': overlap_size,
                'fold_enrichment': fold_enrichment,
                'genes': ';'.join(sorted(overlap))
            })

        if not results:
            return pd.DataFrame()

        # Create DataFrame and apply multiple testing correction
        df = pd.DataFrame(results)
        df = df.sort_values('p_value')

        # Apply Benjamini-Hochberg correction
        _, pvals_corrected, _, _ = multipletests(
            df['p_value'].values,
            method='fdr_bh'
        )
        df['p_adjusted'] = pvals_corrected

        # Filter by significance thresholds
        df = df[(df['p_value'] <= pvalue_cutoff) & (df['p_adjusted'] <= qvalue_cutoff)]

        return df.reset_index(drop=True)

    def _run_enrichr_api(self, gene_list: List[str]) -> Optional[pd.DataFrame]:
        """
        Run Enrichr analysis via their public API.

        Args:
            gene_list: List of gene symbols

        Returns:
            DataFrame with results, or None if API fails
        """
        ENRICHR_URL = 'https://maayanlab.cloud/Enrichr'

        try:
            # Step 1: Submit gene list
            genes_str = '\n'.join(gene_list)
            data = urllib.parse.urlencode({
                'list': genes_str,
                'description': 'enrichment_query'
            }).encode('utf-8')

            req = urllib.request.Request(f"{ENRICHR_URL}/addList", data=data)
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode('utf-8'))

            user_list_id = result.get('userListId')
            if not user_list_id:
                return None

            # Step 2: Query libraries
            libraries = [
                'GO_Biological_Process_2023',
                'GO_Molecular_Function_2023',
                'GO_Cellular_Component_2023',
                'KEGG_2021_Human'
            ]

            all_results = []

            for library in libraries:
                url = f"{ENRICHR_URL}/enrich?userListId={user_list_id}&backgroundType={library}"

                with urllib.request.urlopen(url, timeout=30) as response:
                    lib_result = json.loads(response.read().decode('utf-8'))

                if library in lib_result:
                    for term in lib_result[library]:
                        all_results.append({
                            'term_id': term[1].split('(')[-1].rstrip(')') if '(' in term[1] else term[1],
                            'term_name': term[1].split('(')[0].strip() if '(' in term[1] else term[1],
                            'source': library.replace('_2023', '').replace('_2021_Human', ''),
                            'p_value': term[2],
                            'p_adjusted': term[6],
                            'z_score': term[3],
                            'combined_score': term[4],
                            'genes': term[5] if isinstance(term[5], str) else ';'.join(term[5]),
                            'intersection_size': len(term[5]) if isinstance(term[5], list) else len(term[5].split(';'))
                        })

            if all_results:
                return pd.DataFrame(all_results)
            return None

        except Exception as e:
            logger.warning(f"Enrichr API failed: {e}")
            return None

    def hypergeometric_test(self,
                            query_genes: Set[str],
                            gene_set: Set[str],
                            background_size: int) -> float:
        """
        Calculate hypergeometric p-value for gene set enrichment.

        Uses the survival function (1 - CDF) of the hypergeometric distribution
        to calculate the probability of observing the overlap by chance.

        The hypergeometric distribution models sampling without replacement:
        - M: Total population size (background_size)
        - n: Number of success states in population (gene_set size)
        - N: Number of draws (query_genes size)
        - k: Number of observed successes (overlap)

        P(X >= k) = 1 - P(X < k) = 1 - CDF(k-1)

        Args:
            query_genes: Set of query gene symbols
            gene_set: Set of genes in the gene set being tested
            background_size: Total size of the gene universe

        Returns:
            P-value from hypergeometric test
        """
        # Calculate overlap
        overlap = len(query_genes.intersection(gene_set))

        if overlap == 0:
            return 1.0

        # Hypergeometric parameters
        M = background_size  # Population size
        n = len(gene_set)    # Success states in population
        N = len(query_genes) # Number of draws
        k = overlap          # Observed successes

        # Calculate p-value using survival function
        # sf(k-1) = P(X >= k) = 1 - P(X <= k-1)
        p_value = stats.hypergeom.sf(k - 1, M, n, N)

        return p_value

    def get_method_used(self) -> Optional[str]:
        """
        Return which method was used for the last analysis.

        Returns:
            Name of the method used, or None if no analysis has been run
        """
        return self._method_used

    def _format_results(self,
                        results: pd.DataFrame,
                        gene_list: List[str],
                        pvalue_cutoff: float) -> Dict[str, Any]:
        """
        Format enrichment results into standardized output.

        Args:
            results: DataFrame with raw results
            gene_list: Original query gene list
            pvalue_cutoff: P-value threshold used

        Returns:
            Standardized results dictionary
        """
        if results is None or len(results) == 0:
            return {
                'results': pd.DataFrame(),
                'method_used': self._method_used,
                'parameters': {
                    'query_size': len(gene_list),
                    'pvalue_cutoff': pvalue_cutoff
                },
                'summary': {
                    'total_terms': 0,
                    'significant_terms': 0
                }
            }

        return {
            'results': results,
            'method_used': self._method_used,
            'parameters': {
                'query_size': len(gene_list),
                'pvalue_cutoff': pvalue_cutoff
            },
            'summary': {
                'total_terms': len(results),
                'significant_terms': len(results[results['p_value'] <= pvalue_cutoff]) if 'p_value' in results.columns else len(results)
            }
        }


class EnrichmentAnalyzer:
    """
    Complete enrichment analysis pipeline for methylation biomarkers.

    Provides end-to-end enrichment analysis including CpG-to-gene mapping,
    GO/KEGG/Reactome enrichment, and comprehensive reporting. Uses the
    FallbackEnrichmentStrategy for robust analysis.

    Attributes:
        annotation_file: Path to array annotation file (e.g., EPIC manifest)
        species: Species for analysis (default: 'human')
        strategy: FallbackEnrichmentStrategy instance
        annotations: Loaded annotation DataFrame

    Example:
        >>> analyzer = EnrichmentAnalyzer(
        ...     annotation_file='GPL21145_MethylationEPIC.csv'
        ... )
        >>> results = analyzer.run_comprehensive_analysis(cpg_sites)
        >>> analyzer.generate_report(results, 'enrichment_report.md')
    """

    # Supported array platforms
    PLATFORMS = {
        'EPIC': 'Illumina MethylationEPIC',
        '450K': 'Illumina HumanMethylation450',
        '27K': 'Illumina HumanMethylation27'
    }

    def __init__(self,
                 annotation_file: Optional[str] = None,
                 species: str = 'human',
                 msigdb_path: Optional[str] = None):
        """
        Initialize the enrichment analyzer.

        Args:
            annotation_file: Path to array annotation file (CSV format)
            species: Species for analysis (default: 'human')
            msigdb_path: Path to MSigDB GMT files directory
        """
        self.annotation_file = annotation_file
        self.species = species
        self.annotations: Optional[pd.DataFrame] = None
        self._annotation_loaded = False

        # Initialize enrichment strategy
        self.strategy = FallbackEnrichmentStrategy(
            primary_method='gprofiler',
            fallback_method='msigdb_hypergeometric',
            msigdb_path=msigdb_path
        )

        # Initialize database interfaces
        self.go_annotator = GOAnnotator()
        self.kegg_mapper = KEGGPathwayMapper(organism='hsa')

        # Load annotations if provided
        if annotation_file:
            self._load_annotations(annotation_file)

    def _load_annotations(self, annotation_file: str) -> None:
        """
        Load array annotation file.

        Args:
            annotation_file: Path to annotation CSV file
        """
        try:
            # Try different loading strategies
            for skiprows in [0, 7, 8]:
                try:
                    df = pd.read_csv(
                        annotation_file,
                        sep=',',
                        skiprows=skiprows,
                        low_memory=False,
                        on_bad_lines='skip'
                    )

                    # Check if required columns exist
                    required_cols = ['Name', 'UCSC_RefGene_Name']
                    alt_cols = ['IlmnID', 'UCSC_RefGene_Name']

                    if all(col in df.columns for col in required_cols):
                        self.annotations = df
                        self._annotation_loaded = True
                        logger.info(f"Loaded {len(df)} annotations from {annotation_file}")
                        return
                    elif all(col in df.columns for col in alt_cols):
                        df = df.rename(columns={'IlmnID': 'Name'})
                        self.annotations = df
                        self._annotation_loaded = True
                        logger.info(f"Loaded {len(df)} annotations from {annotation_file}")
                        return

                except Exception:
                    continue

            logger.warning(f"Could not load annotations from {annotation_file}")

        except Exception as e:
            logger.error(f"Failed to load annotation file: {e}")

    def map_cpg_to_genes(self,
                         cpg_sites: List[str],
                         annotation_platform: str = 'EPIC') -> Tuple[List[str], Dict[str, List[str]]]:
        """
        Map CpG sites to gene symbols using array annotation.

        Args:
            cpg_sites: List of CpG site identifiers (e.g., ['cg00000029', 'cg00000108'])
            annotation_platform: Array platform ('EPIC', '450K', '27K')

        Returns:
            Tuple of:
                - List of unique gene symbols
                - Dictionary mapping CpG sites to their associated genes
        """
        if not self._annotation_loaded:
            logger.warning("No annotation file loaded. Returning empty mapping.")
            return [], {}

        # Filter annotations to query CpG sites
        mask = self.annotations['Name'].isin(cpg_sites)
        matched = self.annotations[mask]

        cpg_gene_map: Dict[str, List[str]] = {}
        all_genes: Set[str] = set()

        for _, row in matched.iterrows():
            cpg_id = row['Name']
            gene_str = row.get('UCSC_RefGene_Name', '')

            if pd.isna(gene_str) or not gene_str:
                cpg_gene_map[cpg_id] = []
                continue

            # Parse gene names (semicolon-separated)
            genes = [g.strip() for g in str(gene_str).split(';') if g.strip()]
            genes = list(set(genes))  # Remove duplicates

            cpg_gene_map[cpg_id] = genes
            all_genes.update(genes)

        logger.info(f"Mapped {len(cpg_sites)} CpG sites to {len(all_genes)} unique genes")

        return sorted(list(all_genes)), cpg_gene_map

    def run_go_enrichment(self,
                          genes: List[str],
                          ontology: str = 'BP',
                          pvalue_cutoff: float = 0.05,
                          qvalue_cutoff: float = 0.1) -> pd.DataFrame:
        """
        Run Gene Ontology enrichment analysis.

        Args:
            genes: List of gene symbols
            ontology: GO ontology ('BP', 'MF', 'CC', or 'ALL')
            pvalue_cutoff: P-value threshold (default: 0.05)
            qvalue_cutoff: Adjusted p-value threshold (default: 0.1)

        Returns:
            DataFrame with GO enrichment results
        """
        ontologies = ['BP', 'MF', 'CC'] if ontology == 'ALL' else [ontology]

        results = self.strategy.run_analysis(
            gene_list=genes,
            ontologies=ontologies,
            pvalue_cutoff=pvalue_cutoff,
            qvalue_cutoff=qvalue_cutoff
        )

        return results['results']

    def run_kegg_enrichment(self,
                            genes: List[str],
                            organism: str = 'hsa',
                            pvalue_cutoff: float = 0.05) -> pd.DataFrame:
        """
        Run KEGG pathway enrichment analysis.

        Args:
            genes: List of gene symbols
            organism: KEGG organism code (default: 'hsa' for human)
            pvalue_cutoff: P-value threshold (default: 0.05)

        Returns:
            DataFrame with KEGG enrichment results
        """
        # Use Enrichr API for KEGG
        try:
            ENRICHR_URL = 'https://maayanlab.cloud/Enrichr'

            # Submit gene list
            genes_str = '\n'.join(genes)
            data = urllib.parse.urlencode({
                'list': genes_str,
                'description': 'kegg_query'
            }).encode('utf-8')

            req = urllib.request.Request(f"{ENRICHR_URL}/addList", data=data)
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode('utf-8'))

            user_list_id = result.get('userListId')
            if not user_list_id:
                return pd.DataFrame()

            # Query KEGG library
            url = f"{ENRICHR_URL}/enrich?userListId={user_list_id}&backgroundType=KEGG_2021_Human"

            with urllib.request.urlopen(url, timeout=30) as response:
                lib_result = json.loads(response.read().decode('utf-8'))

            results = []
            if 'KEGG_2021_Human' in lib_result:
                for term in lib_result['KEGG_2021_Human']:
                    if term[2] <= pvalue_cutoff:
                        results.append({
                            'pathway_id': term[1].split('(')[-1].rstrip(')') if '(' in term[1] else '',
                            'pathway_name': term[1].split('(')[0].strip() if '(' in term[1] else term[1],
                            'p_value': term[2],
                            'p_adjusted': term[6],
                            'z_score': term[3],
                            'combined_score': term[4],
                            'genes': term[5] if isinstance(term[5], str) else ';'.join(term[5]),
                            'gene_count': len(term[5]) if isinstance(term[5], list) else len(term[5].split(';'))
                        })

            return pd.DataFrame(results)

        except Exception as e:
            logger.warning(f"KEGG enrichment failed: {e}")
            return pd.DataFrame()

    def run_reactome_enrichment(self,
                                genes: List[str],
                                pvalue_cutoff: float = 0.05) -> pd.DataFrame:
        """
        Run Reactome pathway enrichment analysis.

        Args:
            genes: List of gene symbols
            pvalue_cutoff: P-value threshold (default: 0.05)

        Returns:
            DataFrame with Reactome enrichment results
        """
        try:
            # Use Reactome Analysis Service API
            url = 'https://reactome.org/AnalysisService/identifiers/projection'

            genes_str = '\n'.join(genes)

            req = urllib.request.Request(
                url,
                data=genes_str.encode('utf-8'),
                headers={'Content-Type': 'text/plain'}
            )

            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode('utf-8'))

            results = []
            if 'pathways' in result:
                for pathway in result['pathways']:
                    fdr = pathway.get('entities', {}).get('fdr', 1.0)
                    pval = pathway.get('entities', {}).get('pValue', 1.0)

                    if pval <= pvalue_cutoff:
                        results.append({
                            'pathway_id': pathway.get('stId', ''),
                            'pathway_name': pathway.get('name', ''),
                            'p_value': pval,
                            'p_adjusted': fdr,
                            'found_entities': pathway.get('entities', {}).get('found', 0),
                            'total_entities': pathway.get('entities', {}).get('total', 0),
                            'species': pathway.get('species', {}).get('name', '')
                        })

            return pd.DataFrame(results)

        except Exception as e:
            logger.warning(f"Reactome enrichment failed: {e}")
            return pd.DataFrame()

    def run_comprehensive_analysis(self,
                                   cpg_sites: List[str],
                                   pvalue_cutoff: float = 0.05) -> Dict[str, Any]:
        """
        Run complete enrichment pipeline: CpG -> Gene -> GO/KEGG/Reactome.

        Args:
            cpg_sites: List of CpG site identifiers
            pvalue_cutoff: P-value threshold for all analyses

        Returns:
            Dictionary containing:
                - 'gene_mapping': CpG to gene mapping results
                - 'go_bp': GO Biological Process results
                - 'go_mf': GO Molecular Function results
                - 'go_cc': GO Cellular Component results
                - 'kegg': KEGG pathway results
                - 'reactome': Reactome pathway results
                - 'summary': Summary statistics
        """
        logger.info(f"Running comprehensive enrichment for {len(cpg_sites)} CpG sites")

        # Step 1: Map CpG sites to genes
        genes, cpg_gene_map = self.map_cpg_to_genes(cpg_sites)

        if not genes:
            logger.warning("No genes mapped from CpG sites")
            return {
                'gene_mapping': {'genes': [], 'cpg_gene_map': cpg_gene_map},
                'go_bp': pd.DataFrame(),
                'go_mf': pd.DataFrame(),
                'go_cc': pd.DataFrame(),
                'kegg': pd.DataFrame(),
                'reactome': pd.DataFrame(),
                'summary': {'total_cpg': len(cpg_sites), 'mapped_genes': 0}
            }

        # Step 2: Run enrichment analyses
        go_bp = self.run_go_enrichment(genes, ontology='BP', pvalue_cutoff=pvalue_cutoff)
        go_mf = self.run_go_enrichment(genes, ontology='MF', pvalue_cutoff=pvalue_cutoff)
        go_cc = self.run_go_enrichment(genes, ontology='CC', pvalue_cutoff=pvalue_cutoff)
        kegg = self.run_kegg_enrichment(genes, pvalue_cutoff=pvalue_cutoff)
        reactome = self.run_reactome_enrichment(genes, pvalue_cutoff=pvalue_cutoff)

        # Create summary
        summary = {
            'total_cpg': len(cpg_sites),
            'mapped_cpg': len([c for c, g in cpg_gene_map.items() if g]),
            'mapped_genes': len(genes),
            'go_bp_terms': len(go_bp) if isinstance(go_bp, pd.DataFrame) else 0,
            'go_mf_terms': len(go_mf) if isinstance(go_mf, pd.DataFrame) else 0,
            'go_cc_terms': len(go_cc) if isinstance(go_cc, pd.DataFrame) else 0,
            'kegg_pathways': len(kegg) if isinstance(kegg, pd.DataFrame) else 0,
            'reactome_pathways': len(reactome) if isinstance(reactome, pd.DataFrame) else 0,
            'method_used': self.strategy.get_method_used()
        }

        return {
            'gene_mapping': {'genes': genes, 'cpg_gene_map': cpg_gene_map},
            'go_bp': go_bp,
            'go_mf': go_mf,
            'go_cc': go_cc,
            'kegg': kegg,
            'reactome': reactome,
            'summary': summary
        }

    def compare_feature_sets(self,
                             feature_sets_dict: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Compare enrichment results across different feature selections.

        Args:
            feature_sets_dict: Dictionary mapping set names to CpG site lists
                              e.g., {'binary': [...], 'multiclass': [...]}

        Returns:
            DataFrame comparing enrichment across feature sets
        """
        comparison_results = []

        for set_name, cpg_sites in feature_sets_dict.items():
            logger.info(f"Analyzing feature set: {set_name}")

            # Run comprehensive analysis
            results = self.run_comprehensive_analysis(cpg_sites)

            # Extract summary statistics
            summary = results['summary']
            summary['feature_set'] = set_name

            # Add top enriched terms
            for source, key in [('GO_BP', 'go_bp'), ('GO_MF', 'go_mf'),
                               ('GO_CC', 'go_cc'), ('KEGG', 'kegg')]:
                df = results.get(key, pd.DataFrame())
                if isinstance(df, pd.DataFrame) and len(df) > 0:
                    if 'term_name' in df.columns:
                        top_term = df.iloc[0]['term_name']
                    elif 'pathway_name' in df.columns:
                        top_term = df.iloc[0]['pathway_name']
                    else:
                        top_term = 'N/A'
                    summary[f'top_{source.lower()}'] = top_term
                else:
                    summary[f'top_{source.lower()}'] = 'N/A'

            comparison_results.append(summary)

        return pd.DataFrame(comparison_results)

    def generate_report(self,
                        results: Dict[str, Any],
                        output_path: str) -> None:
        """
        Generate markdown report of enrichment results.

        Args:
            results: Results dictionary from run_comprehensive_analysis()
            output_path: Path to save the markdown report
        """
        lines = []
        lines.append("# Enrichment Analysis Report")
        lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Summary section
        summary = results.get('summary', {})
        lines.append("## Summary\n")
        lines.append(f"- Total CpG sites analyzed: {summary.get('total_cpg', 'N/A')}")
        lines.append(f"- CpG sites mapped to genes: {summary.get('mapped_cpg', 'N/A')}")
        lines.append(f"- Unique genes identified: {summary.get('mapped_genes', 'N/A')}")
        lines.append(f"- Enrichment method used: {summary.get('method_used', 'N/A')}")
        lines.append("")

        # Enrichment results sections
        enrichment_sources = [
            ('GO Biological Process', 'go_bp'),
            ('GO Molecular Function', 'go_mf'),
            ('GO Cellular Component', 'go_cc'),
            ('KEGG Pathways', 'kegg'),
            ('Reactome Pathways', 'reactome')
        ]

        for title, key in enrichment_sources:
            df = results.get(key, pd.DataFrame())
            lines.append(f"## {title}\n")

            if isinstance(df, pd.DataFrame) and len(df) > 0:
                lines.append(f"Total enriched terms: {len(df)}\n")
                lines.append("### Top 10 Enriched Terms\n")

                # Create table header
                name_col = 'term_name' if 'term_name' in df.columns else 'pathway_name'
                id_col = 'term_id' if 'term_id' in df.columns else 'pathway_id'

                lines.append(f"| {name_col.replace('_', ' ').title()} | P-value | Adjusted P-value |")
                lines.append("|---|---|---|")

                for _, row in df.head(10).iterrows():
                    name = row.get(name_col, 'N/A')
                    pval = row.get('p_value', 'N/A')
                    padj = row.get('p_adjusted', row.get('p_adj', 'N/A'))

                    if isinstance(pval, float):
                        pval = f"{pval:.2e}"
                    if isinstance(padj, float):
                        padj = f"{padj:.2e}"

                    lines.append(f"| {name} | {pval} | {padj} |")

                lines.append("")
            else:
                lines.append("No significant enrichment found.\n")

        # Write report
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))

        logger.info(f"Report saved to {output_path}")


def run_enrichment_analysis(
    gene_list: List[str],
    method: str = 'auto',
    config: Optional[EnrichmentConfig] = None,
    msigdb_path: Optional[str] = None,
    background_size: int = 20000,
    pvalue_cutoff: float = 0.05,
    qvalue_cutoff: float = 0.1
) -> pd.DataFrame:
    """
    Convenience function to run enrichment analysis.

    Provides a simple interface for running enrichment analysis on a gene list
    without manually instantiating strategy or analyzer classes.

    Args:
        gene_list: List of gene symbols to analyze.
        method: Method to use. Options:
            - 'auto': Try primary method, fall back automatically (default)
            - 'clusterprofiler': Use R/clusterProfiler via rpy2
            - 'gprofiler': Use g:Profiler API
            - 'enrichr': Use Enrichr API
            - 'msigdb': Use MSigDB hypergeometric test directly
        config: Optional EnrichmentConfig for custom parameters.
        msigdb_path: Path to directory containing MSigDB GMT files.
            Required when method='msigdb' or as fallback.
        background_size: Size of background gene universe (default: 20000).
        pvalue_cutoff: P-value threshold for significance (default: 0.05).
        qvalue_cutoff: Adjusted p-value threshold (default: 0.1).

    Returns:
        DataFrame with enrichment results containing columns:
        [term_id, term_name, p_value, p_adjusted, intersection_size,
         term_size, genes, source]

    Example:
        >>> results = run_enrichment_analysis(
        ...     gene_list=['TP53', 'BRCA1', 'EGFR', 'MYC'],
        ...     method='auto'
        ... )
        >>> print(results.head())
    """
    if not gene_list:
        raise ValueError("gene_list cannot be empty")

    if config is None:
        config = EnrichmentConfig(
            pvalue_cutoff=pvalue_cutoff,
            qvalue_cutoff=qvalue_cutoff,
            background_size=background_size
        )

    # Determine primary and fallback methods
    method_map = {
        'auto': ('gprofiler', 'msigdb_hypergeometric'),
        'clusterprofiler': ('clusterprofiler', 'msigdb_hypergeometric'),
        'gprofiler': ('gprofiler', 'msigdb_hypergeometric'),
        'enrichr': ('enrichr', 'msigdb_hypergeometric'),
        'msigdb': ('clusterprofiler', 'msigdb_hypergeometric'),  # Force fallback
    }

    if method not in method_map:
        raise ValueError(
            f"Unknown method: {method}. Choose from {list(method_map.keys())}"
        )

    primary, fallback = method_map[method]

    strategy = FallbackEnrichmentStrategy(
        primary_method=primary,
        fallback_method=fallback,
        msigdb_path=msigdb_path
    )

    # If method is 'msigdb', skip primary and go directly to fallback
    if method == 'msigdb':
        strategy._rpy2_available = False

    result = strategy.run_analysis(
        gene_list=gene_list,
        background_size=config.background_size,
        pvalue_cutoff=config.pvalue_cutoff,
        qvalue_cutoff=config.qvalue_cutoff,
        min_gene_set_size=config.min_gene_set_size,
        max_gene_set_size=config.max_gene_set_size
    )

    return result.get('results', pd.DataFrame())
