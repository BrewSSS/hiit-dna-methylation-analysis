# Fallback Enrichment Methodology

This document describes the Fallback Enrichment Methodology used in the HIIT-Methylation-Classification pipeline, a robust approach for functional annotation of identified methylation biomarkers.

## Overview

Pathway enrichment analysis is essential for interpreting the biological significance of identified DNA methylation markers. However, enrichment analysis tools can fail due to:
- API limitations or service unavailability
- Gene list size constraints
- Missing annotations for specific genes
- Network connectivity issues

Our Fallback Enrichment Methodology implements a tiered approach with primary and backup strategies to ensure comprehensive functional annotation regardless of external service availability.

## Enrichment Strategy Overview

```
                    +------------------+
                    |   Gene List      |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    | Primary Method   |
                    | (clusterProfiler)|
                    +--------+---------+
                             |
              +--------------+--------------+
              |                             |
         Success                        Failure
              |                             |
              v                             v
     +----------------+           +------------------+
     | Return Results |           | Fallback Method  |
     +----------------+           | (MSigDB + Hyper) |
                                  +--------+---------+
                                           |
                                           v
                                  +------------------+
                                  | Return Results   |
                                  +------------------+
```

## Primary Method: clusterProfiler Integration

### Overview

[clusterProfiler](https://bioconductor.org/packages/clusterProfiler/) is an R/Bioconductor package providing comprehensive enrichment analysis capabilities. We interface with it through rpy2 for seamless Python integration.

### Supported Analyses

1. **Gene Ontology (GO) Enrichment**
   - Biological Process (BP)
   - Molecular Function (MF)
   - Cellular Component (CC)

2. **KEGG Pathway Analysis**
   - Metabolic pathways
   - Signaling pathways
   - Disease pathways

3. **Reactome Pathway Analysis**
   - Curated biological pathways
   - Hierarchical pathway relationships

### Implementation

```python
from src.enrichment import run_enrichment_analysis

# Primary enrichment using clusterProfiler
results = run_enrichment_analysis(
    gene_list=identified_genes,
    background=all_genes,  # Optional: custom background
    organism="hsapiens",
    databases=["GO_BP", "GO_MF", "GO_CC", "KEGG", "Reactome"],
    p_cutoff=0.05,
    q_cutoff=0.1,
    method="primary"  # Use clusterProfiler
)
```

### Configuration Options

```python
enrichment_config = {
    # Statistical parameters
    "p_value_cutoff": 0.05,
    "q_value_cutoff": 0.1,  # FDR threshold
    "min_gene_set_size": 10,
    "max_gene_set_size": 500,

    # Multiple testing correction
    "p_adjust_method": "BH",  # Benjamini-Hochberg

    # Visualization
    "show_top_n": 20,
    "plot_type": "dotplot"  # or "barplot", "cnetplot"
}
```

## Fallback Method: MSigDB + Hypergeometric Test

### When Fallback is Triggered

The fallback method activates when:
1. clusterProfiler is not installed or fails to import
2. R/rpy2 environment issues occur
3. API rate limits are exceeded
4. Primary analysis returns errors

### MSigDB Gene Sets

We use the [Molecular Signatures Database (MSigDB)](https://www.gsea-msigdb.org/gsea/msigdb/) gene set collections:

| Collection | Description | Use Case |
|------------|-------------|----------|
| H | Hallmark gene sets | Well-defined biological states |
| C2:CP | Canonical pathways | KEGG, Reactome, BioCarta |
| C5:GO | Gene Ontology sets | GO BP, MF, CC |
| C7 | Immunologic signatures | Immune-related studies |

### Hypergeometric Test Implementation

The hypergeometric test calculates the probability of observing k or more genes from a pathway in our gene list:

```python
from scipy import stats
import numpy as np

def hypergeometric_enrichment(
    query_genes: list,
    pathway_genes: list,
    background_size: int
) -> dict:
    """
    Perform hypergeometric test for pathway enrichment.

    Parameters
    ----------
    query_genes : list
        Genes identified in the study
    pathway_genes : list
        Genes in the pathway/gene set
    background_size : int
        Total number of genes in background

    Returns
    -------
    dict
        Enrichment statistics
    """
    # Calculate overlap
    query_set = set(query_genes)
    pathway_set = set(pathway_genes)
    overlap = query_set.intersection(pathway_set)

    k = len(overlap)      # Successes in sample
    M = background_size   # Population size
    n = len(pathway_set)  # Successes in population
    N = len(query_set)    # Sample size

    # Hypergeometric test (upper tail)
    p_value = stats.hypergeom.sf(k - 1, M, n, N)

    # Calculate fold enrichment
    expected = N * n / M
    fold_enrichment = k / expected if expected > 0 else np.inf

    return {
        "overlap_count": k,
        "overlap_genes": list(overlap),
        "p_value": p_value,
        "fold_enrichment": fold_enrichment,
        "expected_count": expected
    }
```

### Fallback Implementation

```python
from src.enrichment import FallbackEnrichment

# Initialize fallback enrichment
fallback = FallbackEnrichment(
    msigdb_path="data/external/msigdb/",  # Pre-downloaded gene sets
    organism="human"
)

# Run fallback analysis
fallback_results = fallback.run(
    gene_list=identified_genes,
    collections=["H", "C2:CP:KEGG", "C5:GO:BP"],
    background_genes=all_measured_genes,
    p_cutoff=0.05
)
```

## Unified Interface

### Single Entry Point

The recommended approach uses a unified interface that automatically handles method selection:

```python
from src.enrichment import EnrichmentAnalyzer

# Initialize analyzer (automatically detects available methods)
analyzer = EnrichmentAnalyzer(
    primary_method="clusterprofiler",
    fallback_method="msigdb_hypergeometric",
    auto_fallback=True  # Automatically switch on failure
)

# Run analysis - will use fallback if primary fails
results = analyzer.analyze(
    gene_list=genes,
    organism="hsapiens",
    databases=["GO_BP", "KEGG"]
)

# Check which method was used
print(f"Method used: {results.method_used}")
print(f"Fallback triggered: {results.fallback_used}")
```

### Result Structure

Both methods return results in a unified format:

```python
@dataclass
class EnrichmentResult:
    """Unified enrichment result structure."""

    term_id: str           # e.g., "GO:0006915"
    term_name: str         # e.g., "apoptotic process"
    database: str          # e.g., "GO_BP"
    p_value: float         # Raw p-value
    p_adjusted: float      # FDR-corrected p-value
    fold_enrichment: float # Observed/expected ratio
    gene_count: int        # Genes in overlap
    gene_list: List[str]   # Overlapping genes
    background_count: int  # Genes in term
    method_used: str       # "clusterprofiler" or "hypergeometric"
```

## Usage Examples

### Basic Enrichment Analysis

```python
from src.enrichment import EnrichmentAnalyzer
from src.features import FeatureSelector

# Get genes from selected features
selector = FeatureSelector(method="ten_level", level=5)
selected_features = selector.fit_transform(data, labels)

# Map CpG sites to genes
from src.data import map_cpg_to_genes
gene_list = map_cpg_to_genes(selected_features.columns)

# Run enrichment
analyzer = EnrichmentAnalyzer()
results = analyzer.analyze(
    gene_list=gene_list,
    databases=["GO_BP", "GO_MF", "KEGG"]
)

# View top results
print(results.get_top_terms(n=10, database="GO_BP"))
```

### Comparative Enrichment

Compare enrichment across different feature sets:

```python
from src.enrichment import compare_enrichments

# Get genes from different analyses
binary_genes = map_cpg_to_genes(binary_features.columns)
multiclass_genes = map_cpg_to_genes(multiclass_features.columns)
temporal_genes = map_cpg_to_genes(temporal_features.columns)

# Run comparative analysis
comparison = compare_enrichments(
    gene_lists={
        "binary": binary_genes,
        "multiclass": multiclass_genes,
        "temporal": temporal_genes
    },
    databases=["GO_BP", "KEGG"],
    p_cutoff=0.05
)

# Identify shared pathways
shared = comparison.get_shared_terms(min_lists=2)

# Visualize comparison
comparison.plot_upset("figures/enrichment_upset.png")
comparison.plot_heatmap("figures/enrichment_heatmap.png")
```

### Visualization

```python
from src.enrichment import EnrichmentAnalyzer
from src.visualization import plot_enrichment

analyzer = EnrichmentAnalyzer()
results = analyzer.analyze(gene_list, databases=["GO_BP"])

# Dot plot
plot_enrichment(
    results,
    plot_type="dotplot",
    top_n=15,
    output_path="figures/go_bp_dotplot.png"
)

# Bar plot
plot_enrichment(
    results,
    plot_type="barplot",
    top_n=10,
    color_by="p_adjusted",
    output_path="figures/go_bp_barplot.png"
)

# Network plot (gene-term relationships)
plot_enrichment(
    results,
    plot_type="network",
    top_n=5,
    output_path="figures/go_bp_network.png"
)
```

## Choosing Between Methods

### When Primary Method is Preferred

- Full clusterProfiler functionality needed
- Publication-quality visualizations required
- GSEA (ranked list analysis) needed
- Complex pathway comparisons

### When Fallback is Sufficient

- Quick exploratory analysis
- Limited computational resources
- R environment unavailable
- Reproducibility in diverse environments

### Method Comparison

| Feature | clusterProfiler | MSigDB + Hypergeometric |
|---------|-----------------|-------------------------|
| GO Analysis | Full support | Full support |
| KEGG | Full support | Full support |
| Reactome | Full support | Via MSigDB |
| GSEA | Supported | Not supported |
| Custom gene sets | Supported | Supported |
| Visualization | Extensive | Basic |
| Dependencies | R + rpy2 | Pure Python |
| Reproducibility | R version dependent | Fully reproducible |

## Data Requirements

### Gene Set Downloads

For fallback method, download MSigDB gene sets:

```bash
# Create directory
mkdir -p data/external/msigdb

# Download gene sets (requires registration)
# Visit: https://www.gsea-msigdb.org/gsea/msigdb/
# Download: c2.cp.kegg.v2023.2.Hs.symbols.gmt
# Download: c5.go.bp.v2023.2.Hs.symbols.gmt
# etc.
```

### Pre-configured Setup

```python
from src.enrichment import setup_enrichment_data

# Download and setup required gene set files
setup_enrichment_data(
    output_dir="data/external/msigdb/",
    collections=["H", "C2:CP", "C5:GO"],
    organism="human"
)
```

## Troubleshooting

### Common Issues

1. **clusterProfiler import fails**
   - Ensure R is installed with clusterProfiler package
   - Check rpy2 configuration
   - Fallback will activate automatically

2. **Empty results**
   - Check gene identifiers (SYMBOL vs ENTREZ)
   - Verify background gene list
   - Adjust p-value threshold

3. **Slow performance**
   - Use pre-downloaded MSigDB files
   - Limit number of databases tested
   - Consider parallel processing

### Debugging

```python
from src.enrichment import EnrichmentAnalyzer

analyzer = EnrichmentAnalyzer(verbose=True)

# Check available methods
print(f"Primary available: {analyzer.primary_available}")
print(f"Fallback available: {analyzer.fallback_available}")

# Run with diagnostics
results = analyzer.analyze(
    gene_list=genes,
    databases=["GO_BP"],
    return_diagnostics=True
)

print(results.diagnostics)
```

## References

- clusterProfiler: Yu G, et al. (2012) clusterProfiler: an R package for comparing biological themes among gene clusters. OMICS.
- MSigDB: Liberzon A, et al. (2015) The Molecular Signatures Database Hallmark Gene Set Collection. Cell Systems.

## See Also

- [Feature Selection Framework](feature_selection.md)
- [Batch Correction Approach](batch_correction.md)
- [API Reference](../api/reference.md)
