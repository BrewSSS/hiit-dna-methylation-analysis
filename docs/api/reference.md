# API Reference

This document provides the API reference for the HIIT-Methylation-Classification pipeline modules.

## Table of Contents

- [src.data](#srcdata)
- [src.features](#srcfeatures)
- [src.models](#srcmodels)
- [src.visualization](#srcvisualization)
- [src.enrichment](#srcenrichment)

---

## src.data

Data loading, preprocessing, and utility functions for methylation data.

### Functions

#### `load_methylation_data`

```python
def load_methylation_data(
    data_path: str,
    format: str = "auto",
    return_metadata: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Load methylation beta values from file or directory.

    Parameters
    ----------
    data_path : str
        Path to data file or directory containing methylation data.
    format : str, default="auto"
        Data format. Options: "auto", "csv", "tsv", "idat", "geo".
    return_metadata : bool, default=False
        If True, return tuple of (data, metadata).

    Returns
    -------
    pd.DataFrame or Tuple[pd.DataFrame, pd.DataFrame]
        Methylation beta values matrix (samples x CpG sites).
        If return_metadata=True, also returns sample metadata.

    Examples
    --------
    >>> data = load_methylation_data("data/raw/")
    >>> data, metadata = load_methylation_data("data/raw/", return_metadata=True)
    """
```

#### `preprocess_data`

```python
def preprocess_data(
    data: pd.DataFrame,
    normalize: bool = True,
    normalization_method: str = "quantile",
    remove_sex_chromosomes: bool = True,
    remove_snp_probes: bool = True,
    remove_cross_reactive: bool = True,
    min_detection_p: float = 0.01,
    min_beads: int = 3,
    impute_missing: bool = True,
    imputation_method: str = "knn"
) -> pd.DataFrame:
    """
    Preprocess methylation data with quality control and normalization.

    Parameters
    ----------
    data : pd.DataFrame
        Raw methylation beta values.
    normalize : bool, default=True
        Whether to apply normalization.
    normalization_method : str, default="quantile"
        Normalization method. Options: "quantile", "bmiq", "swan", "noob".
    remove_sex_chromosomes : bool, default=True
        Remove probes on X and Y chromosomes.
    remove_snp_probes : bool, default=True
        Remove probes containing SNPs.
    remove_cross_reactive : bool, default=True
        Remove cross-reactive probes.
    min_detection_p : float, default=0.01
        Detection p-value threshold for probe filtering.
    min_beads : int, default=3
        Minimum bead count threshold.
    impute_missing : bool, default=True
        Whether to impute missing values.
    imputation_method : str, default="knn"
        Imputation method. Options: "knn", "mean", "median".

    Returns
    -------
    pd.DataFrame
        Preprocessed methylation data.

    Examples
    --------
    >>> processed = preprocess_data(raw_data, normalize=True)
    """
```

#### `extract_batch_info`

```python
def extract_batch_info(
    metadata: pd.DataFrame,
    batch_column: str = None
) -> np.ndarray:
    """
    Extract batch information from sample metadata.

    Parameters
    ----------
    metadata : pd.DataFrame
        Sample metadata containing batch information.
    batch_column : str, optional
        Column name containing batch identifiers. If None, auto-detected.

    Returns
    -------
    np.ndarray
        Array of batch labels for each sample.
    """
```

#### `map_cpg_to_genes`

```python
def map_cpg_to_genes(
    cpg_ids: List[str],
    annotation: str = "EPIC",
    gene_column: str = "UCSC_RefGene_Name"
) -> List[str]:
    """
    Map CpG probe IDs to gene symbols.

    Parameters
    ----------
    cpg_ids : List[str]
        List of CpG probe identifiers (e.g., ["cg00000029", "cg00000108"]).
    annotation : str, default="EPIC"
        Array annotation to use. Options: "EPIC", "450K".
    gene_column : str, default="UCSC_RefGene_Name"
        Annotation column for gene mapping.

    Returns
    -------
    List[str]
        Unique gene symbols mapped from CpG probes.
    """
```

### Classes

#### `GEODataLoader`

```python
class GEODataLoader:
    """
    Load methylation data from Gene Expression Omnibus (GEO).

    Parameters
    ----------
    cache_dir : str, default="data/raw/"
        Directory to cache downloaded data.

    Attributes
    ----------
    accession : str
        Current GEO accession number.

    Methods
    -------
    download(accession)
        Download data from GEO.
    load_series_matrix()
        Load series matrix file.
    get_sample_metadata()
        Extract sample metadata.
    """

    def download(self, accession: str) -> None:
        """Download GEO dataset by accession number."""

    def load_series_matrix(self) -> pd.DataFrame:
        """Load and parse series matrix file."""

    def get_sample_metadata(self) -> pd.DataFrame:
        """Extract sample metadata from GEO record."""
```

---

## src.features

Feature engineering and selection modules.

### Classes

#### `FeatureSelector`

```python
class FeatureSelector:
    """
    Feature selection using the Ten-Level Framework.

    Parameters
    ----------
    method : str, default="ten_level"
        Selection method. Options: "ten_level", "variance", "differential", "combined".
    level : int, default=5
        Selection stringency level (1-10). Only used when method="ten_level".
    analysis_type : str, default="binary"
        Type of analysis. Options: "binary", "multiclass", "temporal".
    batch_aware : bool, default=False
        Whether to account for batch effects in selection.
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    selected_features_ : List[str]
        Names of selected features after fitting.
    selection_stats_ : dict
        Statistics from the selection process.

    Methods
    -------
    fit(X, y, batch=None)
        Fit the selector to data.
    transform(X)
        Transform data to selected features.
    fit_transform(X, y, batch=None)
        Fit and transform in one step.
    get_selection_stats()
        Get detailed selection statistics.
    get_config()
        Get configuration for reproducibility.
    """

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        batch: np.ndarray = None
    ) -> "FeatureSelector":
        """Fit the feature selector."""

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data to selected features."""

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        batch: np.ndarray = None
    ) -> pd.DataFrame:
        """Fit selector and transform data."""

    def get_selection_stats(self) -> dict:
        """Return selection statistics."""
```

### Functions

#### `get_consensus_features`

```python
def get_consensus_features(
    binary_features: List[str],
    multiclass_features: List[str],
    temporal_features: List[str],
    min_approaches: int = 2
) -> List[str]:
    """
    Find features selected across multiple analysis approaches.

    Parameters
    ----------
    binary_features : List[str]
        Features from binary analysis.
    multiclass_features : List[str]
        Features from multiclass analysis.
    temporal_features : List[str]
        Features from temporal analysis.
    min_approaches : int, default=2
        Minimum number of approaches a feature must appear in.

    Returns
    -------
    List[str]
        Consensus feature names.
    """
```

#### `analyze_level_stability`

```python
def analyze_level_stability(
    data: pd.DataFrame,
    labels: np.ndarray,
    levels: List[int],
    analysis_type: str = "binary"
) -> "StabilityReport":
    """
    Analyze feature selection stability across levels.

    Parameters
    ----------
    data : pd.DataFrame
        Methylation data.
    labels : np.ndarray
        Sample labels.
    levels : List[int]
        Levels to test.
    analysis_type : str, default="binary"
        Type of analysis.

    Returns
    -------
    StabilityReport
        Report containing stability metrics and stable features.
    """
```

---

## src.models

Machine learning models for HIIT classification.

### Classes

#### `HIITClassifier`

```python
class HIITClassifier:
    """
    Classification model for HIIT intervention prediction.

    Parameters
    ----------
    model_type : str, default="random_forest"
        Base model type. Options: "random_forest", "gradient_boosting",
        "logistic_regression", "svm", "neural_network".
    batch_aware : bool, default=False
        Whether to use batch-aware modeling.
    batch_strategy : str, default="covariate"
        Batch handling strategy. Options: "covariate", "stratified", "weighted".
    class_weight : str or dict, default="balanced"
        Class weighting strategy.
    random_state : int, optional
        Random seed for reproducibility.
    **model_params
        Additional parameters passed to base model.

    Attributes
    ----------
    model_ : object
        Fitted model object.
    feature_importances_ : np.ndarray
        Feature importance scores (if available).
    batch_encoder_ : object
        Fitted batch encoder (if batch_aware=True).

    Methods
    -------
    fit(X, y, batch=None)
        Fit the classifier.
    predict(X, batch=None)
        Predict class labels.
    predict_proba(X, batch=None)
        Predict class probabilities.
    evaluate(X, y, batch=None, metrics=None)
        Evaluate model performance.
    cross_validate(X, y, batch=None, cv_strategy="standard", n_splits=5)
        Perform cross-validation.
    get_feature_importance()
        Get feature importance ranking.
    """

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        batch: np.ndarray = None
    ) -> "HIITClassifier":
        """Fit the classifier to training data."""

    def predict(
        self,
        X: pd.DataFrame,
        batch: np.ndarray = None,
        batch_invariant: bool = False
    ) -> np.ndarray:
        """Predict class labels for samples."""

    def predict_proba(
        self,
        X: pd.DataFrame,
        batch: np.ndarray = None
    ) -> np.ndarray:
        """Predict class probabilities."""

    def evaluate(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        batch: np.ndarray = None,
        metrics: List[str] = None
    ) -> dict:
        """Evaluate model on test data."""

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        batch: np.ndarray = None,
        cv_strategy: str = "standard",
        n_splits: int = 5
    ) -> dict:
        """Perform cross-validation."""
```

#### `ModelEvaluator`

```python
class ModelEvaluator:
    """
    Comprehensive model evaluation utilities.

    Methods
    -------
    compute_metrics(y_true, y_pred, y_prob=None)
        Compute classification metrics.
    generate_report(model, X_test, y_test)
        Generate evaluation report.
    plot_roc_curve(y_true, y_prob, output_path=None)
        Plot ROC curve.
    plot_confusion_matrix(y_true, y_pred, output_path=None)
        Plot confusion matrix.
    plot_precision_recall(y_true, y_prob, output_path=None)
        Plot precision-recall curve.
    """
```

---

## src.visualization

Plotting and visualization utilities.

### Functions

#### `plot_methylation_heatmap`

```python
def plot_methylation_heatmap(
    data: pd.DataFrame,
    labels: np.ndarray,
    top_n: int = 50,
    cluster_samples: bool = True,
    cluster_features: bool = True,
    cmap: str = "RdBu_r",
    figsize: Tuple[int, int] = (12, 8),
    output_path: str = None
) -> plt.Figure:
    """
    Create heatmap of methylation values.

    Parameters
    ----------
    data : pd.DataFrame
        Methylation beta values.
    labels : np.ndarray
        Sample labels for annotation.
    top_n : int, default=50
        Number of top variable features to show.
    cluster_samples : bool, default=True
        Whether to cluster samples.
    cluster_features : bool, default=True
        Whether to cluster features.
    cmap : str, default="RdBu_r"
        Colormap name.
    figsize : Tuple[int, int], default=(12, 8)
        Figure size.
    output_path : str, optional
        Path to save figure.

    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
```

#### `plot_pca`

```python
def plot_pca(
    data: pd.DataFrame,
    labels: np.ndarray,
    batch: np.ndarray = None,
    n_components: int = 2,
    figsize: Tuple[int, int] = (10, 8),
    output_path: str = None
) -> plt.Figure:
    """
    Create PCA plot of samples.

    Parameters
    ----------
    data : pd.DataFrame
        Feature matrix.
    labels : np.ndarray
        Sample labels for coloring.
    batch : np.ndarray, optional
        Batch labels for shape encoding.
    n_components : int, default=2
        Number of PCA components.
    figsize : Tuple[int, int], default=(10, 8)
        Figure size.
    output_path : str, optional
        Path to save figure.

    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
```

#### `plot_batch_effects`

```python
def plot_batch_effects(
    data: pd.DataFrame,
    batch: np.ndarray,
    labels: np.ndarray,
    method: str = "pca",
    output_path: str = None
) -> plt.Figure:
    """
    Visualize batch effects in data.

    Parameters
    ----------
    data : pd.DataFrame
        Feature matrix.
    batch : np.ndarray
        Batch labels.
    labels : np.ndarray
        Biological labels.
    method : str, default="pca"
        Visualization method. Options: "pca", "tsne", "umap".
    output_path : str, optional
        Path to save figure.

    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
```

#### `plot_feature_importance`

```python
def plot_feature_importance(
    importance_scores: np.ndarray,
    feature_names: List[str],
    top_n: int = 20,
    orientation: str = "horizontal",
    figsize: Tuple[int, int] = (10, 8),
    output_path: str = None
) -> plt.Figure:
    """
    Plot feature importance scores.

    Parameters
    ----------
    importance_scores : np.ndarray
        Importance scores for each feature.
    feature_names : List[str]
        Feature names.
    top_n : int, default=20
        Number of top features to display.
    orientation : str, default="horizontal"
        Bar orientation. Options: "horizontal", "vertical".
    figsize : Tuple[int, int], default=(10, 8)
        Figure size.
    output_path : str, optional
        Path to save figure.

    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
```

#### `plot_enrichment`

```python
def plot_enrichment(
    results: "EnrichmentResults",
    plot_type: str = "dotplot",
    top_n: int = 15,
    color_by: str = "p_adjusted",
    size_by: str = "gene_count",
    figsize: Tuple[int, int] = (10, 8),
    output_path: str = None
) -> plt.Figure:
    """
    Plot enrichment analysis results.

    Parameters
    ----------
    results : EnrichmentResults
        Results from enrichment analysis.
    plot_type : str, default="dotplot"
        Plot type. Options: "dotplot", "barplot", "network".
    top_n : int, default=15
        Number of top terms to show.
    color_by : str, default="p_adjusted"
        Variable for color encoding.
    size_by : str, default="gene_count"
        Variable for size encoding.
    figsize : Tuple[int, int], default=(10, 8)
        Figure size.
    output_path : str, optional
        Path to save figure.

    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
```

---

## src.enrichment

Pathway and gene set enrichment analysis.

### Classes

#### `EnrichmentAnalyzer`

```python
class EnrichmentAnalyzer:
    """
    Unified interface for enrichment analysis.

    Parameters
    ----------
    primary_method : str, default="clusterprofiler"
        Primary enrichment method.
    fallback_method : str, default="msigdb_hypergeometric"
        Fallback method if primary fails.
    auto_fallback : bool, default=True
        Automatically use fallback on primary failure.
    verbose : bool, default=False
        Print diagnostic information.

    Attributes
    ----------
    primary_available : bool
        Whether primary method is available.
    fallback_available : bool
        Whether fallback method is available.

    Methods
    -------
    analyze(gene_list, organism="hsapiens", databases=None, **kwargs)
        Run enrichment analysis.
    compare(gene_lists, **kwargs)
        Compare enrichment across multiple gene lists.
    """

    def analyze(
        self,
        gene_list: List[str],
        organism: str = "hsapiens",
        databases: List[str] = None,
        background_genes: List[str] = None,
        p_cutoff: float = 0.05,
        q_cutoff: float = 0.1
    ) -> "EnrichmentResults":
        """Run enrichment analysis on gene list."""
```

#### `FallbackEnrichment`

```python
class FallbackEnrichment:
    """
    MSigDB-based fallback enrichment using hypergeometric test.

    Parameters
    ----------
    msigdb_path : str
        Path to MSigDB gene set files.
    organism : str, default="human"
        Organism for gene sets.

    Methods
    -------
    run(gene_list, collections, background_genes=None, p_cutoff=0.05)
        Run fallback enrichment analysis.
    load_gene_sets(collection)
        Load gene sets from MSigDB files.
    """
```

### Functions

#### `run_enrichment_analysis`

```python
def run_enrichment_analysis(
    gene_list: List[str],
    background: List[str] = None,
    organism: str = "hsapiens",
    databases: List[str] = None,
    p_cutoff: float = 0.05,
    q_cutoff: float = 0.1,
    method: str = "auto"
) -> "EnrichmentResults":
    """
    Run enrichment analysis with automatic method selection.

    Parameters
    ----------
    gene_list : List[str]
        List of gene symbols to analyze.
    background : List[str], optional
        Background gene list. If None, uses default background.
    organism : str, default="hsapiens"
        Organism identifier.
    databases : List[str], optional
        Databases to query. Default: ["GO_BP", "GO_MF", "GO_CC", "KEGG"].
    p_cutoff : float, default=0.05
        P-value cutoff for significance.
    q_cutoff : float, default=0.1
        FDR cutoff for significance.
    method : str, default="auto"
        Method to use. Options: "auto", "primary", "fallback".

    Returns
    -------
    EnrichmentResults
        Enrichment analysis results.
    """
```

#### `compare_enrichments`

```python
def compare_enrichments(
    gene_lists: Dict[str, List[str]],
    databases: List[str] = None,
    p_cutoff: float = 0.05
) -> "EnrichmentComparison":
    """
    Compare enrichment results across multiple gene lists.

    Parameters
    ----------
    gene_lists : Dict[str, List[str]]
        Dictionary mapping list names to gene lists.
    databases : List[str], optional
        Databases to query.
    p_cutoff : float, default=0.05
        P-value cutoff.

    Returns
    -------
    EnrichmentComparison
        Comparison results with visualization methods.
    """
```

---

## Data Classes

### `EnrichmentResults`

```python
@dataclass
class EnrichmentResults:
    """Container for enrichment analysis results."""

    results: List[EnrichmentResult]
    method_used: str
    fallback_used: bool
    query_genes: List[str]
    background_size: int

    def get_top_terms(
        self,
        n: int = 10,
        database: str = None
    ) -> pd.DataFrame:
        """Get top enriched terms."""

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""

    def save(self, path: str, format: str = "csv") -> None:
        """Save results to file."""
```

### `StabilityReport`

```python
@dataclass
class StabilityReport:
    """Report on feature selection stability across levels."""

    level_features: Dict[int, List[str]]
    stability_scores: Dict[str, float]

    def get_stable_features(self, min_levels: int) -> List[str]:
        """Get features stable across minimum number of levels."""

    def plot_stability(self, output_path: str = None) -> plt.Figure:
        """Plot stability across levels."""
```

---

## See Also

- [Quick Start Guide](../examples/quickstart.md)
- [Feature Selection Framework](../methodology/feature_selection.md)
- [Batch Correction Approach](../methodology/batch_correction.md)
- [Enrichment Analysis](../methodology/enrichment_analysis.md)
