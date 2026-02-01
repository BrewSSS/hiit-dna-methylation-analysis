# Quick Start Guide

This guide will help you get started with the HIIT-Methylation-Classification pipeline.

## Prerequisites

- Python 3.9 or higher
- Conda (recommended) or pip
- Git

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/HIIT-Methylation-Classification.git
cd HIIT-Methylation-Classification
```

### Step 2: Set Up Environment

#### Option A: Using Conda (Recommended)

```bash
# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate dna-methyl

# Install package in development mode
pip install -e .
```

#### Option B: Using pip

```bash
# Create virtual environment
python -m venv venv

# Activate environment
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Step 3: Verify Installation

```python
# Test import
python -c "from src.data import load_methylation_data; print('Installation successful!')"
```

## Data Acquisition

### Downloading from GEO

The pipeline uses data from GEO accession GSE171140. You can download it programmatically:

```python
from src.data import GEODataLoader

# Initialize loader
loader = GEODataLoader(cache_dir="data/raw/")

# Download dataset
loader.download("GSE171140")

# Load data
data = loader.load_series_matrix()
metadata = loader.get_sample_metadata()

print(f"Loaded {data.shape[0]} samples with {data.shape[1]} CpG sites")
```

### Manual Download

Alternatively, download manually from NCBI GEO:
1. Visit https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE171140
2. Download the series matrix file
3. Place in `data/raw/` directory

## Running the Pipeline

### Step 1: Data Preprocessing

```python
from src.data import load_methylation_data, preprocess_data

# Load raw data
data, metadata = load_methylation_data(
    "data/raw/",
    return_metadata=True
)

# Preprocess
processed_data = preprocess_data(
    data,
    normalize=True,
    normalization_method="quantile",
    remove_sex_chromosomes=True,
    remove_snp_probes=True,
    impute_missing=True
)

# Save processed data
processed_data.to_pickle("data/processed/methyl_data_preprocessed.pkl")

print(f"Preprocessed data shape: {processed_data.shape}")
```

### Step 2: Feature Selection

```python
from src.features import FeatureSelector
import numpy as np

# Prepare labels (example: binary classification)
# Adjust based on your metadata structure
labels = np.array([...])  # Your sample labels

# Initialize Ten-Level Feature Selector
selector = FeatureSelector(
    method="ten_level",
    level=5,
    analysis_type="binary"
)

# Select features
selected_features = selector.fit_transform(processed_data, labels)

print(f"Selected {len(selector.selected_features_)} features")

# View selection statistics
stats = selector.get_selection_stats()
print(f"Variance threshold: {stats['variance_threshold']}")
print(f"P-value threshold: {stats['p_value_threshold']}")
```

### Step 3: Model Training

```python
from src.models import HIITClassifier
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    selected_features, labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# Initialize classifier
classifier = HIITClassifier(
    model_type="random_forest",
    batch_aware=True,
    class_weight="balanced",
    random_state=42
)

# Train
classifier.fit(X_train, y_train)

# Evaluate
results = classifier.evaluate(X_test, y_test)

print(f"Accuracy: {results['accuracy']:.3f}")
print(f"AUC-ROC: {results['auc_roc']:.3f}")
```

### Step 4: Cross-Validation

```python
# Perform cross-validation
cv_results = classifier.cross_validate(
    X=selected_features,
    y=labels,
    cv_strategy="stratified",
    n_splits=5
)

print(f"CV Accuracy: {cv_results['accuracy_mean']:.3f} +/- {cv_results['accuracy_std']:.3f}")
print(f"CV AUC: {cv_results['auc_mean']:.3f} +/- {cv_results['auc_std']:.3f}")
```

### Step 5: Feature Importance

```python
from src.visualization import plot_feature_importance

# Get feature importance
importance = classifier.get_feature_importance()

# Plot top features
plot_feature_importance(
    importance_scores=importance['scores'],
    feature_names=importance['names'],
    top_n=20,
    output_path="results/figures/feature_importance.png"
)
```

### Step 6: Enrichment Analysis

```python
from src.enrichment import EnrichmentAnalyzer
from src.data import map_cpg_to_genes

# Map CpG sites to genes
genes = map_cpg_to_genes(selector.selected_features_)

print(f"Mapped to {len(genes)} unique genes")

# Run enrichment analysis
analyzer = EnrichmentAnalyzer()
enrichment_results = analyzer.analyze(
    gene_list=genes,
    databases=["GO_BP", "GO_MF", "KEGG"],
    p_cutoff=0.05
)

# View top results
top_terms = enrichment_results.get_top_terms(n=10, database="GO_BP")
print(top_terms)

# Save results
enrichment_results.save("results/enrichment/go_bp_results.csv")
```

## Using Jupyter Notebooks

The pipeline includes pre-configured notebooks for each analysis step:

```bash
# Start Jupyter Lab
jupyter lab
```

Navigate to `notebooks/` and run in order:
1. `01_data_acquisition.ipynb` - Download and inspect data
2. `02_preprocessing_feature_selection.ipynb` - Preprocess and select features
3. `03_modeling_evaluation.ipynb` - Train and evaluate models
4. `04_further_analysis.ipynb` - Enrichment and interpretation

## Expected Outputs

After running the pipeline, you should have:

### Data Directory
```
data/
├── raw/
│   └── GSE171140_series_matrix.txt
├── processed/
│   ├── methyl_data_preprocessed.pkl
│   └── sample_mapping.csv
└── external/
    └── annotation_files/
```

### Results Directory
```
results/
├── figures/
│   ├── pca_plot.png
│   ├── feature_importance.png
│   └── enrichment_dotplot.png
├── tables/
│   ├── selected_features.csv
│   ├── model_performance.csv
│   └── enrichment_results.csv
└── models/
    └── binary_classifier.pkl
```

### Model Artifacts
```
models/
├── binary/
│   ├── random_forest_model.pkl
│   └── feature_selector.pkl
└── multiclass/
    └── gradient_boosting_model.pkl
```

## Troubleshooting

### Common Issues

1. **Import Error: Module not found**
   ```bash
   # Ensure package is installed
   pip install -e .
   ```

2. **Memory Error during preprocessing**
   ```python
   # Process in chunks
   processed_data = preprocess_data(data, chunk_size=10000)
   ```

3. **clusterProfiler not available**
   - The pipeline will automatically use the fallback enrichment method
   - For full functionality, install R and clusterProfiler

4. **Slow feature selection**
   ```python
   # Use parallel processing
   selector = FeatureSelector(n_jobs=-1)
   ```

## Next Steps

- Explore [Feature Selection Framework](../methodology/feature_selection.md) for advanced options
- Learn about [Batch-Aware Modeling](../methodology/batch_correction.md)
- Check [API Reference](../api/reference.md) for detailed function documentation

## Getting Help

- Check the [documentation](../README.md)
- Open an issue on GitHub
- Review the [CONTRIBUTING guide](../../CONTRIBUTING.md)
