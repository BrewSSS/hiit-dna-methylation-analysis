# Data Directory

This directory contains the DNA methylation data used in the HIIT-Methylation-Classification pipeline.

## Directory Structure

```
data/
├── raw/                    # Raw, unprocessed data
│   └── GSE171140/          # GEO dataset files
├── processed/              # Preprocessed data
│   ├── methyl_data_preprocessed.pkl
│   ├── sample_mapping.csv
│   └── features/           # Selected feature sets
├── external/               # External reference data
│   ├── annotation/         # Array annotation files
│   └── msigdb/             # Gene set databases
└── README.md               # This file
```

## Data Source

### GEO Accession: GSE171140

- **Title**: DNA Methylation Response to High-Intensity Interval Training
- **Platform**: Illumina EPIC 850K Methylation Array
- **Organism**: Homo sapiens
- **Samples**: Peripheral blood samples at multiple time points

## Downloading the Data

### Method 1: Programmatic Download

```python
from src.data import GEODataLoader

# Initialize loader
loader = GEODataLoader(cache_dir="data/raw/")

# Download GSE171140
loader.download("GSE171140")

# Verify download
print(f"Downloaded files: {loader.list_files()}")
```

### Method 2: Manual Download

1. Visit the GEO page: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE171140

2. Download the following files:
   - Series Matrix File (contains beta values)
   - Platform annotation file

3. Place files in `data/raw/GSE171140/`

### Method 3: Using GEOquery (R)

```r
library(GEOquery)

# Download dataset
gse <- getGEO("GSE171140", destdir = "data/raw/")

# Extract expression data
beta_values <- exprs(gse[[1]])
```

## Expected Data Structure

After downloading, your `data/raw/` directory should contain:

```
data/raw/
└── GSE171140/
    ├── GSE171140_series_matrix.txt.gz
    ├── GSE171140_sample_info.csv
    └── GPL21145_annotation.txt
```

## Preprocessing Requirements

Before running the pipeline, the raw data must be preprocessed. Key steps include:

### 1. Quality Control
- Detection p-value filtering (p < 0.01)
- Bead count filtering (minimum 3 beads)
- Sample quality assessment

### 2. Probe Filtering
- Remove probes on sex chromosomes (X, Y)
- Remove probes with SNPs at CpG site
- Remove cross-reactive probes

### 3. Normalization
- Quantile normalization (default)
- Alternative: BMIQ, SWAN, or functional normalization

### 4. Missing Value Imputation
- KNN imputation (default)
- Alternative: mean or median imputation

## Running Preprocessing

```python
from src.data import load_methylation_data, preprocess_data

# Load raw data
raw_data = load_methylation_data("data/raw/GSE171140/")

# Preprocess with default settings
processed_data = preprocess_data(
    raw_data,
    normalize=True,
    remove_sex_chromosomes=True,
    remove_snp_probes=True,
    impute_missing=True
)

# Save processed data
processed_data.to_pickle("data/processed/methyl_data_preprocessed.pkl")
```

## File Formats

### Raw Data
- **Series Matrix**: Tab-delimited text file with beta values
- **Sample Info**: CSV with sample metadata

### Processed Data
- **Pickle files**: Python pickle format for efficient loading
- **CSV files**: For compatibility with other tools

### Annotation Files
- **GPL21145**: Illumina EPIC array annotation
- **MSigDB GMT files**: Gene set definitions

## Data Privacy Note

This pipeline uses publicly available data from GEO. When using with your own data:

1. Ensure proper ethical approval
2. De-identify samples as needed
3. Follow institutional data handling policies

## External Data Requirements

For full pipeline functionality, download these additional files:

### Array Annotation
```bash
# Download EPIC array annotation (if not using minfi)
wget -O data/external/annotation/epic_annotation.csv \
  "https://example.com/EPIC_annotation.csv"
```

### MSigDB Gene Sets (for fallback enrichment)
```bash
# Requires MSigDB registration
# Place .gmt files in data/external/msigdb/
```

## Troubleshooting

### Large File Handling

For memory-efficient processing of large files:

```python
from src.data import load_methylation_data

# Load in chunks
data = load_methylation_data(
    "data/raw/",
    chunk_size=100000  # Process 100K probes at a time
)
```

### Missing Files

If files are missing after download:
1. Check internet connection
2. Verify GEO accession number
3. Try alternative download method

### Format Issues

If data format is unexpected:
1. Check file is properly decompressed
2. Verify column delimiters
3. Check for header rows

## See Also

- [Quick Start Guide](../docs/examples/quickstart.md)
- [Preprocessing Documentation](../docs/methodology/feature_selection.md)
- [API Reference](../docs/api/reference.md)
