# HIIT-Methylation-Classification

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive machine learning pipeline for DNA methylation-based classification of High-Intensity Interval Training (HIIT) intervention responses. This project implements novel methodological approaches for epigenetic biomarker discovery and exercise response prediction.

## Overview

This pipeline analyzes DNA methylation data to:
1. **Binary Classification**: Distinguish between pre- and post-HIIT intervention samples
2. **Multiclass Classification**: Classify intervention duration (4/8/12 weeks)
3. **Biomarker Discovery**: Identify methylation sites associated with HIIT response

## Key Methodological Innovations

### 1. Ten-Level Feature Selection Framework
A hierarchical feature selection approach using progressively stringent thresholds (L1-L10), enabling systematic identification of robust biomarkers while balancing sensitivity and specificity.

### 2. Triple Analysis Strategy
Comprehensive analysis through three complementary approaches:
- **Binary Analysis**: Pre vs. Post intervention discrimination
- **Multiclass Analysis**: Intervention duration classification
- **Time-series Analysis**: Temporal methylation trajectory modeling

### 3. Batch-Aware Modeling
Novel approach treating batch effects as covariates rather than removing them, preserving biological signal while accounting for technical variation.

### 4. Multi-Version Data Comparison
Systematic comparison across different preprocessing strategies and normalization methods to ensure robust and reproducible results.

### 5. Fallback Enrichment Methodology
Robust pathway enrichment analysis with primary (clusterProfiler) and fallback (MSigDB + hypergeometric test) strategies to ensure comprehensive functional annotation.

## Installation

### Option 1: Conda Environment (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/HIIT-Methylation-Classification.git
cd HIIT-Methylation-Classification

# Create and activate conda environment
conda env create -f environment.yml
conda activate dna-methyl

# Install package in development mode
pip install -e .
```

### Option 2: pip Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/HIIT-Methylation-Classification.git
cd HIIT-Methylation-Classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## Quick Start

```python
from src.data import load_methylation_data, preprocess_data
from src.features import FeatureSelector
from src.models import HIITClassifier

# Load and preprocess data
data = load_methylation_data("data/raw/")
processed_data = preprocess_data(data, normalize=True)

# Feature selection with Ten-Level Framework
selector = FeatureSelector(method="ten_level", level=5)
selected_features = selector.fit_transform(processed_data)

# Train classifier with batch-aware modeling
classifier = HIITClassifier(
    model_type="random_forest",
    batch_aware=True
)
classifier.fit(selected_features, labels)

# Evaluate
results = classifier.evaluate(test_data, test_labels)
```

## Project Structure

```
HIIT-Methylation-Classification/
├── data/                      # Data directory
│   ├── raw/                   # Raw methylation data
│   ├── processed/             # Preprocessed data
│   └── external/              # External reference data
├── models/                    # Trained model artifacts
│   ├── binary/                # Binary classification models
│   └── multiclass/            # Multiclass models
├── notebooks/                 # Jupyter notebooks
│   ├── 01_data_acquisition.ipynb
│   ├── 02_preprocessing_feature_selection.ipynb
│   ├── 03_modeling_evaluation.ipynb
│   └── 04_further_analysis.ipynb
├── src/                       # Source code modules
│   ├── data/                  # Data loading and processing
│   ├── features/              # Feature engineering
│   ├── models/                # Model implementations
│   ├── visualization/         # Plotting utilities
│   └── enrichment/            # Pathway enrichment analysis
├── scripts/                   # Utility scripts
│   ├── enrichment/            # Enrichment analysis scripts
│   ├── visualization/         # Figure generation
│   └── utilities/             # Helper scripts
├── results/                   # Analysis outputs
│   ├── figures/               # Generated figures
│   ├── tables/                # Result tables
│   └── reports/               # Analysis reports
├── docs/                      # Documentation
│   ├── methodology/           # Method descriptions
│   ├── api/                   # API reference
│   └── examples/              # Usage examples
├── environment.yml            # Conda environment
├── requirements.txt           # pip dependencies
├── setup.py                   # Package setup
└── README.md                  # This file
```

## Documentation

- [Quick Start Guide](docs/examples/quickstart.md)
- [Feature Selection Framework](docs/methodology/feature_selection.md)
- [Batch Correction Approach](docs/methodology/batch_correction.md)
- [Enrichment Analysis](docs/methodology/enrichment_analysis.md)
- [API Reference](docs/api/reference.md)

## Data

This project uses data from the Gene Expression Omnibus (GEO):
- **Dataset**: GSE171140
- **Platform**: Illumina EPIC 850K methylation array

See [data/README.md](data/README.md) for data acquisition instructions.

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@article{hiit_methylation_2024,
  title={DNA Methylation-Based Classification of HIIT Intervention Response:
         A Ten-Level Feature Selection Framework},
  author={[Authors]},
  journal={[Journal - Under Review]},
  year={2024},
  note={Manuscript under review}
}
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Gene Expression Omnibus (GEO) for hosting the original dataset
- The scientific community for developing the bioinformatics tools used in this pipeline
