# HIIT DNA Methylation Analysis

A comprehensive Python package for machine learning analysis of High-Intensity Interval Training (HIIT) intervention effects on DNA methylation patterns.

## 🧬 Overview

This package provides tools for analyzing epigenetic changes in response to HIIT interventions using state-of-the-art machine learning methods. It's designed for researchers studying the molecular mechanisms underlying exercise-induced health benefits.

## ✨ Features

- **Data Preprocessing**: Robust preprocessing pipeline for DNA methylation data
- **Machine Learning Models**: Classification and regression models optimized for methylation analysis
- **Feature Selection**: Multiple feature selection methods for high-dimensional epigenetic data
- **Validation Tools**: Comprehensive cross-validation and study design validation
- **Visualization**: Rich plotting functions for methylation patterns and model results
- **HIIT-Specific Metrics**: Specialized metrics for evaluating intervention responses

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/BrewSSS/hiit-dna-methylation-analysis.git
cd hiit-dna-methylation-analysis

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Basic Usage

```python
import pandas as pd
from hiit_methylation import (
    load_methylation_data, 
    HIITMethylationClassifier,
    plot_methylation_patterns
)

# Load your data
methylation_data, metadata = load_methylation_data('your_data.csv', 'metadata.csv')

# Preprocess data
from hiit_methylation.data import preprocess_methylation_data
processed_data = preprocess_methylation_data(
    methylation_data,
    normalize_method='logit',
    filter_variance=True
)

# Train a classifier
classifier = HIITMethylationClassifier(
    model_type='random_forest',
    feature_selection='univariate',
    n_features=1000
)

# Fit model (example: classify HIIT vs Control)
y_labels = (metadata['hiit_group'] == 'HIIT').astype(int)
classifier.fit(processed_data, y_labels)

# Make predictions
predictions = classifier.predict(processed_data)

# Visualize results
fig = plot_methylation_patterns(methylation_data, metadata)
fig.show()
```

### Example Analysis

Run the complete example:

```bash
python examples/basic_analysis_example.py
```

## 📊 Core Components

### Data Processing (`hiit_methylation.data`)

- `load_methylation_data()`: Load methylation data from various formats
- `preprocess_methylation_data()`: Comprehensive preprocessing pipeline
- `create_sample_data()`: Generate synthetic data for testing

### Machine Learning Models (`hiit_methylation.models`)

- `HIITMethylationClassifier`: Classification models for group/response prediction
- `HIITMethylationRegressor`: Regression models for continuous outcomes
- Support for Random Forest, SVM, Logistic Regression, Elastic Net

### Analysis Tools (`hiit_methylation.utils`)

- `methylation_metrics()`: Specialized evaluation metrics
- `cross_validate_methylation()`: Robust cross-validation
- `calculate_differential_methylation()`: Statistical testing for group differences
- `hiit_response_score()`: Composite response scoring

### Visualization (`hiit_methylation.visualization`)

- `plot_methylation_patterns()`: Overview of methylation distributions
- `plot_hiit_effects()`: Visualization of intervention effects
- `methylation_heatmap()`: Clustered heatmaps with annotations
- `plot_model_performance()`: Model evaluation plots

## 🔬 Scientific Applications

### 1. HIIT Response Prediction
Predict which individuals will respond positively to HIIT interventions based on baseline methylation patterns.

### 2. Intervention Effect Analysis
Identify CpG sites that show significant methylation changes following HIIT training.

### 3. Biomarker Discovery
Discover epigenetic biomarkers associated with fitness improvements and metabolic benefits.

### 4. Longitudinal Analysis
Analyze methylation trajectories over the course of HIIT interventions.

## 📈 Model Types and Use Cases

| Model Type | Use Case | Input | Output |
|------------|----------|-------|---------|
| **Classification** | Group assignment, Response prediction | Methylation β-values | HIIT/Control, Responder/Non-responder |
| **Regression** | Fitness prediction, Change quantification | Baseline methylation + clinical | Fitness score, Methylation change |

## 🛠 Advanced Features

### Feature Selection Methods
- **Variance Threshold**: Remove low-variance CpG sites
- **Univariate Selection**: Statistical tests (F-test, mutual information)
- **L1 Regularization**: Lasso-based feature selection

### Validation Strategies
- **Stratified Cross-Validation**: Maintains group balance
- **Temporal Validation**: Subject-based splits for longitudinal studies
- **Batch Effect Assessment**: Detection and correction of technical artifacts

### Data Quality Control
- Missing value imputation
- Outlier detection
- Beta value range validation
- Sample and CpG site filtering

## 📋 Requirements

- Python ≥ 3.7
- NumPy ≥ 1.21.0
- Pandas ≥ 1.3.0
- Scikit-learn ≥ 1.0.0
- Matplotlib ≥ 3.4.0
- Seaborn ≥ 0.11.0
- SciPy ≥ 1.7.0

See `requirements.txt` for complete dependencies.

## 📖 Documentation

### Key Classes

#### `HIITMethylationClassifier`
```python
classifier = HIITMethylationClassifier(
    model_type='random_forest',           # 'random_forest', 'svm', 'logistic'
    hyperparameter_tuning=False,          # Enable grid search
    feature_selection='univariate',       # Feature selection method
    n_features=1000,                      # Number of features to select
    random_state=42
)
```

#### `HIITMethylationRegressor`
```python
regressor = HIITMethylationRegressor(
    model_type='random_forest',           # 'random_forest', 'svm', 'elastic_net', 'ridge'
    hyperparameter_tuning=False,
    feature_selection='univariate',
    n_features=1000,
    random_state=42
)
```

### Data Format Requirements

**Methylation Data**: Samples × CpG sites matrix with β-values (0-1 range)
```
             cg00000001  cg00000002  ...
Sample_001   0.234       0.567       ...
Sample_002   0.123       0.789       ...
```

**Metadata**: Sample information with required columns
```
             hiit_group  time_point  subject_id  age  gender
Sample_001   HIIT        Pre         S001        25   M
Sample_002   Control     Pre         S002        28   F
```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for:

- Code style and standards
- Testing requirements
- Documentation guidelines
- Issue reporting

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 Citation

If you use this package in your research, please cite:

```bibtex
@software{hiit_methylation_analysis,
  title={HIIT DNA Methylation Analysis: Machine Learning Tools for Exercise Epigenetics},
  author={HIIT Methylation Analysis Team},
  year={2024},
  url={https://github.com/BrewSSS/hiit-dna-methylation-analysis}
}
```

## 🔗 Related Resources

- [DNA Methylation Analysis Guidelines](https://www.nature.com/articles/nrg.2016.133)
- [Exercise Epigenetics Research](https://www.nature.com/articles/s41576-019-0103-3)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/BrewSSS/hiit-dna-methylation-analysis/issues)
- **Documentation**: [Wiki](https://github.com/BrewSSS/hiit-dna-methylation-analysis/wiki)
- **Email**: hiit-analysis@example.com

---

**Keywords**: bioinformatics, DNA methylation, HIIT, machine learning, epigenetics, exercise, sklearn, python
