# Ten-Level Feature Selection Framework

This document describes the Ten-Level Feature Selection Framework, a novel hierarchical approach for identifying robust DNA methylation biomarkers in the HIIT-Methylation-Classification pipeline.

## Overview

The Ten-Level Feature Selection Framework addresses a fundamental challenge in high-dimensional methylation data analysis: balancing the need for comprehensive biomarker discovery with the requirement for robust, reproducible features. Traditional single-threshold approaches often miss important biological signals or include too much noise.

Our framework implements a hierarchical selection strategy with ten progressively stringent thresholds, allowing researchers to systematically explore the sensitivity-specificity trade-off.

## Conceptual Foundation

### The Problem

DNA methylation arrays (e.g., Illumina EPIC 850K) measure hundreds of thousands of CpG sites simultaneously. Selecting informative features requires:

1. **Statistical significance**: Features must show meaningful differences between conditions
2. **Effect size**: Differences must be biologically meaningful
3. **Reproducibility**: Features should be stable across different samples and conditions
4. **Interpretability**: The number of features should be manageable for downstream analysis

### The Solution

Rather than committing to a single threshold, our framework provides ten predefined levels that researchers can explore systematically. This approach:

- Enables sensitivity analysis across different stringency levels
- Facilitates comparison of results across studies using the same framework
- Provides a standardized vocabulary for describing feature selection stringency

## Level Definitions

### Level Thresholds

| Level | Variance Threshold | Differential Threshold | Combined Criteria | Expected Feature Count |
|-------|-------------------|------------------------|-------------------|------------------------|
| L1    | Top 50%           | p < 0.1                | Union             | Very High              |
| L2    | Top 40%           | p < 0.05               | Union             | High                   |
| L3    | Top 30%           | p < 0.05               | Intersection      | Moderate-High          |
| L4    | Top 25%           | p < 0.01               | Intersection      | Moderate               |
| L5    | Top 20%           | p < 0.01, |delta| > 0.05 | Intersection    | Moderate               |
| L6    | Top 15%           | p < 0.005, |delta| > 0.05 | Intersection   | Moderate-Low           |
| L7    | Top 10%           | p < 0.001, |delta| > 0.1 | Intersection    | Low                    |
| L8    | Top 5%            | p < 0.001, |delta| > 0.1 | Intersection    | Very Low               |
| L9    | Top 2.5%          | FDR < 0.01, |delta| > 0.15 | Intersection  | Minimal                |
| L10   | Top 1%            | FDR < 0.001, |delta| > 0.2 | Intersection  | Highly Selective       |

### Parameter Descriptions

- **Variance Threshold**: Percentile cutoff for feature variance across all samples
- **Differential Threshold**: Statistical significance threshold for differential methylation
- **|delta|**: Absolute difference in mean beta values between conditions
- **FDR**: False Discovery Rate (Benjamini-Hochberg corrected p-values)
- **Combined Criteria**: How variance and differential criteria are combined
  - *Union*: Features meeting either criterion
  - *Intersection*: Features meeting both criteria

## Three-Pronged Approach

The framework is applied across three complementary analytical perspectives:

### 1. Binary Analysis Features

Features selected for distinguishing pre- vs. post-intervention samples:

```python
from src.features import FeatureSelector

# Binary feature selection
binary_selector = FeatureSelector(
    method="ten_level",
    level=5,
    analysis_type="binary"
)
binary_features = binary_selector.fit_transform(
    data,
    labels=binary_labels  # 0: pre, 1: post
)
```

**Use case**: Identifying biomarkers that change with any HIIT intervention, regardless of duration.

### 2. Multiclass Analysis Features

Features selected for distinguishing intervention duration:

```python
# Multiclass feature selection
multiclass_selector = FeatureSelector(
    method="ten_level",
    level=5,
    analysis_type="multiclass"
)
multiclass_features = multiclass_selector.fit_transform(
    data,
    labels=duration_labels  # 0: 4wk, 1: 8wk, 2: 12wk
)
```

**Use case**: Identifying biomarkers that differentiate between intervention durations.

### 3. Time-Series Analysis Features

Features selected based on temporal trajectory patterns:

```python
# Time-series feature selection
temporal_selector = FeatureSelector(
    method="ten_level",
    level=5,
    analysis_type="temporal"
)
temporal_features = temporal_selector.fit_transform(
    data,
    time_points=[0, 4, 8, 12],  # weeks
    subject_ids=subject_ids
)
```

**Use case**: Identifying biomarkers with consistent temporal patterns across individuals.

## Integration Strategy

### Consensus Features

Features identified across multiple analytical approaches may represent the most robust biomarkers:

```python
from src.features import get_consensus_features

# Find features selected in all three approaches
consensus = get_consensus_features(
    binary_features=binary_features.columns,
    multiclass_features=multiclass_features.columns,
    temporal_features=temporal_features.columns,
    min_approaches=2  # Require selection in at least 2 approaches
)
```

### Level Stability Analysis

Assess feature robustness by examining selection across multiple levels:

```python
from src.features import analyze_level_stability

# Check which features are stable across levels
stability_report = analyze_level_stability(
    data=data,
    labels=labels,
    levels=[3, 4, 5, 6, 7],  # Test range of levels
    analysis_type="binary"
)

# Features selected at all tested levels are most robust
robust_features = stability_report.get_stable_features(min_levels=4)
```

## Usage Examples

### Basic Usage

```python
from src.features import FeatureSelector

# Initialize selector with desired level
selector = FeatureSelector(method="ten_level", level=5)

# Fit and transform data
selected_data = selector.fit_transform(X, y)

# Access selected feature names
selected_features = selector.selected_features_

# Access selection statistics
selection_stats = selector.get_selection_stats()
```

### Comparing Multiple Levels

```python
from src.features import FeatureSelector
import pandas as pd

results = {}
for level in range(1, 11):
    selector = FeatureSelector(method="ten_level", level=level)
    selected = selector.fit_transform(X, y)
    results[f"L{level}"] = {
        "n_features": len(selector.selected_features_),
        "features": selector.selected_features_
    }

# Create comparison table
comparison_df = pd.DataFrame([
    {"Level": k, "N_Features": v["n_features"]}
    for k, v in results.items()
])
```

### Custom Thresholds

For advanced users who need custom threshold configurations:

```python
from src.features import FeatureSelector

# Define custom level parameters
custom_params = {
    "variance_percentile": 0.15,
    "p_threshold": 0.005,
    "delta_threshold": 0.08,
    "use_fdr": True,
    "combine_method": "intersection"
}

selector = FeatureSelector(
    method="ten_level",
    level="custom",
    custom_params=custom_params
)
```

## Best Practices

### Choosing the Right Level

1. **Exploratory Analysis**: Start with L3-L5 for initial exploration
2. **Biomarker Discovery**: Use L5-L7 for balanced sensitivity/specificity
3. **Validation Studies**: Use L7-L10 for highly confident feature sets
4. **Cross-Study Comparison**: Report results at multiple levels (e.g., L5 and L8)

### Reporting Guidelines

When publishing results using this framework, please report:

1. The level(s) used for primary analysis
2. The number of features selected at each level
3. Any deviations from default level parameters
4. Stability analysis across adjacent levels

### Reproducibility

To ensure reproducibility:

```python
from src.features import FeatureSelector

selector = FeatureSelector(
    method="ten_level",
    level=5,
    random_state=42  # Set for reproducibility
)

# Export configuration for reproducibility
config = selector.get_config()
config.to_json("feature_selection_config.json")
```

## Theoretical Background

### Statistical Foundation

The framework integrates multiple statistical approaches:

1. **Variance-based filtering**: Removes low-variance features that are unlikely to be informative
2. **Differential analysis**: Identifies features with significant between-group differences
3. **Effect size filtering**: Ensures differences are biologically meaningful
4. **Multiple testing correction**: Controls false discovery rate at stringent levels

### Relationship to Other Methods

| Method | Relationship to Ten-Level Framework |
|--------|-------------------------------------|
| limma | Differential analysis component uses similar linear modeling |
| Variance filtering | Incorporated as first filtering step |
| LASSO | Can be applied after ten-level selection for further refinement |
| Random Forest importance | Complementary approach for validation |

## References

For methodological details and validation, please refer to the main manuscript (citation pending).

## See Also

- [Batch Correction Approach](batch_correction.md)
- [Enrichment Analysis](enrichment_analysis.md)
- [API Reference](../api/reference.md)
