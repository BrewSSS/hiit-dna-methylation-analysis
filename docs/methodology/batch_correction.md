# Batch-Aware Modeling Approach

This document describes the Batch-Aware Modeling approach used in the HIIT-Methylation-Classification pipeline, a novel strategy for handling batch effects while preserving biological signal.

## Overview

Batch effects are systematic technical variations that arise from processing samples in different batches, on different days, or using different reagent lots. In DNA methylation studies, batch effects can confound biological signals and lead to spurious findings.

Traditional approaches remove batch effects entirely through correction methods like ComBat. However, this can inadvertently remove true biological variation that correlates with batch structure. Our Batch-Aware Modeling approach takes a different strategy: treating batch as a covariate rather than removing it entirely.

## The Problem with Batch Removal

### Traditional Batch Correction

Standard batch correction methods (e.g., ComBat, SVA) aim to remove all batch-associated variation:

```
Corrected_Data = Original_Data - Batch_Effects
```

**Limitations:**
1. **Over-correction risk**: If biological groups are confounded with batches, correction removes real signal
2. **Information loss**: Batch effects may capture real technical sensitivity differences
3. **Assumption violation**: Many methods assume batch is independent of biological variables
4. **Reproducibility concerns**: Corrected data depends on all samples in the correction

### When Batch Removal Fails

Consider a study where:
- Batch 1: Processed Week 0 and Week 4 samples
- Batch 2: Processed Week 8 and Week 12 samples

Traditional batch correction would remove variation between Batch 1 and Batch 2, potentially eliminating the very temporal signal we aim to detect.

## Batch-Aware Modeling Strategy

### Conceptual Framework

Instead of removing batch effects, we explicitly model them as covariates:

```
Y = f(Biological_Variables) + g(Batch) + error
```

This approach:
1. Preserves all biological variation
2. Accounts for batch effects in predictions
3. Provides interpretable batch coefficients
4. Allows batch-stratified validation

### Implementation Overview

```python
from src.models import HIITClassifier

# Initialize with batch-aware modeling
classifier = HIITClassifier(
    model_type="random_forest",
    batch_aware=True,
    batch_strategy="covariate"  # Include batch as feature
)

# Fit with batch information
classifier.fit(
    X=features,
    y=labels,
    batch=batch_labels  # Batch identifiers
)
```

## Implementation Details

### Strategy 1: Batch as Covariate

Include batch indicators as additional features in the model:

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def add_batch_covariates(X, batch_labels):
    """Add one-hot encoded batch indicators to feature matrix."""
    encoder = OneHotEncoder(sparse=False, drop='first')
    batch_encoded = encoder.fit_transform(batch_labels.reshape(-1, 1))

    batch_df = pd.DataFrame(
        batch_encoded,
        columns=[f"batch_{i}" for i in range(batch_encoded.shape[1])],
        index=X.index
    )

    return pd.concat([X, batch_df], axis=1)
```

**Advantages:**
- Simple to implement
- Works with any classifier
- Explicitly models batch effects

**Considerations:**
- Increases feature dimensionality
- Requires batch information at prediction time

### Strategy 2: Batch-Stratified Training

Ensure each cross-validation fold contains samples from all batches:

```python
from sklearn.model_selection import StratifiedGroupKFold

def batch_stratified_cv(X, y, batch, n_splits=5):
    """Create CV folds stratified by both label and batch."""
    # Combine label and batch for stratification
    stratify_labels = [f"{yi}_{bi}" for yi, bi in zip(y, batch)]

    cv = StratifiedGroupKFold(n_splits=n_splits)

    for train_idx, test_idx in cv.split(X, stratify_labels, groups=batch):
        yield train_idx, test_idx
```

**Advantages:**
- Ensures generalization across batches
- Detects batch-confounded features
- Validates batch robustness

### Strategy 3: Batch-Weighted Loss

Weight samples inversely to batch size to prevent batch dominance:

```python
import numpy as np

def compute_batch_weights(batch_labels):
    """Compute sample weights inversely proportional to batch size."""
    unique, counts = np.unique(batch_labels, return_counts=True)
    batch_weights = {b: 1.0 / c for b, c in zip(unique, counts)}

    # Normalize weights
    total = sum(batch_weights.values())
    batch_weights = {b: w / total * len(unique) for b, w in batch_weights.items()}

    return np.array([batch_weights[b] for b in batch_labels])
```

### Strategy 4: Domain Adaptation

For advanced applications, use domain adaptation techniques:

```python
from src.models import DomainAdaptiveClassifier

classifier = DomainAdaptiveClassifier(
    base_model="gradient_boosting",
    adaptation_method="coral",  # Correlation alignment
    source_domains=training_batches,
    target_domains=test_batches
)
```

## Integration with Classification Pipeline

### Complete Workflow

```python
from src.data import load_methylation_data, extract_batch_info
from src.features import FeatureSelector
from src.models import HIITClassifier

# Load data with batch information
data, metadata = load_methylation_data("data/raw/", return_metadata=True)
batch_labels = extract_batch_info(metadata)

# Feature selection (batch-aware)
selector = FeatureSelector(
    method="ten_level",
    level=5,
    batch_aware=True
)
selected_features = selector.fit_transform(data, labels, batch=batch_labels)

# Classification with batch modeling
classifier = HIITClassifier(
    model_type="random_forest",
    batch_aware=True,
    batch_strategy="covariate"
)

# Batch-stratified cross-validation
from sklearn.model_selection import cross_val_score

cv_scores = classifier.cross_validate(
    X=selected_features,
    y=labels,
    batch=batch_labels,
    cv_strategy="batch_stratified",
    n_splits=5
)
```

### Prediction with New Batches

When applying the model to new data from unseen batches:

```python
# Option 1: Treat new batch as reference category
new_data_with_batch = add_batch_covariates(
    new_data,
    batch_labels=np.array(["new_batch"] * len(new_data)),
    encoder=classifier.batch_encoder_  # Use fitted encoder
)
predictions = classifier.predict(new_data_with_batch)

# Option 2: Use batch-invariant prediction
predictions = classifier.predict(
    new_data,
    batch_invariant=True  # Ignore batch features
)
```

## Batch Effect Diagnostics

### Assessing Batch Effects

```python
from src.visualization import plot_batch_effects

# PCA visualization colored by batch
plot_batch_effects(
    data=selected_features,
    batch=batch_labels,
    labels=labels,
    method="pca",
    output_path="figures/batch_pca.png"
)

# Quantify batch effect magnitude
from src.data import calculate_batch_metrics

batch_metrics = calculate_batch_metrics(
    data=selected_features,
    batch=batch_labels,
    labels=labels
)

print(f"Batch variance explained: {batch_metrics['variance_explained']:.2%}")
print(f"Label-batch correlation: {batch_metrics['label_batch_correlation']:.3f}")
```

### Feature-Level Batch Assessment

```python
from src.features import assess_feature_batch_association

# Check each feature for batch association
batch_assessment = assess_feature_batch_association(
    data=selected_features,
    batch=batch_labels,
    method="anova"
)

# Flag features strongly associated with batch
batch_confounded = batch_assessment[
    batch_assessment['batch_pvalue'] < 0.001
]['feature'].tolist()

print(f"Features confounded with batch: {len(batch_confounded)}")
```

## Comparison with Traditional Methods

### When to Use Batch-Aware Modeling

| Scenario | Recommendation |
|----------|----------------|
| Batch independent of biology | Either approach works |
| Partial batch-biology confounding | Batch-aware modeling preferred |
| Strong batch-biology confounding | Batch-aware modeling required |
| Single batch study | Not applicable |
| Prediction on new batches | Batch-aware with invariant prediction |

### Empirical Comparison Framework

```python
from src.models import compare_batch_strategies

comparison = compare_batch_strategies(
    data=features,
    labels=labels,
    batch=batch_labels,
    strategies=["none", "combat", "covariate", "stratified"],
    cv_splits=5,
    metrics=["accuracy", "auc", "f1"]
)

# Visualize comparison
comparison.plot_comparison("figures/batch_strategy_comparison.png")
```

## Best Practices

### Experimental Design

1. **Randomize samples across batches** when possible
2. **Include technical replicates** to quantify batch variation
3. **Document batch information** (date, operator, reagent lot)
4. **Balance biological groups** across batches

### Analysis Recommendations

1. **Always assess batch effects** before analysis
2. **Try multiple strategies** and compare results
3. **Report batch handling** in methods section
4. **Validate across batches** when possible

### Red Flags

Watch for these warning signs:
- Perfect separation of batches in PCA
- Model performance drops on held-out batches
- Top features strongly correlated with batch
- Different results with different batch correction methods

## Technical Notes

### Computational Considerations

- Batch covariates add minimal computational overhead
- Batch-stratified CV may require larger datasets
- Domain adaptation methods may increase training time

### Software Dependencies

```python
# Core dependencies for batch-aware modeling
requirements = [
    "scikit-learn>=1.0",
    "pandas>=1.3",
    "numpy>=1.20",
    "statsmodels>=0.13",  # For batch effect testing
]
```

## References

For detailed methodology and empirical validation, please refer to the main manuscript (citation pending).

## See Also

- [Feature Selection Framework](feature_selection.md)
- [Enrichment Analysis](enrichment_analysis.md)
- [API Reference](../api/reference.md)
