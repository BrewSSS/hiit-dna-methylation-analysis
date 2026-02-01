# Models Directory

This directory contains trained model artifacts and related files for the HIIT-Methylation-Classification pipeline.

## Directory Structure

```
models/
├── binary/                 # Binary classification models
│   └── model.pkl           # Binary classifier weights
├── multiclass/             # Multiclass classification models
│   └── model.pkl           # Multiclass classifier weights
└── README.md               # This file
```

## Model Architecture Descriptions

### Binary Classification Models

Trained to distinguish between pre-intervention and post-intervention samples.

| Model | Description | Key Parameters |
|-------|-------------|----------------|
| Random Forest | Ensemble of decision trees | n_estimators, max_depth |
| Gradient Boosting | Sequential tree boosting | learning_rate, n_estimators |
| Logistic Regression | Linear classifier | C (regularization), penalty |
| SVM | Support Vector Machine | C, kernel, gamma |

### Multiclass Classification Models

Trained to classify intervention duration (4 weeks, 8 weeks, 12 weeks).

| Model | Description | Approach |
|-------|-------------|----------|
| Random Forest | Ensemble classifier | One-vs-Rest |
| Gradient Boosting | Boosted trees | Multinomial |
| Neural Network | MLP classifier | Softmax output |

## Usage

```python
import pickle

# Load binary classification model
with open('binary/model.pkl', 'rb') as f:
    binary_model = pickle.load(f)

# Load multiclass classification model
with open('multiclass/model.pkl', 'rb') as f:
    multiclass_model = pickle.load(f)

# Make predictions
predictions = binary_model.predict(X_test)
```

## Training Models

### Training a Binary Classifier

```python
from src.models import HIITClassifier

# Initialize model
classifier = HIITClassifier(
    model_type="random_forest",
    n_estimators=100,
    max_depth=10,
    class_weight="balanced",
    random_state=42
)

# Train
classifier.fit(X_train, y_train)

# Save model
classifier.save("models/binary/model.pkl")
```

### Training with Batch-Aware Modeling

```python
from src.models import HIITClassifier

# Initialize with batch awareness
classifier = HIITClassifier(
    model_type="gradient_boosting",
    batch_aware=True,
    batch_strategy="covariate"
)

# Train with batch information
classifier.fit(X_train, y_train, batch=batch_labels)

# Save model (includes batch encoder)
classifier.save("models/binary/batch_aware_model.pkl")
```

### Training a Multiclass Model

```python
from src.models import HIITClassifier

# Initialize for multiclass
classifier = HIITClassifier(
    model_type="random_forest",
    n_estimators=200,
    multiclass_strategy="ovr",  # One-vs-Rest
    random_state=42
)

# Train on duration labels (0=4wk, 1=8wk, 2=12wk)
classifier.fit(X_train, duration_labels)

# Save
classifier.save("models/multiclass/model.pkl")
```

## Important Notes

### No Pre-Trained Models Included

This repository does not include pre-trained model weights for open-source release. Users must:
1. Download the data from GEO
2. Run the preprocessing pipeline
3. Train models using provided notebooks or scripts

This ensures:
- Reproducibility from raw data
- Compliance with data usage agreements
- Flexibility for custom configurations

### Model Reproducibility

To ensure reproducible training:

```python
import numpy as np
import random

# Set all random seeds
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Initialize with seed
classifier = HIITClassifier(
    model_type="random_forest",
    random_state=SEED
)
```

## Related Information

- Model performance evaluation: See `../results/models/` directory
- Features used by models: See `../results/feature_selection/` directory
- Model construction process: See `../notebooks/03_modeling_evaluation.ipynb`

## Hyperparameter Reference

### Random Forest

```python
default_params = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1
}
```

### Gradient Boosting

```python
default_params = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 3,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "subsample": 0.8,
    "random_state": 42
}
```

## See Also

- [Quick Start Guide](../docs/examples/quickstart.md)
- [Feature Selection Framework](../docs/methodology/feature_selection.md)
- [Batch-Aware Modeling](../docs/methodology/batch_correction.md)
- [API Reference](../docs/api/reference.md)
