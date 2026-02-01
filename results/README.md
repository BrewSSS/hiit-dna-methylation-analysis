# Results Directory

This directory contains all analysis and modeling outputs, organized by analysis stage.

## Directory Structure

```
results/
├── feature_selection/      # Feature selection results
│   ├── final_recommendations/  # Final recommended features
│   └── intermediate/       # Intermediate process results
├── models/                 # Model performance and evaluation
│   ├── binary/             # Binary classification results
│   └── multiclass/         # Multiclass classification results
├── enrichment/             # Pathway enrichment analysis
│   ├── gene_lists/         # Input gene lists
│   ├── gprofiler/          # g:Profiler results
│   ├── clusterprofiler/    # R clusterProfiler results
│   ├── annotations/        # Gene and GO annotations
│   └── reports/            # Enrichment analysis reports
├── tables/                 # Output tables
└── README.md               # This file
```

## Feature Selection Results (`feature_selection/`)

### Final Recommendations (`final_recommendations/`)
- `all_selected_features.csv` - Comprehensive list of all recommended features
- `best_lasso_features.csv` - Best features from Lasso method
- `best_elasticnet_features.csv` - Best features from ElasticNet method
- `best_enhanced_elasticnet_features.csv` - Best enhanced ElasticNet features
- `feature_comparison.csv` - Comparison between feature selection methods
- `multiclass_recommended_features.csv` - Recommended features for multiclass

### Intermediate Results (`intermediate/`)
- `lasso_*_results.csv` - Detailed Lasso method results
- `elasticnet_*_results.csv` - Detailed ElasticNet results
- `lasso_elasticnet_intersection_original.csv` - Common features between methods
- `*_stability.csv` - Feature stability assessments
- `*_feature_selection_results.pkl` - Serialized selection results

## Model Results (`models/`)

### Binary Classification (`binary/`)
- `model.pkl` - Saved model weights
- `predictions.csv` - Test set predictions
- `performance.pkl` - Performance metrics
- `classification_report.txt` - Classification report (accuracy, precision, etc.)

### Multiclass Classification (`multiclass/`)
- `model.pkl` - Saved model weights
- `predictions.csv` - Test set predictions
- `performance.pkl` - Performance metrics
- `classification_report.txt` - Classification report

### Root Directory Files
- `model_performance_comparison.csv` - Performance comparison across models
- `model_comparison_report.md` - Detailed model comparison report
- `final_model_report.md` - Final model report
- `binary_feature_importance.csv` - Binary classification feature importance scores
- `multiclass_rf_*.csv` - Multiclass random forest importance

## Enrichment Analysis (`enrichment/`)

### Gene Lists (`gene_lists/`)
- `all_genes.txt` - All differentially methylated genes
- `binary_genes.txt` - Binary classification-related genes
- `multiclass_genes.txt` - Multiclass classification-related genes
- `timeseries_genes.txt` - Time-series related genes
- `gene_lists_for_enrichment/` - Gene lists formatted for enrichment analysis

### g:Profiler Results (`gprofiler/`)
- `strict_results/` - Results with strict threshold (p<0.05)
- `relaxed_results/` - Results with relaxed threshold (p<0.10)
  - `full_results.json` - Complete g:Profiler results
  - `GO:BP/CC/MF_*.json` - GO category enrichment terms
  - `KEGG_*.json` - KEGG pathway enrichment results
  - `REAC_*.json` - Reactome pathway enrichment results

### clusterProfiler Results (`clusterprofiler/`)
- `all/` - Enrichment results for all genes
- `binary/` - Enrichment for binary classification genes
- `multiclass/` - Enrichment for multiclass classification genes
- `timeseries/` - Enrichment for time-series related genes
- `annotation_references/` - Gene annotation references (HGNC, etc.)

### Annotations (`annotations/`)
- `annotated_all_features.csv` - Annotations for all features
- `annotated_binary_features.csv` - Binary feature annotations
- `annotated_multiclass_features.csv` - Multiclass feature annotations
- `manual_go_annotations/` - Manually curated GO annotations

### Reports (`reports/`)
- `enrichment_analysis_report.md` - Enrichment analysis summary
- `GO_KEGG_Enrichment_Comprehensive_Report.md` - Detailed enrichment results
- `clusterProfiler_enrichment_report.md` - clusterProfiler analysis report

## Output Tables (`tables/`)

- `Table_S1_hierarchical_screening.html` - Supplementary Table 1: Hierarchical feature screening
- `Table_S2_selected_features.html/xlsx` - Supplementary Table 2: Selected features
- `Table_S3_full_enrichment.html` - Supplementary Table 3: Complete enrichment results
- `Table_S4_ML_feature_selection.html` - Supplementary Table 4: ML feature selection
- `Table2_Methylation_Features.html` - Main Table: Methylation features

## Data Flow

```
Raw Data (data/raw/)
    |
    v
Preprocessing (notebooks/02_preprocessing_feature_selection.ipynb)
    |
    v
Feature Selection --> results/feature_selection/
    |
    v
Model Training --> results/models/
    |
    v
Pathway Enrichment --> results/enrichment/
    |
    v
Figure Generation --> data/figures/
    |
    v
Final Publication (Manuscript)
```

## File Size Reference

- `feature_selection/`: ~35MB (including intermediate results)
- `models/`: ~20MB
- `enrichment/`: ~10MB
- `tables/`: ~5MB
- **Total**: ~70MB (can be compressed to reduce version control size)

## Usage Recommendations

1. **Quick Reference** - Check `*/final_recommendations/` and `/reports/` directories
2. **In-depth Analysis** - Review `intermediate/` and tool-specific output directories
3. **Visualization** - See corresponding figures in `data/figures/`
4. **Method Verification** - Review corresponding Jupyter notebooks in `notebooks/`

## Output File Formats

### CSV Files
- Delimiter: Comma (`,`)
- Encoding: UTF-8
- Header: First row contains column names
- Missing values: Empty string or `NA`

### JSON Files
- Standard JSON format
- UTF-8 encoding
- Nested structure for complex results

### Pickle Files
- Python pickle format for complex objects
- Contains model artifacts, fitted transformers
- Load with Python's `pickle` module

## Notes

- All CSV and JSON files are text format and can be viewed directly
- All .pkl files are Python serialized objects, load with Python
- Some larger intermediate results (e.g., pickle files) may be added to .gitignore
- Regularly backup important final result files

## See Also

- [Quick Start Guide](../docs/examples/quickstart.md)
- [API Reference](../docs/api/reference.md)
- [Data README](../data/README.md)
- [Models README](../models/README.md)
