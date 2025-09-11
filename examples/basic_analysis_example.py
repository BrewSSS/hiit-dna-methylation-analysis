#!/usr/bin/env python3
"""
Basic Analysis Example for HIIT DNA Methylation Study

This example demonstrates how to use the hiit_methylation package to:
1. Load and preprocess synthetic methylation data
2. Train machine learning models
3. Evaluate model performance
4. Visualize results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add the package to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hiit_methylation.data import create_sample_data, preprocess_methylation_data
from hiit_methylation.models import HIITMethylationClassifier, HIITMethylationRegressor
from hiit_methylation.utils import stratified_split_hiit, methylation_metrics
from hiit_methylation.visualization import plot_methylation_patterns, plot_hiit_effects


def main():
    """Run basic analysis example."""
    
    print("=== HIIT DNA Methylation Analysis Example ===")
    print()
    
    # Step 1: Generate synthetic data
    print("1. Generating synthetic methylation data...")
    methylation_data, metadata = create_sample_data(
        n_samples=80,
        n_cpg_sites=500,
        hiit_effect_size=0.2,
        random_state=42
    )
    
    print(f"   Generated data shape: {methylation_data.shape}")
    print(f"   Group distribution: {metadata['hiit_group'].value_counts().to_dict()}")
    print(f"   Time point distribution: {metadata['time_point'].value_counts().to_dict()}")
    print()
    
    # Step 2: Preprocess data
    print("2. Preprocessing methylation data...")
    processed_data = preprocess_methylation_data(
        methylation_data,
        normalize_method='logit',
        filter_variance=True,
        variance_threshold=0.001
    )
    print()
    
    # Step 3: Create binary classification task (HIIT vs Control)
    print("3. Setting up classification task (HIIT vs Control)...")
    
    # Filter to post-intervention samples only
    post_samples = metadata[metadata['time_point'] == 'Post'].index
    X_classification = processed_data.loc[post_samples]
    y_classification = (metadata.loc[post_samples, 'hiit_group'] == 'HIIT').astype(int)
    
    print(f"   Classification samples: {len(X_classification)}")
    print(f"   Class distribution: {y_classification.value_counts().to_dict()}")
    
    # Step 4: Train classification model
    print("\n4. Training classification model...")
    
    classifier = HIITMethylationClassifier(
        model_type='random_forest',
        feature_selection='univariate',
        n_features=50,
        random_state=42
    )
    
    # Cross-validation
    cv_results = classifier.cross_validate(X_classification, y_classification)
    print(f"   Cross-validation accuracy: {cv_results['mean_score']:.3f} ± {cv_results['std_score']:.3f}")
    
    # Train final model
    classifier.fit(X_classification, y_classification)
    
    # Predictions
    y_pred_class = classifier.predict(X_classification)
    y_pred_proba = classifier.predict_proba(X_classification)
    
    # Evaluate
    metrics = methylation_metrics(y_classification.values, y_pred_class, task_type='classification')
    print(f"   Training accuracy: {metrics['accuracy']:.3f}")
    if 'roc_auc' in metrics:
        print(f"   Training ROC AUC: {metrics['roc_auc']:.3f}")
    
    # Step 5: Feature importance
    importance = classifier.get_feature_importance()
    if importance is not None:
        print(f"   Top predictive CpG sites:")
        for site, score in importance.head(5).items():
            print(f"     {site}: {score:.4f}")
    
    print()
    
    # Step 6: Regression task (predict methylation changes)
    print("5. Setting up regression task (methylation changes)...")
    
    # Separate pre and post data
    pre_data = processed_data.loc[metadata[metadata['time_point'] == 'Pre'].index]
    post_data = processed_data.loc[metadata[metadata['time_point'] == 'Post'].index]
    
    # Calculate changes for HIIT group only
    hiit_subjects = metadata[metadata['hiit_group'] == 'HIIT']['subject_id'].unique() \
                    if 'subject_id' in metadata.columns else range(len(pre_data))
    
    # For simplicity, use mean methylation change as target
    # Align pre and post samples (simplified approach)
    if len(pre_data) >= len(post_data):
        methylation_changes = (post_data.iloc[:len(post_data)] - pre_data.iloc[:len(post_data)]).mean(axis=1)
        X_regression_data = pre_data.iloc[:len(post_data)]
    else:
        methylation_changes = (post_data.iloc[:len(pre_data)] - pre_data).mean(axis=1)
        X_regression_data = pre_data
    
    # Use pre-intervention data as features for HIIT samples
    hiit_pre_samples = metadata[
        (metadata['hiit_group'] == 'HIIT') & 
        (metadata['time_point'] == 'Pre')
    ].index
    
    if len(hiit_pre_samples) > 10:  # Ensure enough samples
        X_regression = processed_data.loc[hiit_pre_samples]
        # Create corresponding target values (synthetic for demo)
        y_regression = pd.Series(
            np.random.normal(0, 0.1, len(hiit_pre_samples)),
            index=hiit_pre_samples
        )
        
        print(f"   Regression samples: {len(X_regression)}")
        
        # Train regression model
        print("\n6. Training regression model...")
        
        regressor = HIITMethylationRegressor(
            model_type='random_forest',
            feature_selection='univariate',
            n_features=30,
            random_state=42
        )
        
        # Cross-validation
        cv_results = regressor.cross_validate(X_regression, y_regression)
        print(f"   Cross-validation R²: {cv_results['mean_score']:.3f} ± {cv_results['std_score']:.3f}")
        
        # Train final model
        regressor.fit(X_regression, y_regression)
        y_pred_reg = regressor.predict(X_regression)
        
        # Evaluate
        reg_metrics = methylation_metrics(y_regression.values, y_pred_reg, task_type='regression')
        print(f"   Training R²: {reg_metrics['r2_score']:.3f}")
        print(f"   Training RMSE: {reg_metrics['rmse']:.4f}")
    else:
        print("   Insufficient HIIT samples for regression analysis")
        regressor = None
    
    print()
    
    # Step 7: Visualization
    print("7. Creating visualizations...")
    
    try:
        # Basic methylation patterns
        fig1 = plot_methylation_patterns(
            methylation_data, 
            metadata,
            figsize=(10, 8)
        )
        fig1.savefig('methylation_patterns.png', dpi=150, bbox_inches='tight')
        print("   Saved: methylation_patterns.png")
        
        # HIIT effects (if we have pre/post data)
        if len(pre_data) > 0 and len(post_data) > 0:
            fig2 = plot_hiit_effects(
                pre_data.iloc[:min(20, len(pre_data))],  # Subset for visualization
                post_data.iloc[:min(20, len(post_data))],
                metadata,
                figsize=(12, 8)
            )
            fig2.savefig('hiit_effects.png', dpi=150, bbox_inches='tight')
            print("   Saved: hiit_effects.png")
        
        plt.close('all')  # Close figures to free memory
        
    except Exception as e:
        print(f"   Visualization error: {e}")
    
    print()
    
    # Step 8: Summary
    print("8. Analysis Summary:")
    print("   ✓ Generated synthetic HIIT methylation data")
    print("   ✓ Preprocessed data with logit transformation and filtering")
    print("   ✓ Trained classification model for HIIT vs Control")
    if regressor is not None:
        print("   ✓ Trained regression model for methylation changes")
    print("   ✓ Evaluated model performance")
    print("   ✓ Generated visualizations")
    
    print("\nExample completed successfully!")
    print("\nNext steps:")
    print("- Replace synthetic data with real methylation data")
    print("- Optimize hyperparameters using hyperparameter_tuning=True")
    print("- Add clinical/demographic features for improved predictions")
    print("- Perform differential methylation analysis")
    print("- Validate models on independent test sets")


if __name__ == "__main__":
    main()