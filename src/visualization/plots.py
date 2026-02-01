"""
Standard visualization functions for HIIT methylation analysis.

This module provides reusable plotting functions for dimensionality reduction,
feature importance, differential methylation analysis, and statistical comparisons.
All functions support publication-quality output with configurable styling.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix
from scipy import stats
from typing import Optional, Union, List, Tuple, Dict, Any

# Default color palette (colorblind-friendly, Nature-style)
DEFAULT_COLORS = {
    'primary': '#E64B35',      # Red
    'secondary': '#4DBBD5',    # Blue
    'tertiary': '#00A087',     # Green
    'quaternary': '#3C5488',   # Deep blue
    'neutral': '#808080',      # Gray
    'highlight': '#F39B7F',    # Light orange
    'accent': '#91D1C2',       # Teal
}

# Default plot parameters
DEFAULT_FIGSIZE = (10, 8)
DEFAULT_DPI = 300
DEFAULT_FONTSIZE = {
    'title': 12,
    'label': 10,
    'tick': 9,
    'legend': 9,
}


def _apply_base_style(ax: plt.Axes,
                      remove_top_right: bool = True,
                      grid: bool = False,
                      grid_alpha: float = 0.3) -> None:
    """
    Apply base styling to matplotlib axes.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes object.
    remove_top_right : bool, optional
        Whether to remove top and right spines. Default is True.
    grid : bool, optional
        Whether to show grid. Default is False.
    grid_alpha : float, optional
        Grid transparency. Default is 0.3.
    """
    if remove_top_right:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    if grid:
        ax.grid(True, alpha=grid_alpha, linestyle='--', linewidth=0.5)


def plot_pca_visualization(
    data: pd.DataFrame,
    labels: pd.Series,
    batch: Optional[pd.Series] = None,
    n_components: int = 2,
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    colors: Optional[Dict[str, str]] = None,
    markers: Optional[Dict[str, str]] = None,
    alpha: float = 0.7,
    point_size: int = 50,
    show_legend: bool = True,
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes, PCA]:
    """
    Create PCA visualization with group coloring and optional batch shapes.

    Parameters
    ----------
    data : pd.DataFrame
        Methylation data matrix (samples x features) or (features x samples).
        If features x samples, will be transposed automatically.
    labels : pd.Series
        Group labels for each sample, indexed by sample ID.
    batch : pd.Series, optional
        Batch information for different marker shapes.
    n_components : int, optional
        Number of PCA components. Default is 2.
    figsize : tuple, optional
        Figure size in inches. Default is (10, 8).
    title : str, optional
        Plot title. If None, uses default title.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    colors : dict, optional
        Dictionary mapping label values to colors.
    markers : dict, optional
        Dictionary mapping batch values to marker shapes.
    alpha : float, optional
        Point transparency. Default is 0.7.
    point_size : int, optional
        Size of scatter points. Default is 50.
    show_legend : bool, optional
        Whether to show legend. Default is True.
    ax : plt.Axes, optional
        Existing axes to plot on. If None, creates new figure.

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure object.
    ax : plt.Axes
        Matplotlib axes object.
    pca : PCA
        Fitted PCA object for further analysis.

    Examples
    --------
    >>> fig, ax, pca = plot_pca_visualization(
    ...     data=methyl_data,
    ...     labels=sample_info['group'],
    ...     batch=sample_info['batch'],
    ...     title='PCA of Methylation Data'
    ... )
    """
    # Ensure data is samples x features
    if data.shape[0] < data.shape[1]:
        data = data.T

    # Align samples
    common_samples = data.index.intersection(labels.index)
    X = data.loc[common_samples].values
    y = labels.loc[common_samples]

    # Handle missing values
    X = np.nan_to_num(X, nan=0)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Default colors and markers
    unique_labels = y.unique()
    if colors is None:
        color_list = list(DEFAULT_COLORS.values())
        colors = {label: color_list[i % len(color_list)]
                  for i, label in enumerate(unique_labels)}

    default_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h']
    if batch is not None and markers is None:
        unique_batches = batch.loc[common_samples].unique()
        markers = {b: default_markers[i % len(default_markers)]
                   for i, b in enumerate(unique_batches)}

    # Plot points
    if batch is not None:
        batch_aligned = batch.loc[common_samples]
        for label in unique_labels:
            for batch_val in batch_aligned.unique():
                mask = (y == label) & (batch_aligned == batch_val)
                if mask.sum() > 0:
                    ax.scatter(
                        X_pca[mask, 0], X_pca[mask, 1],
                        c=colors.get(label, DEFAULT_COLORS['neutral']),
                        marker=markers.get(batch_val, 'o'),
                        s=point_size, alpha=alpha,
                        edgecolors='white', linewidth=0.5,
                        label=f'{label} ({batch_val})'
                    )
    else:
        for label in unique_labels:
            mask = y == label
            n_samples = mask.sum()
            ax.scatter(
                X_pca[mask, 0], X_pca[mask, 1],
                c=colors.get(label, DEFAULT_COLORS['neutral']),
                s=point_size, alpha=alpha,
                edgecolors='white', linewidth=0.5,
                label=f'{label} (n={n_samples})'
            )

    # Labels and title
    var_explained = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f'PC1 ({var_explained[0]:.1f}%)', fontsize=DEFAULT_FONTSIZE['label'])
    ax.set_ylabel(f'PC2 ({var_explained[1]:.1f}%)', fontsize=DEFAULT_FONTSIZE['label'])

    if title:
        ax.set_title(title, fontsize=DEFAULT_FONTSIZE['title'], fontweight='bold')

    # Styling
    _apply_base_style(ax, grid=True)

    if show_legend:
        ax.legend(loc='best', fontsize=DEFAULT_FONTSIZE['legend'],
                  frameon=True, fancybox=False, edgecolor='gray')

    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')

    return fig, ax, pca


def plot_tsne_visualization(
    data: pd.DataFrame,
    labels: pd.Series,
    perplexity: int = 30,
    n_iter: int = 1000,
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    colors: Optional[Dict[str, str]] = None,
    alpha: float = 0.7,
    point_size: int = 50,
    random_state: int = 42,
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create t-SNE visualization for methylation data.

    Parameters
    ----------
    data : pd.DataFrame
        Methylation data matrix (samples x features) or (features x samples).
    labels : pd.Series
        Group labels for each sample.
    perplexity : int, optional
        t-SNE perplexity parameter. Default is 30.
    n_iter : int, optional
        Number of iterations. Default is 1000.
    figsize : tuple, optional
        Figure size. Default is (10, 8).
    title : str, optional
        Plot title.
    save_path : str, optional
        Path to save figure.
    colors : dict, optional
        Color mapping for labels.
    alpha : float, optional
        Point transparency. Default is 0.7.
    point_size : int, optional
        Size of scatter points. Default is 50.
    random_state : int, optional
        Random seed for reproducibility. Default is 42.
    ax : plt.Axes, optional
        Existing axes to plot on.

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure object.
    ax : plt.Axes
        Matplotlib axes object.

    Examples
    --------
    >>> fig, ax = plot_tsne_visualization(
    ...     data=methyl_data,
    ...     labels=sample_info['group'],
    ...     perplexity=30
    ... )
    """
    # Ensure data is samples x features
    if data.shape[0] < data.shape[1]:
        data = data.T

    # Align samples
    common_samples = data.index.intersection(labels.index)
    X = data.loc[common_samples].values
    y = labels.loc[common_samples]

    # Handle missing values
    X = np.nan_to_num(X, nan=0)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter,
                random_state=random_state, init='pca')
    X_tsne = tsne.fit_transform(X_scaled)

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Default colors
    unique_labels = y.unique()
    if colors is None:
        color_list = list(DEFAULT_COLORS.values())
        colors = {label: color_list[i % len(color_list)]
                  for i, label in enumerate(unique_labels)}

    # Plot points
    for label in unique_labels:
        mask = y == label
        n_samples = mask.sum()
        ax.scatter(
            X_tsne[mask, 0], X_tsne[mask, 1],
            c=colors.get(label, DEFAULT_COLORS['neutral']),
            s=point_size, alpha=alpha,
            edgecolors='white', linewidth=0.5,
            label=f'{label} (n={n_samples})'
        )

    # Labels and title
    ax.set_xlabel('t-SNE 1', fontsize=DEFAULT_FONTSIZE['label'])
    ax.set_ylabel('t-SNE 2', fontsize=DEFAULT_FONTSIZE['label'])

    if title:
        ax.set_title(title, fontsize=DEFAULT_FONTSIZE['title'], fontweight='bold')
    else:
        ax.set_title(f't-SNE (perplexity={perplexity})',
                     fontsize=DEFAULT_FONTSIZE['title'], fontweight='bold')

    # Styling
    _apply_base_style(ax, grid=True)
    ax.legend(loc='best', fontsize=DEFAULT_FONTSIZE['legend'],
              frameon=True, fancybox=False, edgecolor='gray')

    if save_path:
        fig.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')

    return fig, ax


def plot_umap_visualization(
    data: pd.DataFrame,
    labels: pd.Series,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    colors: Optional[Dict[str, str]] = None,
    alpha: float = 0.7,
    point_size: int = 50,
    random_state: int = 42,
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create UMAP visualization for methylation data.

    Parameters
    ----------
    data : pd.DataFrame
        Methylation data matrix (samples x features) or (features x samples).
    labels : pd.Series
        Group labels for each sample.
    n_neighbors : int, optional
        UMAP n_neighbors parameter. Default is 15.
    min_dist : float, optional
        UMAP min_dist parameter. Default is 0.1.
    figsize : tuple, optional
        Figure size. Default is (10, 8).
    title : str, optional
        Plot title.
    save_path : str, optional
        Path to save figure.
    colors : dict, optional
        Color mapping for labels.
    alpha : float, optional
        Point transparency. Default is 0.7.
    point_size : int, optional
        Size of scatter points. Default is 50.
    random_state : int, optional
        Random seed. Default is 42.
    ax : plt.Axes, optional
        Existing axes to plot on.

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure object.
    ax : plt.Axes
        Matplotlib axes object.

    Raises
    ------
    ImportError
        If umap-learn is not installed.

    Examples
    --------
    >>> fig, ax = plot_umap_visualization(
    ...     data=methyl_data,
    ...     labels=sample_info['group'],
    ...     n_neighbors=15
    ... )
    """
    try:
        import umap
    except ImportError:
        raise ImportError(
            "UMAP requires umap-learn package. "
            "Install with: pip install umap-learn"
        )

    # Ensure data is samples x features
    if data.shape[0] < data.shape[1]:
        data = data.T

    # Align samples
    common_samples = data.index.intersection(labels.index)
    X = data.loc[common_samples].values
    y = labels.loc[common_samples]

    # Handle missing values
    X = np.nan_to_num(X, nan=0)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform UMAP
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        random_state=random_state)
    X_umap = reducer.fit_transform(X_scaled)

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Default colors
    unique_labels = y.unique()
    if colors is None:
        color_list = list(DEFAULT_COLORS.values())
        colors = {label: color_list[i % len(color_list)]
                  for i, label in enumerate(unique_labels)}

    # Plot points
    for label in unique_labels:
        mask = y == label
        n_samples = mask.sum()
        ax.scatter(
            X_umap[mask, 0], X_umap[mask, 1],
            c=colors.get(label, DEFAULT_COLORS['neutral']),
            s=point_size, alpha=alpha,
            edgecolors='white', linewidth=0.5,
            label=f'{label} (n={n_samples})'
        )

    # Labels and title
    ax.set_xlabel('UMAP 1', fontsize=DEFAULT_FONTSIZE['label'])
    ax.set_ylabel('UMAP 2', fontsize=DEFAULT_FONTSIZE['label'])

    if title:
        ax.set_title(title, fontsize=DEFAULT_FONTSIZE['title'], fontweight='bold')
    else:
        ax.set_title(f'UMAP (n_neighbors={n_neighbors})',
                     fontsize=DEFAULT_FONTSIZE['title'], fontweight='bold')

    # Styling
    _apply_base_style(ax, grid=True)
    ax.legend(loc='best', fontsize=DEFAULT_FONTSIZE['legend'],
              frameon=True, fancybox=False, edgecolor='gray')

    if save_path:
        fig.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')

    return fig, ax


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    show_direction: bool = True,
    figsize: Tuple[int, int] = (10, 8),
    horizontal: bool = True,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    color_positive: str = DEFAULT_COLORS['primary'],
    color_negative: str = DEFAULT_COLORS['secondary'],
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot feature importance with positive/negative direction indicators.

    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame with columns 'feature', 'importance', and optionally 'direction'.
        If 'direction' column is not present but importance values are signed,
        direction is inferred from the sign.
    top_n : int, optional
        Number of top features to display. Default is 20.
    show_direction : bool, optional
        Whether to color bars by direction. Default is True.
    figsize : tuple, optional
        Figure size. Default is (10, 8).
    horizontal : bool, optional
        Whether to use horizontal bars. Default is True.
    title : str, optional
        Plot title.
    save_path : str, optional
        Path to save figure.
    color_positive : str, optional
        Color for positive/hypermethylated features.
    color_negative : str, optional
        Color for negative/hypomethylated features.
    ax : plt.Axes, optional
        Existing axes to plot on.

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure object.
    ax : plt.Axes
        Matplotlib axes object.

    Examples
    --------
    >>> importance_df = pd.DataFrame({
    ...     'feature': ['cg001', 'cg002', 'cg003'],
    ...     'importance': [0.8, -0.5, 0.3],
    ...     'gene': ['GENE1', 'GENE2', 'GENE3']
    ... })
    >>> fig, ax = plot_feature_importance(importance_df, top_n=10)
    """
    df = importance_df.copy()

    # Ensure importance column exists
    if 'importance' not in df.columns:
        # Try to find a suitable column
        importance_cols = [c for c in df.columns if 'importance' in c.lower()
                          or 'coefficient' in c.lower() or 'weight' in c.lower()]
        if importance_cols:
            df['importance'] = df[importance_cols[0]]
        else:
            raise ValueError("DataFrame must have an 'importance' column")

    # Determine direction if not present
    if 'direction' not in df.columns and show_direction:
        df['direction'] = np.where(df['importance'] >= 0, 'positive', 'negative')

    # Sort by absolute importance and get top_n
    df['abs_importance'] = df['importance'].abs()
    df = df.nlargest(top_n, 'abs_importance')

    # Create labels
    if 'gene' in df.columns:
        df['label'] = df.apply(
            lambda x: f"{x['feature']} ({x['gene']})" if pd.notna(x['gene']) else x['feature'],
            axis=1
        )
    else:
        df['label'] = df['feature']

    # Sort for plotting (smallest at bottom for horizontal bars)
    df = df.sort_values('abs_importance', ascending=True)

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Determine colors
    if show_direction and 'direction' in df.columns:
        colors = [color_positive if d == 'positive' else color_negative
                  for d in df['direction']]
    else:
        colors = [DEFAULT_COLORS['quaternary']] * len(df)

    # Create bar plot
    if horizontal:
        bars = ax.barh(range(len(df)), df['abs_importance'], color=colors,
                       edgecolor='white', linewidth=0.5)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df['label'], fontsize=DEFAULT_FONTSIZE['tick'])
        ax.set_xlabel('Importance', fontsize=DEFAULT_FONTSIZE['label'])
    else:
        bars = ax.bar(range(len(df)), df['abs_importance'], color=colors,
                      edgecolor='white', linewidth=0.5)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['label'], rotation=45, ha='right',
                           fontsize=DEFAULT_FONTSIZE['tick'])
        ax.set_ylabel('Importance', fontsize=DEFAULT_FONTSIZE['label'])

    # Title
    if title:
        ax.set_title(title, fontsize=DEFAULT_FONTSIZE['title'], fontweight='bold')
    else:
        ax.set_title(f'Top {top_n} Feature Importance',
                     fontsize=DEFAULT_FONTSIZE['title'], fontweight='bold')

    # Legend for direction
    if show_direction and 'direction' in df.columns:
        legend_elements = [
            mpatches.Patch(color=color_positive, label='Hypermethylation'),
            mpatches.Patch(color=color_negative, label='Hypomethylation')
        ]
        ax.legend(handles=legend_elements, loc='lower right',
                  fontsize=DEFAULT_FONTSIZE['legend'])

    # Styling
    _apply_base_style(ax, grid=True)

    if save_path:
        fig.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')

    return fig, ax


def plot_volcano(
    pvalues: Union[np.ndarray, pd.Series],
    effect_sizes: Union[np.ndarray, pd.Series],
    feature_names: Optional[Union[np.ndarray, pd.Series, List[str]]] = None,
    pvalue_threshold: float = 0.05,
    effect_threshold: float = 0.3,
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    highlight_features: Optional[List[str]] = None,
    color_up: str = DEFAULT_COLORS['primary'],
    color_down: str = DEFAULT_COLORS['secondary'],
    color_ns: str = DEFAULT_COLORS['neutral'],
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """
    Create volcano plot for differential methylation analysis.

    Parameters
    ----------
    pvalues : array-like
        P-values for each feature.
    effect_sizes : array-like
        Effect sizes (e.g., Cohen's d or log fold change) for each feature.
    feature_names : array-like, optional
        Names of features for labeling.
    pvalue_threshold : float, optional
        P-value significance threshold. Default is 0.05.
    effect_threshold : float, optional
        Effect size threshold for significance. Default is 0.3.
    figsize : tuple, optional
        Figure size. Default is (10, 8).
    title : str, optional
        Plot title.
    save_path : str, optional
        Path to save figure.
    highlight_features : list, optional
        List of feature names to highlight with labels.
    color_up : str, optional
        Color for upregulated/hypermethylated features.
    color_down : str, optional
        Color for downregulated/hypomethylated features.
    color_ns : str, optional
        Color for non-significant features.
    ax : plt.Axes, optional
        Existing axes to plot on.

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure object.
    ax : plt.Axes
        Matplotlib axes object.
    results_df : pd.DataFrame
        DataFrame with classification results.

    Examples
    --------
    >>> fig, ax, results = plot_volcano(
    ...     pvalues=diff_results['pvalue'],
    ...     effect_sizes=diff_results['effect_size'],
    ...     feature_names=diff_results['cpg'],
    ...     pvalue_threshold=0.05,
    ...     effect_threshold=0.4
    ... )
    """
    # Convert to numpy arrays
    pvalues = np.array(pvalues)
    effect_sizes = np.array(effect_sizes)

    if feature_names is not None:
        feature_names = np.array(feature_names)
    else:
        feature_names = np.array([f'Feature_{i}' for i in range(len(pvalues))])

    # Calculate -log10(p-value)
    log_pvalues = -np.log10(np.clip(pvalues, 1e-300, 1))

    # Classify points
    significant = (pvalues < pvalue_threshold) & (np.abs(effect_sizes) > effect_threshold)
    hyper = significant & (effect_sizes > 0)
    hypo = significant & (effect_sizes < 0)
    nonsig = ~significant

    # Create results DataFrame
    results_df = pd.DataFrame({
        'feature': feature_names,
        'pvalue': pvalues,
        'effect_size': effect_sizes,
        'log_pvalue': log_pvalues,
        'significant': significant,
        'direction': np.where(hyper, 'hyper', np.where(hypo, 'hypo', 'nonsig'))
    })

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Plot non-significant points (background)
    ax.scatter(effect_sizes[nonsig], log_pvalues[nonsig],
               c=color_ns, s=1, alpha=0.3, rasterized=True, label='NS')

    # Plot significant points
    n_hyper = hyper.sum()
    n_hypo = hypo.sum()

    ax.scatter(effect_sizes[hyper], log_pvalues[hyper],
               c=color_up, s=3, alpha=0.6, rasterized=True,
               label=f'Hyper ({n_hyper:,})')
    ax.scatter(effect_sizes[hypo], log_pvalues[hypo],
               c=color_down, s=3, alpha=0.6, rasterized=True,
               label=f'Hypo ({n_hypo:,})')

    # Add threshold lines
    ax.axhline(-np.log10(pvalue_threshold), color='black', linestyle='--',
               linewidth=0.8, alpha=0.5, zorder=0)
    ax.axvline(effect_threshold, color='black', linestyle='--',
               linewidth=0.8, alpha=0.5, zorder=0)
    ax.axvline(-effect_threshold, color='black', linestyle='--',
               linewidth=0.8, alpha=0.5, zorder=0)

    # Highlight specific features
    if highlight_features:
        for feat in highlight_features:
            idx = np.where(feature_names == feat)[0]
            if len(idx) > 0:
                i = idx[0]
                ax.annotate(feat, (effect_sizes[i], log_pvalues[i]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=7, arrowprops=dict(arrowstyle='->', lw=0.5))

    # Labels and title
    ax.set_xlabel("Effect Size (Cohen's d)", fontsize=DEFAULT_FONTSIZE['label'])
    ax.set_ylabel('-log10(P-value)', fontsize=DEFAULT_FONTSIZE['label'])

    if title:
        ax.set_title(title, fontsize=DEFAULT_FONTSIZE['title'], fontweight='bold')
    else:
        ax.set_title('Volcano Plot', fontsize=DEFAULT_FONTSIZE['title'], fontweight='bold')

    # Legend
    legend = ax.legend(loc='upper right', fontsize=DEFAULT_FONTSIZE['legend'],
                       frameon=True, markerscale=3)
    legend.set_title(f'p < {pvalue_threshold}, |d| > {effect_threshold}',
                     prop={'size': DEFAULT_FONTSIZE['legend'] - 1})

    # Styling
    _apply_base_style(ax)

    # Add summary annotation
    total = n_hyper + n_hypo
    ax.text(0.98, 0.02, f'n = {total:,} DMPs',
            transform=ax.transAxes, fontsize=DEFAULT_FONTSIZE['tick'],
            va='bottom', ha='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    if save_path:
        fig.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')

    return fig, ax, results_df


def plot_heatmap(
    data: pd.DataFrame,
    row_labels: Optional[pd.Series] = None,
    col_labels: Optional[pd.Series] = None,
    cluster_rows: bool = True,
    cluster_cols: bool = True,
    cmap: str = 'RdBu_r',
    figsize: Tuple[int, int] = (12, 10),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    center: Optional[float] = None,
    row_colors: Optional[pd.DataFrame] = None,
    col_colors: Optional[pd.DataFrame] = None,
    **kwargs
) -> sns.matrix.ClusterGrid:
    """
    Create clustered heatmap for methylation data.

    Parameters
    ----------
    data : pd.DataFrame
        Data matrix (features x samples) for visualization.
    row_labels : pd.Series, optional
        Labels for row annotations.
    col_labels : pd.Series, optional
        Labels for column annotations.
    cluster_rows : bool, optional
        Whether to cluster rows. Default is True.
    cluster_cols : bool, optional
        Whether to cluster columns. Default is True.
    cmap : str, optional
        Colormap name. Default is 'RdBu_r'.
    figsize : tuple, optional
        Figure size. Default is (12, 10).
    title : str, optional
        Plot title.
    save_path : str, optional
        Path to save figure.
    vmin : float, optional
        Minimum value for color scale.
    vmax : float, optional
        Maximum value for color scale.
    center : float, optional
        Center value for diverging colormap.
    row_colors : pd.DataFrame, optional
        Color annotations for rows.
    col_colors : pd.DataFrame, optional
        Color annotations for columns.
    **kwargs : dict
        Additional arguments passed to seaborn.clustermap.

    Returns
    -------
    g : sns.matrix.ClusterGrid
        Seaborn ClusterGrid object.

    Examples
    --------
    >>> g = plot_heatmap(
    ...     data=top_features_data,
    ...     col_labels=sample_info['group'],
    ...     cluster_cols=True,
    ...     cmap='RdBu_r',
    ...     center=0.5
    ... )
    """
    # Create color annotations
    if col_labels is not None and col_colors is None:
        unique_labels = col_labels.unique()
        color_list = list(DEFAULT_COLORS.values())
        color_map = {label: color_list[i % len(color_list)]
                     for i, label in enumerate(unique_labels)}
        col_colors = col_labels.map(color_map)

    if row_labels is not None and row_colors is None:
        unique_labels = row_labels.unique()
        color_list = list(DEFAULT_COLORS.values())
        color_map = {label: color_list[i % len(color_list)]
                     for i, label in enumerate(unique_labels)}
        row_colors = row_labels.map(color_map)

    # Create clustermap
    g = sns.clustermap(
        data,
        cmap=cmap,
        figsize=figsize,
        row_cluster=cluster_rows,
        col_cluster=cluster_cols,
        row_colors=row_colors,
        col_colors=col_colors,
        vmin=vmin,
        vmax=vmax,
        center=center,
        linewidths=0,
        xticklabels=False,
        yticklabels=True if data.shape[0] <= 50 else False,
        **kwargs
    )

    # Title
    if title:
        g.fig.suptitle(title, fontsize=DEFAULT_FONTSIZE['title'],
                       fontweight='bold', y=1.02)

    # Adjust colorbar
    g.cax.set_ylabel('Methylation (beta)', fontsize=DEFAULT_FONTSIZE['label'])

    if save_path:
        g.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')

    return g


def plot_boxplot_comparison(
    data: pd.DataFrame,
    groups: pd.Series,
    feature_name: str,
    test: str = 'ttest',
    show_pvalue: bool = True,
    figsize: Tuple[int, int] = (8, 6),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    colors: Optional[Dict[str, str]] = None,
    show_points: bool = True,
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes, float]:
    """
    Create boxplot comparing methylation levels across groups with statistical test.

    Parameters
    ----------
    data : pd.DataFrame
        Methylation data (features x samples or samples x features).
    groups : pd.Series
        Group labels indexed by sample ID.
    feature_name : str
        Name of feature (CpG) to compare.
    test : str, optional
        Statistical test: 'ttest', 'mannwhitneyu', or 'kruskal'. Default is 'ttest'.
    show_pvalue : bool, optional
        Whether to show p-value annotation. Default is True.
    figsize : tuple, optional
        Figure size. Default is (8, 6).
    title : str, optional
        Plot title.
    save_path : str, optional
        Path to save figure.
    colors : dict, optional
        Color mapping for groups.
    show_points : bool, optional
        Whether to show individual data points. Default is True.
    ax : plt.Axes, optional
        Existing axes to plot on.

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure object.
    ax : plt.Axes
        Matplotlib axes object.
    pvalue : float
        P-value from statistical test.

    Examples
    --------
    >>> fig, ax, pval = plot_boxplot_comparison(
    ...     data=methyl_data,
    ...     groups=sample_info['group'],
    ...     feature_name='cg00001234',
    ...     test='mannwhitneyu'
    ... )
    """
    # Extract feature data
    if feature_name in data.index:
        values = data.loc[feature_name]
    elif feature_name in data.columns:
        values = data[feature_name]
    else:
        raise ValueError(f"Feature '{feature_name}' not found in data")

    # Align with groups
    common_samples = values.index.intersection(groups.index)
    values = values.loc[common_samples]
    groups_aligned = groups.loc[common_samples]

    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'value': values.values,
        'group': groups_aligned.values
    })

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Default colors
    unique_groups = groups_aligned.unique()
    if colors is None:
        color_list = list(DEFAULT_COLORS.values())
        colors = {g: color_list[i % len(color_list)]
                  for i, g in enumerate(unique_groups)}

    palette = [colors.get(g, DEFAULT_COLORS['neutral']) for g in unique_groups]

    # Create boxplot
    bp = ax.boxplot(
        [plot_df[plot_df['group'] == g]['value'].dropna() for g in unique_groups],
        patch_artist=True,
        widths=0.6
    )

    # Color boxes
    for patch, color in zip(bp['boxes'], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Style other elements
    for element in ['whiskers', 'caps']:
        for item in bp[element]:
            item.set_color('black')
            item.set_linewidth(1)
    for median in bp['medians']:
        median.set_color('red')
        median.set_linewidth(2)

    # Add individual points with jitter
    if show_points:
        for i, g in enumerate(unique_groups):
            group_data = plot_df[plot_df['group'] == g]['value'].dropna()
            jitter = np.random.normal(0, 0.05, size=len(group_data))
            ax.scatter(np.ones(len(group_data)) * (i + 1) + jitter,
                      group_data, s=15, alpha=0.4, c='black', zorder=3)

    # Perform statistical test
    group_values = [plot_df[plot_df['group'] == g]['value'].dropna().values
                    for g in unique_groups]

    if len(unique_groups) == 2:
        if test == 'ttest':
            stat, pvalue = stats.ttest_ind(group_values[0], group_values[1])
        elif test == 'mannwhitneyu':
            stat, pvalue = stats.mannwhitneyu(group_values[0], group_values[1])
        else:
            stat, pvalue = stats.kruskal(*group_values)
    else:
        stat, pvalue = stats.kruskal(*group_values)

    # Add p-value annotation
    if show_pvalue:
        if pvalue < 0.001:
            pval_text = 'p < 0.001'
        else:
            pval_text = f'p = {pvalue:.3f}'

        ax.text(0.98, 0.98, pval_text, transform=ax.transAxes,
                ha='right', va='top', fontsize=DEFAULT_FONTSIZE['tick'],
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # Labels
    ax.set_xticklabels(unique_groups, fontsize=DEFAULT_FONTSIZE['tick'])
    ax.set_ylabel('Methylation (beta)', fontsize=DEFAULT_FONTSIZE['label'])

    if title:
        ax.set_title(title, fontsize=DEFAULT_FONTSIZE['title'], fontweight='bold')
    else:
        ax.set_title(feature_name, fontsize=DEFAULT_FONTSIZE['title'], fontweight='bold')

    # Styling
    _apply_base_style(ax)

    if save_path:
        fig.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')

    return fig, ax, pvalue


def plot_trajectory(
    data: pd.DataFrame,
    timepoints: pd.Series,
    feature_names: Union[str, List[str]],
    sample_ids: Optional[pd.Series] = None,
    show_individual: bool = True,
    show_mean: bool = True,
    show_ci: bool = True,
    figsize: Tuple[int, int] = (10, 6),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    colors: Optional[Dict[str, str]] = None,
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot methylation trajectories over time.

    Parameters
    ----------
    data : pd.DataFrame
        Methylation data (features x samples).
    timepoints : pd.Series
        Time point labels indexed by sample ID.
    feature_names : str or list
        Feature name(s) to plot.
    sample_ids : pd.Series, optional
        Subject IDs for connecting individual trajectories.
    show_individual : bool, optional
        Whether to show individual data points. Default is True.
    show_mean : bool, optional
        Whether to show mean trajectory. Default is True.
    show_ci : bool, optional
        Whether to show 95% confidence interval. Default is True.
    figsize : tuple, optional
        Figure size. Default is (10, 6).
    title : str, optional
        Plot title.
    save_path : str, optional
        Path to save figure.
    colors : dict, optional
        Color mapping for timepoints.
    ax : plt.Axes, optional
        Existing axes to plot on.

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure object.
    ax : plt.Axes
        Matplotlib axes object.

    Examples
    --------
    >>> fig, ax = plot_trajectory(
    ...     data=methyl_data,
    ...     timepoints=sample_info['time_point'],
    ...     feature_names='cg00001234',
    ...     show_mean=True,
    ...     show_ci=True
    ... )
    """
    if isinstance(feature_names, str):
        feature_names = [feature_names]

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Get unique timepoints in order
    time_order = ['Baseline', '4W', '8W', '12W']  # Default order
    unique_times = timepoints.unique()
    ordered_times = [t for t in time_order if t in unique_times]
    if len(ordered_times) != len(unique_times):
        ordered_times = list(unique_times)

    time_to_num = {t: i for i, t in enumerate(ordered_times)}

    # Default colors
    if colors is None:
        colors = {
            'Baseline': DEFAULT_COLORS['neutral'],
            '4W': DEFAULT_COLORS['accent'],
            '8W': DEFAULT_COLORS['quaternary'],
            '12W': DEFAULT_COLORS['primary']
        }

    for feat_idx, feature in enumerate(feature_names):
        if feature not in data.index:
            continue

        values = data.loc[feature]
        common_samples = values.index.intersection(timepoints.index)
        values = values.loc[common_samples]
        times = timepoints.loc[common_samples]

        # Prepare data by timepoint
        data_by_time = {}
        for t in ordered_times:
            mask = times == t
            if mask.sum() > 0:
                data_by_time[t] = values[mask].dropna()

        positions = list(range(len(ordered_times)))

        # Box plots
        plot_data = [data_by_time.get(t, pd.Series([])).values for t in ordered_times]
        valid_positions = [i for i, d in enumerate(plot_data) if len(d) > 0]
        valid_data = [d for d in plot_data if len(d) > 0]
        valid_colors = [colors.get(ordered_times[i], DEFAULT_COLORS['neutral'])
                        for i in valid_positions]

        bp = ax.boxplot(valid_data, positions=valid_positions, widths=0.5,
                        patch_artist=True, showfliers=False)

        for patch, color in zip(bp['boxes'], valid_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        for median in bp['medians']:
            median.set_color('red')
            median.set_linewidth(2)

        # Individual points with jitter
        if show_individual:
            for i, (pos, d) in enumerate(zip(valid_positions, valid_data)):
                jitter = np.random.normal(0, 0.05, size=len(d))
                ax.scatter(pos + jitter, d, s=12, alpha=0.3, c='black', zorder=3)

        # Mean trajectory
        if show_mean:
            means = [np.mean(d) for d in valid_data]
            ax.plot(valid_positions, means, 'o-', color='darkblue', linewidth=2.5,
                    markersize=8, zorder=5, label='Mean' if feat_idx == 0 else None)

            # Confidence interval
            if show_ci and len(valid_data) > 0:
                cis_low = []
                cis_high = []
                for d in valid_data:
                    if len(d) >= 2:
                        ci = stats.sem(d) * 1.96
                        cis_low.append(np.mean(d) - ci)
                        cis_high.append(np.mean(d) + ci)
                    else:
                        cis_low.append(np.mean(d))
                        cis_high.append(np.mean(d))

                ax.fill_between(valid_positions, cis_low, cis_high,
                               alpha=0.2, color='darkblue')

    # Labels
    ax.set_xticks(positions)
    ax.set_xticklabels(ordered_times, fontsize=DEFAULT_FONTSIZE['tick'])
    ax.set_xlabel('Time Point', fontsize=DEFAULT_FONTSIZE['label'])
    ax.set_ylabel('Methylation (beta)', fontsize=DEFAULT_FONTSIZE['label'])

    if title:
        ax.set_title(title, fontsize=DEFAULT_FONTSIZE['title'], fontweight='bold')
    elif len(feature_names) == 1:
        ax.set_title(feature_names[0], fontsize=DEFAULT_FONTSIZE['title'], fontweight='bold')

    # Styling
    _apply_base_style(ax)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    if show_mean:
        ax.legend(loc='best', fontsize=DEFAULT_FONTSIZE['legend'])

    if save_path:
        fig.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')

    return fig, ax


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: Union[np.ndarray, Dict[str, np.ndarray]],
    model_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (8, 8),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    colors: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes, Dict[str, float]]:
    """
    Plot ROC curve(s) for binary classification.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_scores : np.ndarray or dict
        Predicted probabilities. Can be a single array or dict mapping
        model names to score arrays.
    model_names : list, optional
        Names for each model (if y_scores is a list of arrays).
    figsize : tuple, optional
        Figure size. Default is (8, 8).
    title : str, optional
        Plot title.
    save_path : str, optional
        Path to save figure.
    colors : list, optional
        List of colors for each curve.
    ax : plt.Axes, optional
        Existing axes to plot on.

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure object.
    ax : plt.Axes
        Matplotlib axes object.
    auc_scores : dict
        Dictionary mapping model names to AUC scores.

    Examples
    --------
    >>> fig, ax, aucs = plot_roc_curve(
    ...     y_true=labels,
    ...     y_scores={'Baseline': probs_base, 'Optimized': probs_opt}
    ... )
    """
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Handle different input formats
    if isinstance(y_scores, dict):
        scores_dict = y_scores
    elif isinstance(y_scores, np.ndarray):
        name = model_names[0] if model_names else 'Model'
        scores_dict = {name: y_scores}
    else:
        raise ValueError("y_scores must be array or dict")

    # Default colors
    if colors is None:
        color_list = list(DEFAULT_COLORS.values())
    else:
        color_list = colors

    # Plot diagonal reference
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Random (AUC=0.500)')

    # Plot ROC curves
    auc_scores = {}
    for i, (name, scores) in enumerate(scores_dict.items()):
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc_score = auc(fpr, tpr)
        auc_scores[name] = auc_score

        color = color_list[i % len(color_list)]
        ax.plot(fpr, tpr, color=color, lw=2.5,
                label=f'{name} (AUC={auc_score:.3f})')

        # Fill under curve for last model
        if i == len(scores_dict) - 1:
            ax.fill_between(fpr, tpr, alpha=0.15, color=color)

    # Styling
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate (1 - Specificity)',
                  fontsize=DEFAULT_FONTSIZE['label'])
    ax.set_ylabel('True Positive Rate (Sensitivity)',
                  fontsize=DEFAULT_FONTSIZE['label'])

    if title:
        ax.set_title(title, fontsize=DEFAULT_FONTSIZE['title'], fontweight='bold')
    else:
        ax.set_title('ROC Curve', fontsize=DEFAULT_FONTSIZE['title'], fontweight='bold')

    ax.legend(loc='lower right', fontsize=DEFAULT_FONTSIZE['legend'],
              frameon=True, fancybox=True)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    if save_path:
        fig.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')

    return fig, ax, auc_scores


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    figsize: Tuple[int, int] = (8, 7),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    cmap: str = 'Blues',
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes, np.ndarray]:
    """
    Plot confusion matrix for classification results.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    class_names : list, optional
        Names of classes for axis labels.
    normalize : bool, optional
        Whether to normalize by row (true labels). Default is True.
    figsize : tuple, optional
        Figure size. Default is (8, 7).
    title : str, optional
        Plot title.
    save_path : str, optional
        Path to save figure.
    cmap : str, optional
        Colormap name. Default is 'Blues'.
    ax : plt.Axes, optional
        Existing axes to plot on.

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure object.
    ax : plt.Axes
        Matplotlib axes object.
    cm : np.ndarray
        Confusion matrix array.

    Examples
    --------
    >>> fig, ax, cm = plot_confusion_matrix(
    ...     y_true=true_labels,
    ...     y_pred=predicted_labels,
    ...     class_names=['4W', '8W', '12W']
    ... )
    """
    # Compute confusion matrix
    if class_names is None:
        class_names = sorted(list(set(y_true) | set(y_pred)))

    cm = confusion_matrix(y_true, y_pred, labels=class_names)

    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)
    else:
        cm_normalized = cm

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Create heatmap
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=cmap,
                   vmin=0, vmax=1 if normalize else None)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Proportion' if normalize else 'Count',
                   rotation=270, labelpad=15)

    # Add text annotations
    thresh = 0.5 if normalize else cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = 'white' if cm_normalized[i, j] > thresh else 'black'
            if normalize:
                text = f'{cm[i, j]}\n({cm_normalized[i, j]:.1%})'
            else:
                text = str(cm[i, j])
            ax.text(j, i, text, ha='center', va='center', color=color,
                    fontsize=DEFAULT_FONTSIZE['tick'], fontweight='bold')

    # Labels
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=DEFAULT_FONTSIZE['tick'])
    ax.set_yticklabels(class_names, fontsize=DEFAULT_FONTSIZE['tick'])
    ax.set_xlabel('Predicted Label', fontsize=DEFAULT_FONTSIZE['label'])
    ax.set_ylabel('True Label', fontsize=DEFAULT_FONTSIZE['label'])

    if title:
        ax.set_title(title, fontsize=DEFAULT_FONTSIZE['title'], fontweight='bold')
    else:
        ax.set_title('Confusion Matrix', fontsize=DEFAULT_FONTSIZE['title'], fontweight='bold')

    if save_path:
        fig.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')

    return fig, ax, cm


def plot_roc_curves(
    y_true_list: List[np.ndarray],
    y_prob_list: List[np.ndarray],
    model_names: List[str],
    figsize: Tuple[int, int] = (8, 8),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    colors: Optional[List[str]] = None,
    show_ci: bool = False,
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes, Dict[str, float]]:
    """
    Plot multiple ROC curves for model comparison.

    This is a convenience wrapper around plot_roc_curve for comparing
    multiple models with different true labels and predictions.

    Parameters
    ----------
    y_true_list : list of np.ndarray
        List of true binary label arrays for each model.
    y_prob_list : list of np.ndarray
        List of predicted probability arrays for each model.
    model_names : list of str
        Names for each model.
    figsize : tuple, optional
        Figure size. Default is (8, 8).
    title : str, optional
        Plot title.
    save_path : str, optional
        Path to save figure.
    colors : list, optional
        List of colors for each curve.
    show_ci : bool, optional
        Whether to show confidence interval (requires bootstrapping). Default is False.
    ax : plt.Axes, optional
        Existing axes to plot on.

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure object.
    ax : plt.Axes
        Matplotlib axes object.
    auc_scores : dict
        Dictionary mapping model names to AUC scores.

    Examples
    --------
    >>> y_true_list = [labels_model1, labels_model2]
    >>> y_prob_list = [probs_model1, probs_model2]
    >>> model_names = ['Baseline', 'Optimized']
    >>> fig, ax, aucs = plot_roc_curves(y_true_list, y_prob_list, model_names)
    """
    if len(y_true_list) != len(y_prob_list) or len(y_true_list) != len(model_names):
        raise ValueError("y_true_list, y_prob_list, and model_names must have same length")

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Default colors
    if colors is None:
        color_list = list(DEFAULT_COLORS.values())
    else:
        color_list = colors

    # Plot diagonal reference
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Random (AUC=0.500)')

    # Plot ROC curves
    auc_scores = {}
    for i, (y_true, y_prob, name) in enumerate(zip(y_true_list, y_prob_list, model_names)):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = auc(fpr, tpr)
        auc_scores[name] = auc_score

        color = color_list[i % len(color_list)]
        ax.plot(fpr, tpr, color=color, lw=2.5,
                label=f'{name} (AUC={auc_score:.3f})')

        # Fill under curve for last model
        if i == len(model_names) - 1:
            ax.fill_between(fpr, tpr, alpha=0.15, color=color)

        # Bootstrap CI if requested
        if show_ci:
            n_bootstrap = 100
            tprs_boot = []
            mean_fpr = np.linspace(0, 1, 100)

            for _ in range(n_bootstrap):
                indices = np.random.choice(len(y_true), len(y_true), replace=True)
                if len(np.unique(y_true[indices])) < 2:
                    continue
                fpr_boot, tpr_boot, _ = roc_curve(y_true[indices], y_prob[indices])
                tpr_interp = np.interp(mean_fpr, fpr_boot, tpr_boot)
                tprs_boot.append(tpr_interp)

            if len(tprs_boot) > 10:
                tprs_boot = np.array(tprs_boot)
                tpr_lower = np.percentile(tprs_boot, 2.5, axis=0)
                tpr_upper = np.percentile(tprs_boot, 97.5, axis=0)
                ax.fill_between(mean_fpr, tpr_lower, tpr_upper,
                               alpha=0.1, color=color)

    # Styling
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate (1 - Specificity)',
                  fontsize=DEFAULT_FONTSIZE['label'])
    ax.set_ylabel('True Positive Rate (Sensitivity)',
                  fontsize=DEFAULT_FONTSIZE['label'])

    if title:
        ax.set_title(title, fontsize=DEFAULT_FONTSIZE['title'], fontweight='bold')
    else:
        ax.set_title('ROC Curve Comparison', fontsize=DEFAULT_FONTSIZE['title'],
                     fontweight='bold')

    ax.legend(loc='lower right', fontsize=DEFAULT_FONTSIZE['legend'],
              frameon=True, fancybox=True)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    if save_path:
        fig.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')

    return fig, ax, auc_scores
