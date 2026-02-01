"""
Publication-ready figure generation following journal guidelines.

This module provides the PublicationFigureGenerator class for creating
figures that conform to scientific journal formatting requirements.
Supports Nature, Cell, and custom default styles with appropriate
font sizes, DPI settings, color palettes, and layout parameters.

Examples
--------
>>> generator = PublicationFigureGenerator(style='nature', output_dir='figures')
>>> fig = generator.create_figure_methylation_landscape(data, sample_info)
>>> paths = generator.save_figure(fig, 'figure1', formats=['pdf', 'png', 'svg'])
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import logging

from .plots import (
    plot_pca_visualization,
    plot_volcano,
    plot_heatmap,
    plot_feature_importance,
    plot_boxplot_comparison,
    plot_roc_curve,
    plot_confusion_matrix,
    DEFAULT_COLORS,
    DEFAULT_FONTSIZE,
    _apply_base_style,
)

logger = logging.getLogger(__name__)

# Conversion constant
MM_TO_INCH = 1 / 25.4


class PublicationFigureGenerator:
    """
    Generate publication-ready figures following journal guidelines.

    Supports multiple journal styles (Nature, Cell, default) with
    appropriate font sizes, DPI, and formatting. Figures are created
    with consistent styling and can be saved in multiple formats
    suitable for journal submission.

    Parameters
    ----------
    style : str, optional
        Journal style preset. One of 'nature', 'cell', or 'default'.
        Default is 'nature'.
    output_dir : str or Path, optional
        Directory for saving figures. Default is 'figures'.

    Attributes
    ----------
    style_name : str
        Name of the active journal style.
    style_config : dict
        Configuration dictionary for the active style.
    output_dir : Path
        Path to the output directory.

    Examples
    --------
    >>> gen = PublicationFigureGenerator(style='nature')
    >>> gen.set_style()
    >>> fig = gen.create_figure_methylation_landscape(data, sample_info)
    >>> gen.save_figure(fig, 'figure1')
    """

    STYLES = {
        'nature': {
            'font_size': 8,
            'title_size': 9,
            'label_size': 8,
            'tick_size': 7,
            'legend_size': 7,
            'dpi': 300,
            'font_family': 'Arial',
            'figure_width': 180,   # mm, double column
            'figure_width_single': 90,  # mm, single column
            'line_width': 1.0,
            'axes_linewidth': 0.8,
            'panel_label_size': 12,
        },
        'cell': {
            'font_size': 7,
            'title_size': 8,
            'label_size': 7,
            'tick_size': 6,
            'legend_size': 6,
            'dpi': 300,
            'font_family': 'Helvetica',
            'figure_width': 174,   # mm
            'figure_width_single': 85,
            'line_width': 0.75,
            'axes_linewidth': 0.6,
            'panel_label_size': 11,
        },
        'default': {
            'font_size': 10,
            'title_size': 12,
            'label_size': 10,
            'tick_size': 9,
            'legend_size': 9,
            'dpi': 150,
            'font_family': 'sans-serif',
            'figure_width': 200,
            'figure_width_single': 100,
            'line_width': 1.5,
            'axes_linewidth': 1.0,
            'panel_label_size': 14,
        }
    }

    # Colorblind-friendly palette (Nature-recommended)
    PALETTE = {
        'primary': '#E64B35',
        'secondary': '#4DBBD5',
        'tertiary': '#00A087',
        'quaternary': '#3C5488',
        'quinary': '#F39B7F',
        'senary': '#91D1C2',
        'neutral': '#808080',
        'light_gray': '#B0B0B0',
    }

    PALETTE_LIST = [
        '#E64B35', '#4DBBD5', '#00A087', '#3C5488',
        '#F39B7F', '#91D1C2', '#8491B4', '#B09C85',
    ]

    def __init__(self, style: str = 'nature', output_dir: str = 'figures'):
        if style not in self.STYLES:
            raise ValueError(
                f"Unknown style '{style}'. Choose from: {list(self.STYLES.keys())}"
            )
        self.style_name = style
        self.style_config = self.STYLES[style].copy()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.set_style()

    def set_style(self) -> None:
        """
        Apply journal-specific matplotlib style settings globally.

        Configures font family, font sizes, line widths, DPI, and
        spine appearance to match the selected journal style.
        """
        cfg = self.style_config
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': [cfg['font_family'], 'Arial', 'Helvetica',
                                'DejaVu Sans'],
            'font.size': cfg['font_size'],
            'axes.labelsize': cfg['label_size'],
            'axes.titlesize': cfg['title_size'],
            'xtick.labelsize': cfg['tick_size'],
            'ytick.labelsize': cfg['tick_size'],
            'legend.fontsize': cfg['legend_size'],
            'figure.dpi': cfg['dpi'],
            'savefig.dpi': cfg['dpi'],
            'axes.linewidth': cfg['axes_linewidth'],
            'lines.linewidth': cfg['line_width'],
            'xtick.major.width': cfg['axes_linewidth'],
            'ytick.major.width': cfg['axes_linewidth'],
            'axes.spines.top': False,
            'axes.spines.right': False,
            'mathtext.fontset': 'custom',
            'mathtext.rm': cfg['font_family'],
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.05,
        })
        logger.info("Applied '%s' journal style settings", self.style_name)

    def _get_figsize(
        self,
        width_mm: Optional[float] = None,
        height_ratio: float = 0.75,
        single_column: bool = False
    ) -> Tuple[float, float]:
        """
        Calculate figure size in inches from millimeter specifications.

        Parameters
        ----------
        width_mm : float, optional
            Figure width in mm. If None, uses style default.
        height_ratio : float, optional
            Height as ratio of width. Default is 0.75.
        single_column : bool, optional
            Whether to use single-column width. Default is False.

        Returns
        -------
        tuple of float
            (width_inches, height_inches)
        """
        if width_mm is None:
            key = 'figure_width_single' if single_column else 'figure_width'
            width_mm = self.style_config[key]
        w = width_mm * MM_TO_INCH
        h = w * height_ratio
        return (w, h)

    def _add_panel_label(
        self,
        ax: plt.Axes,
        label: str,
        x: float = -0.12,
        y: float = 1.08
    ) -> None:
        """
        Add a panel label (A, B, C, ...) to axes.

        Parameters
        ----------
        ax : plt.Axes
            Target axes.
        label : str
            Panel label text.
        x, y : float
            Position in axes coordinates.
        """
        ax.text(
            x, y, label,
            transform=ax.transAxes,
            fontsize=self.style_config['panel_label_size'],
            fontweight='bold',
            va='top',
            ha='left',
        )

    # ------------------------------------------------------------------
    # Publication Figure Methods
    # ------------------------------------------------------------------

    def create_figure_methylation_landscape(
        self,
        data: pd.DataFrame,
        sample_info: pd.DataFrame,
        diff_results: Optional[pd.DataFrame] = None,
        include_density: bool = True,
        pvalue_col: str = 'pvalue',
        effect_col: str = 'effect_size',
    ) -> plt.Figure:
        """
        Create global methylation landscape visualization.

        Generates a multi-panel figure with:
        - Panel A: PCA of global methylation (with optional density contours)
        - Panel B: Volcano plot of differential methylation (if diff_results provided)

        Parameters
        ----------
        data : pd.DataFrame
            Methylation data matrix (features x samples or samples x features).
        sample_info : pd.DataFrame
            Sample metadata with at least 'sample_id' and group columns.
        diff_results : pd.DataFrame, optional
            Differential methylation results with p-value and effect size columns.
        include_density : bool, optional
            Whether to overlay kernel density contours on PCA. Default is True.
        pvalue_col : str, optional
            Column name for p-values in diff_results. Default is 'pvalue'.
        effect_col : str, optional
            Column name for effect sizes in diff_results. Default is 'effect_size'.

        Returns
        -------
        fig : plt.Figure
            Matplotlib figure object.

        Examples
        --------
        >>> gen = PublicationFigureGenerator(style='nature')
        >>> fig = gen.create_figure_methylation_landscape(
        ...     data=methyl_data,
        ...     sample_info=sample_mapping,
        ...     diff_results=diff_df
        ... )
        """
        has_volcano = diff_results is not None
        ncols = 2 if has_volcano else 1
        width_ratio = [1, 1] if has_volcano else [1]

        figsize = self._get_figsize(height_ratio=0.5 if has_volcano else 0.8)
        fig, axes = plt.subplots(1, ncols, figsize=figsize,
                                 gridspec_kw={'width_ratios': width_ratio})

        if ncols == 1:
            axes = [axes]

        # ---- Panel A: PCA ----
        ax_pca = axes[0]

        # Determine group column
        group_col = None
        for candidate in ['time_point', 'group', 'binary_class', 'multi_class']:
            if candidate in sample_info.columns:
                group_col = candidate
                break

        if group_col is None:
            raise ValueError(
                "sample_info must contain a group column "
                "(e.g., 'time_point', 'group', 'binary_class')"
            )

        # Build labels Series
        if 'sample_id' in sample_info.columns:
            labels = sample_info.set_index('sample_id')[group_col]
        else:
            labels = sample_info[group_col]

        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        # Orient data as samples x features
        if data.shape[0] < data.shape[1]:
            X_df = data.T
        else:
            X_df = data

        common = X_df.index.intersection(labels.index)
        X = np.nan_to_num(X_df.loc[common].values, nan=0)
        y = labels.loc[common]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        unique_labels = y.unique()
        colors_map = {lbl: self.PALETTE_LIST[i % len(self.PALETTE_LIST)]
                      for i, lbl in enumerate(unique_labels)}

        for lbl in unique_labels:
            mask = (y == lbl).values
            n = mask.sum()
            ax_pca.scatter(
                X_pca[mask, 0], X_pca[mask, 1],
                c=colors_map[lbl], s=30, alpha=0.7,
                edgecolors='white', linewidth=0.5,
                label=f'{lbl} (n={n})'
            )

        # Density contours
        if include_density:
            try:
                from scipy.stats import gaussian_kde
                xy = X_pca.T
                kde = gaussian_kde(xy)
                xmin, xmax = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
                ymin, ymax = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
                xx, yy = np.mgrid[xmin:xmax:50j, ymin:ymax:50j]
                positions = np.vstack([xx.ravel(), yy.ravel()])
                z = kde(positions).reshape(xx.shape)
                ax_pca.contour(xx, yy, z, levels=4, colors='gray',
                               alpha=0.3, linewidths=0.5)
            except Exception:
                pass  # Skip density if it fails

        var_exp = pca.explained_variance_ratio_ * 100
        ax_pca.set_xlabel(f'PC1 ({var_exp[0]:.1f}%)')
        ax_pca.set_ylabel(f'PC2 ({var_exp[1]:.1f}%)')
        ax_pca.legend(loc='best', frameon=True, fancybox=False, edgecolor='gray')
        ax_pca.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        self._add_panel_label(ax_pca, 'A')

        # ---- Panel B: Volcano ----
        if has_volcano:
            ax_vol = axes[1]
            pvalues = diff_results[pvalue_col].values
            effects = diff_results[effect_col].values
            feature_names = None
            if 'cpg' in diff_results.columns:
                feature_names = diff_results['cpg'].values
            elif 'feature' in diff_results.columns:
                feature_names = diff_results['feature'].values

            plot_volcano(
                pvalues=pvalues,
                effect_sizes=effects,
                feature_names=feature_names,
                ax=ax_vol
            )
            self._add_panel_label(ax_vol, 'B')

        fig.tight_layout()
        return fig

    def create_figure_feature_selection(
        self,
        selection_results: Dict[str, Any],
        show_overlap: bool = True,
    ) -> plt.Figure:
        """
        Create feature selection results visualization across levels.

        Generates a multi-panel figure showing:
        - Panel A: Feature count per selection level (bar chart)
        - Panel B: Overlap between selection methods (if show_overlap=True)
        - Panel C: Top feature importance scores

        Parameters
        ----------
        selection_results : dict
            Dictionary with keys per selection level. Each value should
            contain at minimum a list of selected features. Expected keys:
            - 'level_features': dict mapping level names to feature lists
            - 'importance_df': pd.DataFrame with feature importance scores
            - 'overlap_matrix': pd.DataFrame of pairwise overlap counts (optional)
        show_overlap : bool, optional
            Whether to show overlap panel. Default is True.

        Returns
        -------
        fig : plt.Figure
            Matplotlib figure object.

        Examples
        --------
        >>> results = {
        ...     'level_features': {'L1': [...], 'L2': [...], 'L3': [...]},
        ...     'importance_df': importance_df,
        ... }
        >>> fig = gen.create_figure_feature_selection(results)
        """
        level_features = selection_results.get('level_features', {})
        importance_df = selection_results.get('importance_df', None)
        overlap_matrix = selection_results.get('overlap_matrix', None)

        ncols = 2 + (1 if show_overlap and overlap_matrix is not None else 0)
        figsize = self._get_figsize(height_ratio=0.45)
        fig, axes = plt.subplots(1, ncols, figsize=figsize)
        if ncols == 1:
            axes = [axes]

        panel_idx = 0
        panel_labels = 'ABCDEFGH'

        # Panel A: Feature counts per level
        ax = axes[panel_idx]
        if level_features:
            levels = list(level_features.keys())
            counts = [len(v) for v in level_features.values()]
            bars = ax.barh(range(len(levels)), counts,
                          color=self.PALETTE_LIST[:len(levels)],
                          edgecolor='white', linewidth=0.5)
            ax.set_yticks(range(len(levels)))
            ax.set_yticklabels(levels)
            ax.set_xlabel('Number of Features')
            ax.set_title('Features per Selection Level')

            # Add count labels
            for bar, count in zip(bars, counts):
                ax.text(bar.get_width() + max(counts) * 0.02,
                        bar.get_y() + bar.get_height() / 2,
                        f'{count:,}', va='center',
                        fontsize=self.style_config['tick_size'])
        self._add_panel_label(ax, panel_labels[panel_idx])
        panel_idx += 1

        # Panel B: Overlap heatmap (optional)
        if show_overlap and overlap_matrix is not None:
            ax = axes[panel_idx]
            import seaborn as sns
            sns.heatmap(overlap_matrix, annot=True, fmt='d', cmap='YlOrRd',
                       ax=ax, cbar_kws={'label': 'Shared Features'})
            ax.set_title('Feature Overlap')
            self._add_panel_label(ax, panel_labels[panel_idx])
            panel_idx += 1

        # Panel C: Top feature importance
        if importance_df is not None:
            ax = axes[panel_idx]
            plot_feature_importance(importance_df, top_n=15, ax=ax)
            ax.set_title('Top Feature Importance')
            self._add_panel_label(ax, panel_labels[panel_idx])
        panel_idx += 1

        fig.tight_layout()
        return fig

    def create_figure_model_performance(
        self,
        evaluation_results: Dict[str, Any],
        metrics: List[str] = None,
    ) -> plt.Figure:
        """
        Create model performance comparison figure.

        Generates a multi-panel figure with:
        - Panel A: ROC curves for binary classification
        - Panel B: Confusion matrix for multi-class classification
        - Panel C: Metric comparison bar chart

        Parameters
        ----------
        evaluation_results : dict
            Dictionary containing evaluation data. Expected keys:
            - 'binary': dict with 'y_true', 'y_scores' (dict of model -> probs)
            - 'multiclass': dict with 'y_true', 'y_pred', 'class_names'
            - 'metrics': dict of model_name -> dict of metric_name -> value
        metrics : list of str, optional
            Metrics to display. Default is ['accuracy', 'auc', 'f1'].

        Returns
        -------
        fig : plt.Figure
            Matplotlib figure object.

        Examples
        --------
        >>> results = {
        ...     'binary': {'y_true': y, 'y_scores': {'Model': probs}},
        ...     'multiclass': {'y_true': y_mc, 'y_pred': y_pred, 'class_names': ['4W','8W','12W']},
        ...     'metrics': {'Baseline': {'accuracy': 0.6, 'auc': 0.65}, ...}
        ... }
        >>> fig = gen.create_figure_model_performance(results)
        """
        if metrics is None:
            metrics = ['accuracy', 'auc', 'f1']

        figsize = self._get_figsize(height_ratio=0.38)
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(1, 3, width_ratios=[1, 1, 1.2], wspace=0.35)

        # ---- Panel A: ROC Curves ----
        binary = evaluation_results.get('binary', {})
        ax1 = fig.add_subplot(gs[0])

        if binary:
            y_true = binary['y_true']
            y_scores = binary.get('y_scores', {})
            plot_roc_curve(y_true=y_true, y_scores=y_scores, ax=ax1)
        else:
            ax1.text(0.5, 0.5, 'No binary results', transform=ax1.transAxes,
                    ha='center', va='center')
        self._add_panel_label(ax1, 'A')

        # ---- Panel B: Confusion Matrix ----
        mc = evaluation_results.get('multiclass', {})
        ax2 = fig.add_subplot(gs[1])

        if mc:
            plot_confusion_matrix(
                y_true=mc['y_true'],
                y_pred=mc['y_pred'],
                class_names=mc.get('class_names'),
                ax=ax2
            )
        else:
            ax2.text(0.5, 0.5, 'No multiclass results', transform=ax2.transAxes,
                    ha='center', va='center')
        self._add_panel_label(ax2, 'B')

        # ---- Panel C: Metric Comparison ----
        metrics_data = evaluation_results.get('metrics', {})
        ax3 = fig.add_subplot(gs[2])

        if metrics_data:
            model_names = list(metrics_data.keys())
            x = np.arange(len(metrics))
            width = 0.8 / max(len(model_names), 1)

            for i, model in enumerate(model_names):
                values = [metrics_data[model].get(m, 0) for m in metrics]
                offset = (i - len(model_names) / 2 + 0.5) * width
                bars = ax3.bar(x + offset, values, width,
                              label=model,
                              color=self.PALETTE_LIST[i % len(self.PALETTE_LIST)],
                              alpha=0.85, edgecolor='black', linewidth=0.5)
                # Value labels
                for bar in bars:
                    h = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                            f'{h:.2f}', ha='center', va='bottom',
                            fontsize=self.style_config['tick_size'] - 1,
                            fontweight='bold')

            ax3.set_xticks(x)
            ax3.set_xticklabels([m.upper() for m in metrics], rotation=15, ha='right')
            ax3.set_ylabel('Score')
            ax3.set_ylim(0, 1.15)
            ax3.legend(loc='upper left', frameon=True)
            ax3.axhline(0.5, color='red', linestyle='--', alpha=0.4, linewidth=0.8)
            ax3.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
            ax3.set_title('Performance Comparison')
        self._add_panel_label(ax3, 'C')

        fig.tight_layout()
        return fig

    def create_figure_biomarker_panel(
        self,
        top_features: List[str],
        data: pd.DataFrame,
        labels: pd.Series,
        n_features: int = 6,
    ) -> plt.Figure:
        """
        Create top biomarker panel with boxplots.

        Generates a grid of boxplots showing methylation levels
        across groups for the top discriminative CpG sites.

        Parameters
        ----------
        top_features : list of str
            Feature names (CpG IDs) ranked by importance.
        data : pd.DataFrame
            Methylation data matrix.
        labels : pd.Series
            Group labels indexed by sample ID.
        n_features : int, optional
            Number of top features to display. Default is 6.

        Returns
        -------
        fig : plt.Figure
            Matplotlib figure object.

        Examples
        --------
        >>> fig = gen.create_figure_biomarker_panel(
        ...     top_features=['cg001', 'cg002', 'cg003', 'cg004', 'cg005', 'cg006'],
        ...     data=methyl_data,
        ...     labels=sample_info['group']
        ... )
        """
        features = top_features[:n_features]
        ncols = min(3, n_features)
        nrows = int(np.ceil(n_features / ncols))

        figsize = self._get_figsize(height_ratio=0.5 * nrows)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = np.atleast_2d(axes)

        panel_labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

        for idx, feature in enumerate(features):
            row, col = divmod(idx, ncols)
            ax = axes[row, col]

            try:
                plot_boxplot_comparison(
                    data=data,
                    groups=labels,
                    feature_name=feature,
                    ax=ax
                )
            except (ValueError, KeyError) as e:
                ax.text(0.5, 0.5, f'{feature}\n(not found)',
                       transform=ax.transAxes, ha='center', va='center')
                logger.warning("Could not plot feature '%s': %s", feature, e)

            self._add_panel_label(ax, panel_labels[idx])

        # Hide unused axes
        for idx in range(n_features, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row, col].set_visible(False)

        fig.tight_layout()
        return fig

    def create_figure_enrichment(
        self,
        enrichment_results: pd.DataFrame,
        top_n: int = 15,
        plot_type: str = 'dotplot',
        category_col: str = 'category',
        pvalue_col: str = 'pvalue',
        fold_enrichment_col: str = 'fold_enrichment',
        gene_count_col: str = 'gene_count',
        pathway_col: str = 'pathway',
    ) -> plt.Figure:
        """
        Create pathway enrichment visualization.

        Parameters
        ----------
        enrichment_results : pd.DataFrame
            Enrichment results with pathway, p-value, fold enrichment,
            gene count, and category columns.
        top_n : int, optional
            Number of top pathways to display. Default is 15.
        plot_type : str, optional
            Type of plot: 'dotplot' (bubble chart) or 'barplot'. Default is 'dotplot'.
        category_col : str, optional
            Column name for pathway category. Default is 'category'.
        pvalue_col : str, optional
            Column name for p-values. Default is 'pvalue'.
        fold_enrichment_col : str, optional
            Column name for fold enrichment. Default is 'fold_enrichment'.
        gene_count_col : str, optional
            Column name for gene counts. Default is 'gene_count'.
        pathway_col : str, optional
            Column name for pathway names. Default is 'pathway'.

        Returns
        -------
        fig : plt.Figure
            Matplotlib figure object.

        Examples
        --------
        >>> fig = gen.create_figure_enrichment(
        ...     enrichment_results=enrichment_df,
        ...     top_n=15,
        ...     plot_type='dotplot'
        ... )
        """
        df = enrichment_results.copy()
        df['neg_log_p'] = -np.log10(df[pvalue_col].clip(lower=1e-300))

        # Select top pathways by p-value
        df = df.nsmallest(top_n, pvalue_col)

        figsize = self._get_figsize(height_ratio=0.7, single_column=False)
        fig, ax = plt.subplots(figsize=figsize)

        # Assign colors by category
        if category_col in df.columns:
            categories = df[category_col].unique()
            cat_colors = {cat: self.PALETTE_LIST[i % len(self.PALETTE_LIST)]
                         for i, cat in enumerate(categories)}
        else:
            cat_colors = {}

        # Sort by category then p-value
        if category_col in df.columns:
            df = df.sort_values([category_col, 'neg_log_p'], ascending=[True, False])
        else:
            df = df.sort_values('neg_log_p', ascending=True)

        y_positions = range(len(df))

        if plot_type == 'dotplot':
            for cat in (categories if category_col in df.columns else [None]):
                if cat is not None:
                    subset = df[df[category_col] == cat]
                    color = cat_colors[cat]
                else:
                    subset = df
                    color = self.PALETTE['primary']

                mask = df.index.isin(subset.index)
                y_pos = [i for i, m in enumerate(mask) if m]

                ax.scatter(
                    subset['neg_log_p'].values,
                    y_pos,
                    s=subset[gene_count_col].values * 25 if gene_count_col in df.columns else 50,
                    c=color,
                    alpha=0.7,
                    edgecolors='white',
                    linewidths=1.0,
                    label=cat,
                    zorder=3,
                )

            ax.set_xlabel(r'-log$_{10}$(P-value)')

            # Size legend
            if gene_count_col in df.columns:
                size_values = [3, 8, 15]
                max_count = df[gene_count_col].max()
                if max_count > 15:
                    size_values = [5, 10, int(max_count)]
                legend_elements = [
                    plt.scatter([], [], s=v * 25, c='gray', alpha=0.5,
                               edgecolors='white', label=str(v))
                    for v in size_values
                ]
                size_legend = ax.legend(
                    handles=legend_elements, title='Gene Count',
                    loc='lower right', frameon=True,
                    fontsize=self.style_config['legend_size']
                )
                ax.add_artist(size_legend)

        elif plot_type == 'barplot':
            if category_col in df.columns:
                bar_colors = [cat_colors.get(c, self.PALETTE['neutral'])
                             for c in df[category_col]]
            else:
                bar_colors = self.PALETTE['primary']

            ax.barh(y_positions, df['neg_log_p'].values,
                   color=bar_colors, edgecolor='white', linewidth=0.5)
            ax.set_xlabel(r'-log$_{10}$(P-value)')

        # Common formatting
        ax.set_yticks(list(y_positions))
        ax.set_yticklabels(df[pathway_col].values)

        # Significance threshold lines
        ax.axvline(-np.log10(0.05), color='gray', linestyle='--',
                   linewidth=0.8, alpha=0.5)
        ax.axvline(-np.log10(0.01), color='gray', linestyle=':',
                   linewidth=0.8, alpha=0.5)

        # Category legend
        if category_col in df.columns and len(categories) > 1:
            cat_handles = [
                mpatches.Patch(color=cat_colors[c], label=c, alpha=0.7)
                for c in categories
            ]
            ax.legend(handles=cat_handles, loc='upper right', frameon=True)

        ax.set_title('Pathway Enrichment Analysis', fontweight='bold')
        ax.grid(axis='x', alpha=0.2, linestyle='--')

        fig.tight_layout()
        return fig

    def create_supplementary_figures(
        self,
        all_results: Dict[str, Any],
    ) -> Dict[str, plt.Figure]:
        """
        Generate all supplementary figures from analysis results.

        This method dispatches to individual figure creation methods
        based on the available data in all_results.

        Parameters
        ----------
        all_results : dict
            Dictionary containing all analysis results. Recognized keys:
            - 'methylation_data': pd.DataFrame
            - 'sample_info': pd.DataFrame
            - 'diff_results': pd.DataFrame
            - 'selection_results': dict
            - 'evaluation_results': dict
            - 'enrichment_results': pd.DataFrame
            - 'top_features': list
            - 'labels': pd.Series

        Returns
        -------
        dict of str -> plt.Figure
            Dictionary mapping figure names to Figure objects.

        Examples
        --------
        >>> all_results = {'methylation_data': data, 'sample_info': info, ...}
        >>> supp_figs = gen.create_supplementary_figures(all_results)
        >>> for name, fig in supp_figs.items():
        ...     gen.save_figure(fig, f'supplementary_{name}')
        """
        figures = {}

        # S1: Methylation landscape
        if 'methylation_data' in all_results and 'sample_info' in all_results:
            try:
                fig = self.create_figure_methylation_landscape(
                    data=all_results['methylation_data'],
                    sample_info=all_results['sample_info'],
                    diff_results=all_results.get('diff_results'),
                )
                figures['methylation_landscape'] = fig
            except Exception as e:
                logger.warning("Failed to create methylation landscape: %s", e)

        # S2: Feature selection
        if 'selection_results' in all_results:
            try:
                fig = self.create_figure_feature_selection(
                    selection_results=all_results['selection_results'],
                )
                figures['feature_selection'] = fig
            except Exception as e:
                logger.warning("Failed to create feature selection figure: %s", e)

        # S3: Model performance
        if 'evaluation_results' in all_results:
            try:
                fig = self.create_figure_model_performance(
                    evaluation_results=all_results['evaluation_results'],
                )
                figures['model_performance'] = fig
            except Exception as e:
                logger.warning("Failed to create model performance figure: %s", e)

        # S4: Biomarker panel
        if 'top_features' in all_results and 'methylation_data' in all_results:
            labels = all_results.get('labels')
            if labels is not None:
                try:
                    fig = self.create_figure_biomarker_panel(
                        top_features=all_results['top_features'],
                        data=all_results['methylation_data'],
                        labels=labels,
                    )
                    figures['biomarker_panel'] = fig
                except Exception as e:
                    logger.warning("Failed to create biomarker panel: %s", e)

        # S5: Enrichment
        if 'enrichment_results' in all_results:
            try:
                fig = self.create_figure_enrichment(
                    enrichment_results=all_results['enrichment_results'],
                )
                figures['enrichment'] = fig
            except Exception as e:
                logger.warning("Failed to create enrichment figure: %s", e)

        logger.info("Generated %d supplementary figures", len(figures))
        return figures

    def save_figure(
        self,
        fig: plt.Figure,
        name: str,
        formats: Optional[List[str]] = None,
        close: bool = True,
    ) -> List[Path]:
        """
        Save figure in multiple formats for publication.

        Parameters
        ----------
        fig : plt.Figure
            Figure to save.
        name : str
            Base filename (without extension).
        formats : list of str, optional
            Output formats. Default is ['png', 'pdf', 'svg'].
        close : bool, optional
            Whether to close the figure after saving. Default is True.

        Returns
        -------
        list of Path
            Paths to saved files.

        Examples
        --------
        >>> paths = gen.save_figure(fig, 'Figure1', formats=['pdf', 'png'])
        >>> print(paths)
        [PosixPath('figures/Figure1.pdf'), PosixPath('figures/Figure1.png')]
        """
        if formats is None:
            formats = ['png', 'pdf', 'svg']

        saved_paths = []
        for fmt in formats:
            path = self.output_dir / f'{name}.{fmt}'
            fig.savefig(
                str(path),
                format=fmt,
                dpi=self.style_config['dpi'],
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none',
            )
            saved_paths.append(path)
            logger.info("Saved %s", path)

        if close:
            plt.close(fig)

        return saved_paths
