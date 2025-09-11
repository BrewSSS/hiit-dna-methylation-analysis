"""
Heatmap visualizations for HIIT DNA methylation analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Any, Tuple
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist


def methylation_heatmap(
    methylation_data: pd.DataFrame,
    metadata: Optional[pd.DataFrame] = None,
    sample_cpg_sites: int = 100,
    cluster_samples: bool = True,
    cluster_sites: bool = True,
    annotation_columns: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = 'RdYlBu_r'
) -> plt.Figure:
    """
    Create a clustered heatmap of DNA methylation data.
    
    Parameters:
    -----------
    methylation_data : pd.DataFrame
        Methylation data (samples x CpG sites)
    metadata : pd.DataFrame, optional
        Sample metadata for annotation
    sample_cpg_sites : int
        Number of CpG sites to sample for visualization
    cluster_samples : bool
        Whether to cluster samples
    cluster_sites : bool
        Whether to cluster CpG sites
    annotation_columns : List[str], optional
        Metadata columns to use for sample annotation
    figsize : Tuple[int, int]
        Figure size
    cmap : str
        Colormap name
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    
    # Sample CpG sites if needed
    if methylation_data.shape[1] > sample_cpg_sites:
        # Select most variable sites
        site_variance = methylation_data.var()
        selected_sites = site_variance.nlargest(sample_cpg_sites).index
        plot_data = methylation_data[selected_sites]
    else:
        plot_data = methylation_data.copy()
    
    # Handle missing values
    plot_data = plot_data.fillna(plot_data.mean())
    
    # Prepare annotation colors if metadata provided
    annotation_colors = None
    row_colors = None
    
    if metadata is not None and annotation_columns:
        available_columns = [col for col in annotation_columns if col in metadata.columns]
        if available_columns:
            # Create annotation dataframe
            common_samples = plot_data.index.intersection(metadata.index)
            annotation_df = metadata.loc[common_samples, available_columns]
            
            # Generate colors for categorical variables
            annotation_colors = {}
            row_colors_list = []
            
            for col in available_columns:
                unique_values = annotation_df[col].unique()
                if len(unique_values) <= 10:  # Categorical
                    colors = sns.color_palette("Set1", n_colors=len(unique_values))
                    color_map = dict(zip(unique_values, colors))
                    annotation_colors[col] = color_map
                    
                    # Add to row colors
                    col_colors = annotation_df[col].map(color_map)
                    row_colors_list.append(col_colors)
            
            if row_colors_list:
                row_colors = pd.concat(row_colors_list, axis=1)
                row_colors = row_colors.loc[plot_data.index]  # Ensure same order
    
    # Create clustered heatmap
    fig = plt.figure(figsize=figsize)
    
    try:
        g = sns.clustermap(
            plot_data.T,  # Transpose so CpG sites are on y-axis
            cmap=cmap,
            center=0.5,  # Center colormap at 50% methylation
            row_cluster=cluster_sites,
            col_cluster=cluster_samples,
            col_colors=row_colors,
            figsize=figsize,
            cbar_kws={'label': 'β-value'},
            xticklabels=False,  # Hide sample labels
            yticklabels=False   # Hide CpG site labels for clarity
        )
        
        # Add title
        g.fig.suptitle('DNA Methylation Heatmap', y=1.02, fontsize=16)
        
        # Add legend for annotations
        if annotation_colors:
            legend_elements = []
            for col, color_map in annotation_colors.items():
                for value, color in color_map.items():
                    legend_elements.append(
                        plt.Rectangle((0, 0), 1, 1, facecolor=color, label=f'{col}: {value}')
                    )
            
            if legend_elements:
                g.ax_heatmap.legend(
                    handles=legend_elements, 
                    bbox_to_anchor=(1.05, 1), 
                    loc='upper left'
                )
        
        return g.fig
        
    except Exception as e:
        # Fallback to simple heatmap if clustering fails
        print(f"Clustering failed: {e}. Creating simple heatmap.")
        
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(plot_data.T, aspect='auto', cmap=cmap, vmin=0, vmax=1)
        ax.set_title('DNA Methylation Heatmap')
        ax.set_xlabel('Samples')
        ax.set_ylabel('CpG Sites')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('β-value')
        
        return fig


def differential_methylation_heatmap(
    group1_data: pd.DataFrame,
    group2_data: pd.DataFrame,
    group1_name: str = 'Group 1',
    group2_name: str = 'Group 2',
    top_sites: int = 50,
    p_value_threshold: float = 0.05,
    effect_size_threshold: float = 0.1,
    figsize: Tuple[int, int] = (12, 8),
    cmap: str = 'RdBu_r'
) -> plt.Figure:
    """
    Create heatmap showing differentially methylated CpG sites.
    
    Parameters:
    -----------
    group1_data : pd.DataFrame
        Methylation data for group 1
    group2_data : pd.DataFrame
        Methylation data for group 2
    group1_name : str
        Name of group 1
    group2_name : str
        Name of group 2
    top_sites : int
        Number of top differential sites to show
    p_value_threshold : float
        P-value threshold for significance
    effect_size_threshold : float
        Effect size threshold
    figsize : Tuple[int, int]
        Figure size
    cmap : str
        Colormap name
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    
    from scipy import stats
    
    # Find common CpG sites
    common_sites = group1_data.columns.intersection(group2_data.columns)
    
    # Calculate differential methylation
    diff_results = []
    
    for site in common_sites:
        values1 = group1_data[site].dropna()
        values2 = group2_data[site].dropna()
        
        if len(values1) >= 3 and len(values2) >= 3:
            # T-test
            stat, p_val = stats.ttest_ind(values1, values2)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(values1) - 1) * values1.var() + 
                                 (len(values2) - 1) * values2.var()) / 
                                (len(values1) + len(values2) - 2))
            
            if pooled_std > 0:
                cohens_d = (values1.mean() - values2.mean()) / pooled_std
            else:
                cohens_d = 0
            
            diff_results.append({
                'site': site,
                'group1_mean': values1.mean(),
                'group2_mean': values2.mean(),
                'mean_diff': values1.mean() - values2.mean(),
                'cohens_d': cohens_d,
                'p_value': p_val,
                'significant': (p_val < p_value_threshold) and (abs(cohens_d) > effect_size_threshold)
            })
    
    # Convert to DataFrame
    diff_df = pd.DataFrame(diff_results)
    
    if len(diff_df) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No differential sites found', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Differential Methylation Heatmap')
        return fig
    
    # Select significant sites
    significant_sites = diff_df[diff_df['significant']]['site']
    
    if len(significant_sites) == 0:
        # Use top sites by effect size if no significant sites
        top_sites_by_effect = diff_df.nlargest(top_sites, 'cohens_d')['site']
        selected_sites = top_sites_by_effect
        title_suffix = f"(Top {top_sites} by Effect Size)"
    else:
        # Use significant sites, limiting to top_sites
        selected_sites = significant_sites[:top_sites]
        title_suffix = f"(Significant Sites, p<{p_value_threshold})"
    
    if len(selected_sites) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No sites to display', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Differential Methylation Heatmap')
        return fig
    
    # Prepare data for heatmap
    heatmap_data = pd.concat([
        group1_data[selected_sites],
        group2_data[selected_sites]
    ], axis=0)
    
    # Create group annotation
    group_annotation = pd.Series(
        [group1_name] * len(group1_data) + [group2_name] * len(group2_data),
        index=heatmap_data.index,
        name='Group'
    )
    
    # Create color map for groups
    group_colors = {group1_name: 'lightblue', group2_name: 'lightcoral'}
    row_colors = group_annotation.map(group_colors)
    
    # Create heatmap
    fig = plt.figure(figsize=figsize)
    
    try:
        g = sns.clustermap(
            heatmap_data.T,
            cmap=cmap,
            center=0.5,
            col_colors=row_colors,
            row_cluster=True,
            col_cluster=False,  # Keep group ordering
            figsize=figsize,
            cbar_kws={'label': 'β-value'},
            xticklabels=False,
            yticklabels=True if len(selected_sites) <= 20 else False
        )
        
        # Add title
        title = f'Differential Methylation: {group1_name} vs {group2_name} {title_suffix}'
        g.fig.suptitle(title, y=1.02, fontsize=14)
        
        # Add group legend
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor=color, label=group)
            for group, color in group_colors.items()
        ]
        g.ax_heatmap.legend(
            handles=legend_elements,
            bbox_to_anchor=(1.05, 1),
            loc='upper left'
        )
        
        return g.fig
        
    except Exception as e:
        print(f"Clustering failed: {e}. Creating simple heatmap.")
        
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(heatmap_data.T, aspect='auto', cmap=cmap, vmin=0, vmax=1)
        
        # Add group boundaries
        group1_end = len(group1_data) - 0.5
        ax.axvline(group1_end, color='black', linewidth=2)
        
        # Labels
        ax.set_xlabel('Samples')
        ax.set_ylabel('CpG Sites')
        ax.set_title(f'Differential Methylation: {group1_name} vs {group2_name}')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('β-value')
        
        # Add group labels
        ax.text(len(group1_data)/2 - 0.5, -0.5, group1_name, ha='center')
        ax.text(len(group1_data) + len(group2_data)/2 - 0.5, -0.5, group2_name, ha='center')
        
        return fig


def hiit_timeline_heatmap(
    methylation_data: pd.DataFrame,
    metadata: pd.DataFrame,
    subject_column: str = 'subject_id',
    time_column: str = 'time_point',
    group_column: str = 'hiit_group',
    top_variable_sites: int = 50,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Create timeline heatmap showing methylation changes over time for HIIT study.
    
    Parameters:
    -----------
    methylation_data : pd.DataFrame
        Methylation data
    metadata : pd.DataFrame
        Sample metadata
    subject_column : str
        Column with subject IDs
    time_column : str
        Column with time points
    group_column : str
        Column with group information
    top_variable_sites : int
        Number of most variable sites to show
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    
    # Check required columns
    required_cols = [subject_column, time_column, group_column]
    missing_cols = [col for col in required_cols if col not in metadata.columns]
    if missing_cols:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f'Missing columns: {missing_cols}', 
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Find most variable CpG sites
    site_variance = methylation_data.var()
    top_sites = site_variance.nlargest(top_variable_sites).index
    
    # Prepare data for heatmap
    plot_data = methylation_data[top_sites]
    
    # Get common samples
    common_samples = plot_data.index.intersection(metadata.index)
    plot_data = plot_data.loc[common_samples]
    sample_metadata = metadata.loc[common_samples]
    
    # Sort by group, subject, and time
    sample_metadata = sample_metadata.sort_values([group_column, subject_column, time_column])
    plot_data = plot_data.loc[sample_metadata.index]
    
    # Create annotations for samples
    annotation_df = sample_metadata[[group_column, time_column]]
    
    # Create color maps
    group_colors = dict(zip(
        sample_metadata[group_column].unique(),
        sns.color_palette("Set1", n_colors=len(sample_metadata[group_column].unique()))
    ))
    
    time_colors = dict(zip(
        sample_metadata[time_column].unique(),
        sns.color_palette("Set2", n_colors=len(sample_metadata[time_column].unique()))
    ))
    
    # Create annotation colors
    row_colors = pd.DataFrame({
        'Group': sample_metadata[group_column].map(group_colors),
        'Time': sample_metadata[time_column].map(time_colors)
    })
    
    # Create heatmap
    fig = plt.figure(figsize=figsize)
    
    try:
        g = sns.clustermap(
            plot_data.T,
            cmap='RdYlBu_r',
            center=0.5,
            col_colors=row_colors,
            row_cluster=True,
            col_cluster=False,  # Maintain temporal ordering
            figsize=figsize,
            cbar_kws={'label': 'β-value'},
            xticklabels=False,
            yticklabels=False
        )
        
        # Add title
        g.fig.suptitle('HIIT Study Timeline: Methylation Changes Over Time', y=1.02, fontsize=16)
        
        # Create legend
        legend_elements = []
        
        # Group legend
        for group, color in group_colors.items():
            legend_elements.append(
                plt.Rectangle((0, 0), 1, 1, facecolor=color, label=f'Group: {group}')
            )
        
        # Time legend
        for time_point, color in time_colors.items():
            legend_elements.append(
                plt.Rectangle((0, 0), 1, 1, facecolor=color, label=f'Time: {time_point}')
            )
        
        if legend_elements:
            g.ax_heatmap.legend(
                handles=legend_elements,
                bbox_to_anchor=(1.05, 1),
                loc='upper left'
            )
        
        return g.fig
        
    except Exception as e:
        print(f"Clustering failed: {e}. Creating simple heatmap.")
        
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(plot_data.T, aspect='auto', cmap='RdYlBu_r', vmin=0, vmax=1)
        
        ax.set_title('HIIT Study Timeline: Methylation Changes Over Time')
        ax.set_xlabel('Samples (ordered by group, subject, time)')
        ax.set_ylabel('CpG Sites')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('β-value')
        
        return fig