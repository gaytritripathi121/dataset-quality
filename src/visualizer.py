import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
from utils.helpers import ensure_dir
from utils.logger import default_logger as logger


class Visualizer:
    """Generates visualizations for dataset quality analysis"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        viz_config = self.config.get('visualization', {})
        
        self.figure_size = tuple(viz_config.get('figure_size', [12, 8]))
        self.dpi = viz_config.get('dpi', 100)
        self.style = viz_config.get('style', 'seaborn-v0_8-darkgrid')
        
        try:
            plt.style.use(self.style)
        except:
            plt.style.use('default')
        
        sns.set_palette(viz_config.get('color_palette', 'husl'))
    
    def generate_all_plots(
        self,
        output_dir: str,
        df: pd.DataFrame = None,
        profile: Dict = None,
        quality_checks: Dict = None,
        anomaly_results: Dict = None,
        scores: Dict = None
    ) -> List[str]:
        """Generate all visualization plots"""
        logger.info("Generating visualizations...")
        
        output_dir = ensure_dir(output_dir)
        plot_paths = []
        
        if scores:
            plot_paths.append(
                self.plot_score_dashboard(scores, output_dir / "score_dashboard.png")
            )
        
        if df is not None:
            plot_paths.append(
                self.plot_missing_values(df, output_dir / "missing_values.png")
            )
        
        if df is not None:
            plot_paths.append(
                self.plot_class_distribution(df, output_dir / "class_distribution.png")
            )
        
        if anomaly_results and 'combined' in anomaly_results:
            plot_paths.append(
                self.plot_anomaly_scores(
                    anomaly_results['combined'],
                    output_dir / "anomaly_scores.png"
                )
            )
        
        if df is not None:
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                plot_paths.append(
                    self.plot_correlation_matrix(
                        numeric_df,
                        output_dir / "correlation_matrix.png"
                    )
                )
        
        if df is not None:
            plot_paths.append(
                self.plot_feature_distributions(
                    df,
                    output_dir / "feature_distributions.png"
                )
            )
        
        logger.info(f"Generated {len(plot_paths)} visualizations")
        return plot_paths
    
    def plot_score_dashboard(self, scores: Dict, output_path: str) -> str:
        """Create score dashboard visualization"""
        fig, axes = plt.subplots(2, 2, figsize=self.figure_size)
        fig.suptitle('Dataset Quality Dashboard', fontsize=16, fontweight='bold')
        
        ax = axes[0, 0]
        overall_score = scores['overall_score']
        rating = scores['rating']
        
        theta = np.linspace(0, np.pi, 100)
        r = np.ones(100)
        
        if overall_score >= 90:
            color = 'green'
        elif overall_score >= 75:
            color = 'lightgreen'
        elif overall_score >= 60:
            color = 'yellow'
        else:
            color = 'red'
        
        ax.fill_between(theta, 0, r, color=color, alpha=0.3)
        ax.plot(theta, r, color=color, linewidth=2)
        
        score_theta = np.pi * (1 - overall_score / 100)
        ax.plot([0, score_theta], [0, 1], 'k-', linewidth=3)
        ax.scatter([score_theta], [1], s=200, color='black')
        
        ax.set_ylim(0, 1.2)
        ax.set_xlim(0, np.pi)
        ax.axis('off')
        ax.text(np.pi / 2, 0.5, f"{overall_score:.1f}", fontsize=36, ha='center', fontweight='bold')
        ax.text(np.pi / 2, 0.3, rating, fontsize=16, ha='center')
        ax.set_title('Overall Health Score')
        
        ax = axes[0, 1]
        component_scores = scores['component_scores']
        components = list(component_scores.keys())
        values = list(component_scores.values())
        
        bars = ax.barh(components, values, color='steelblue')
        ax.set_xlim(0, 25)
        ax.set_xlabel('Score (0-25)')
        ax.set_title('Component Scores')
        ax.grid(axis='x', alpha=0.3)
        
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(val + 0.5, i, f'{val:.1f}', va='center')
        
        ax = axes[1, 0]
        ax.pie(
            values,
            labels=components,
            autopct='%1.1f%%',
            startangle=90,
            colors=sns.color_palette('pastel')
        )
        ax.set_title('Score Distribution')
        
        ax = axes[1, 1]
        ax.axis('off')
        ax.text(0.5, 0.7, 'Recommendation:', fontsize=14, fontweight='bold', ha='center')
        ax.text(0.5, 0.5, scores['recommendation'], fontsize=12, ha='center', wrap=True)
        
        if overall_score < 75:
            ax.text(0.5, 0.3, 'Action required', fontsize=12, ha='center', color='red')
        else:
            ax.text(0.5, 0.3, 'Quality acceptable', fontsize=12, ha='center', color='green')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def plot_missing_values(self, df: pd.DataFrame, output_path: str) -> str:
        """Plot missing values heatmap"""
        fig, axes = plt.subplots(1, 2, figsize=self.figure_size)
        
        ax = axes[0]
        missing_data = df.isnull()
        
        if missing_data.any().any():
            sns.heatmap(
                missing_data.T,
                cmap='YlOrRd',
                cbar=True,
                ax=ax,
                yticklabels=df.columns
            )
            ax.set_title('Missing Values Pattern')
            ax.set_xlabel('Sample Index')
        else:
            ax.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', fontsize=14)
            ax.axis('off')
        
        ax = axes[1]
        missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
        missing_pct = missing_pct[missing_pct > 0]
        
        if not missing_pct.empty:
            missing_pct.plot(kind='barh', ax=ax, color='coral')
            ax.set_xlabel('Missing Percentage (%)')
            ax.set_title('Missing Values by Column')
            ax.grid(axis='x', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', fontsize=14)
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def plot_class_distribution(self, df: pd.DataFrame, output_path: str) -> str:
        """Plot class distribution"""
        potential_targets = ['target', 'label', 'class', 'y']
        target_col = None
        
        for col in potential_targets:
            if col in df.columns:
                target_col = col
                break
        
        if target_col is None and len(df.columns) > 0:
            target_col = df.columns[-1]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if target_col and df[target_col].nunique() < 20:
            value_counts = df[target_col].value_counts()
            colors = sns.color_palette('husl', len(value_counts))
            bars = ax.bar(range(len(value_counts)), value_counts.values, color=colors)
            ax.set_xticks(range(len(value_counts)))
            ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
            ax.set_ylabel('Count')
            ax.set_title(f'Class Distribution: {target_col}')
            ax.grid(axis='y', alpha=0.3)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f'{int(height)}',
                    ha='center',
                    va='bottom',
                    fontsize=10
                )
        else:
            ax.text(0.5, 0.5, 'No categorical target found', ha='center', va='center', fontsize=14)
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def plot_anomaly_scores(self, anomaly_results: Dict, output_path: str) -> str:
        """Plot anomaly score distribution"""
        fig, axes = plt.subplots(1, 2, figsize=self.figure_size)
        
        scores = anomaly_results['anomaly_scores']
        labels = anomaly_results['anomaly_labels']
        
        ax = axes[0]
        ax.hist(scores, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(
            scores[labels == 1].min() if any(labels == 1) else 0.8,
            linestyle='--',
            linewidth=2,
            label='Anomaly Threshold'
        )
        ax.set_xlabel('Anomaly Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Anomaly Score Distribution')
        ax.legend()
        ax.grid(alpha=0.3)
        
        ax = axes[1]
        colors = ['red' if label == 1 else 'blue' for label in labels]
        ax.scatter(range(len(scores)), scores, c=colors, alpha=0.6, s=20)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Anomaly Score')
        ax.set_title('Anomaly Scores per Sample')
        ax.grid(alpha=0.3)
        
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor='red', label=f'Anomaly ({(labels == 1).sum()})'),
            Patch(facecolor='blue', label=f'Normal ({(labels == 0).sum()})')
        ])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def plot_correlation_matrix(self, df: pd.DataFrame, output_path: str) -> str:
        """Plot correlation matrix heatmap"""
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        
        ax.set_title('Feature Correlation Matrix')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def plot_feature_distributions(self, df: pd.DataFrame, output_path: str) -> str:
        """Plot distributions of numerical features"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:9]
        
        if len(numeric_cols) == 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'No numerical features', ha='center', va='center', fontsize=14)
            ax.axis('off')
        else:
            n_cols = min(3, len(numeric_cols))
            n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
            axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
            
            for i, col in enumerate(numeric_cols):
                ax = axes[i]
                df[col].hist(bins=30, ax=ax, edgecolor='black', alpha=0.7)
                ax.set_title(col)
                ax.set_xlabel('Value')
                ax.set_ylabel('Frequency')
                ax.grid(alpha=0.3)
            
            for i in range(len(numeric_cols), len(axes)):
                axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
