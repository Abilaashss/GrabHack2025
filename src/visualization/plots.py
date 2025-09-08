"""
Project Nova - Visualization Module
Comprehensive plotting and visualization for credit scoring models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import shap
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import logging
import yaml
import os
from typing import Dict, Any, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ModelVisualizer:
    """Comprehensive visualization for credit scoring models"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize ModelVisualizer with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.logger = self._setup_logger()
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create plots directory if it doesn't exist
        os.makedirs('results/plots', exist_ok=True)
        os.makedirs('results/visualizations', exist_ok=True)
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/visualization.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def plot_model_performance_comparison(self, metrics_df: pd.DataFrame,
                                        metrics_to_plot: List[str] = None) -> None:
        """Plot model performance comparison"""
        self.logger.info("Creating model performance comparison plot...")
        
        if metrics_to_plot is None:
            metrics_to_plot = ['r2_score', 'rmse', 'mae', 'mape']
        
        # Filter available metrics
        available_metrics = [m for m in metrics_to_plot if m in metrics_df.columns]
        
        if not available_metrics:
            self.logger.warning("No metrics available for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, metric in enumerate(available_metrics[:4]):
            ax = axes[i]
            
            # Sort models by metric
            ascending = metric in ['rmse', 'mae', 'mape']  # Lower is better for these
            sorted_data = metrics_df.sort_values(metric, ascending=ascending)
            
            # Create bar plot
            bars = ax.bar(range(len(sorted_data)), sorted_data[metric])
            ax.set_xticks(range(len(sorted_data)))
            ax.set_xticklabels(sorted_data.index, rotation=45, ha='right')
            ax.set_ylabel(metric.upper())
            ax.set_title(f'Model Comparison - {metric.upper()}')
            
            # Color bars based on performance
            if ascending:  # Lower is better
                colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(bars)))
            else:  # Higher is better
                colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(bars)))
            
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            # Add value labels on bars
            for j, (idx, value) in enumerate(sorted_data[metric].items()):
                ax.text(j, value + (max(sorted_data[metric]) - min(sorted_data[metric])) * 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('results/plots/model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig('results/plots/model_performance_comparison.pdf', bbox_inches='tight')
        plt.close()
        
        # Create interactive plotly version
        self._create_interactive_performance_plot(metrics_df, available_metrics)
    
    def _create_interactive_performance_plot(self, metrics_df: pd.DataFrame,
                                           metrics: List[str]) -> None:
        """Create interactive performance comparison plot"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[m.upper() for m in metrics[:4]],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        for i, metric in enumerate(metrics[:4]):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            ascending = metric in ['rmse', 'mae', 'mape']
            sorted_data = metrics_df.sort_values(metric, ascending=ascending)
            
            fig.add_trace(
                go.Bar(
                    x=sorted_data.index,
                    y=sorted_data[metric],
                    name=metric.upper(),
                    showlegend=False,
                    text=[f'{v:.3f}' for v in sorted_data[metric]],
                    textposition='outside'
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title="Model Performance Comparison",
            height=800,
            showlegend=False
        )
        
        fig.write_html('results/visualizations/model_performance_comparison.html')
    
    def plot_prediction_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 model_name: str = "Model") -> None:
        """Plot predictions vs actual values"""
        self.logger.info(f"Creating prediction vs actual plot for {model_name}...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot
        ax1.scatter(y_true, y_pred, alpha=0.6, s=20)
        
        # Perfect prediction line
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax1.set_xlabel('Actual Credit Score')
        ax1.set_ylabel('Predicted Credit Score')
        ax1.set_title(f'{model_name} - Predictions vs Actual')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Residuals plot
        residuals = y_true - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.6, s=20)
        ax2.axhline(y=0, color='r', linestyle='--', lw=2)
        ax2.set_xlabel('Predicted Credit Score')
        ax2.set_ylabel('Residuals')
        ax2.set_title(f'{model_name} - Residuals Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'results/plots/{model_name.lower().replace(" ", "_")}_prediction_vs_actual.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create interactive version
        self._create_interactive_prediction_plot(y_true, y_pred, model_name)
    
    def _create_interactive_prediction_plot(self, y_true: np.ndarray, y_pred: np.ndarray,
                                          model_name: str) -> None:
        """Create interactive prediction vs actual plot"""
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Predictions vs Actual', 'Residuals Plot']
        )
        
        # Scatter plot
        fig.add_trace(
            go.Scatter(
                x=y_true,
                y=y_pred,
                mode='markers',
                name='Predictions',
                opacity=0.6,
                marker=dict(size=4)
            ),
            row=1, col=1
        )
        
        # Perfect prediction line
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=1
        )
        
        # Residuals plot
        residuals = y_true - y_pred
        fig.add_trace(
            go.Scatter(
                x=y_pred,
                y=residuals,
                mode='markers',
                name='Residuals',
                opacity=0.6,
                marker=dict(size=4)
            ),
            row=1, col=2
        )
        
        # Zero line for residuals
        fig.add_trace(
            go.Scatter(
                x=[np.min(y_pred), np.max(y_pred)],
                y=[0, 0],
                mode='lines',
                name='Zero Line',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Actual Credit Score", row=1, col=1)
        fig.update_yaxes(title_text="Predicted Credit Score", row=1, col=1)
        fig.update_xaxes(title_text="Predicted Credit Score", row=1, col=2)
        fig.update_yaxes(title_text="Residuals", row=1, col=2)
        
        fig.update_layout(
            title=f"{model_name} - Prediction Analysis",
            height=500,
            showlegend=True
        )
        
        fig.write_html(f'results/visualizations/{model_name.lower().replace(" ", "_")}_prediction_analysis.html')
    
    def plot_feature_importance(self, importance_dict: Dict[str, float],
                              model_name: str = "Model", top_n: int = 20) -> None:
        """Plot feature importance"""
        self.logger.info(f"Creating feature importance plot for {model_name}...")
        
        # Sort and select top features
        sorted_features = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n])
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        features = list(sorted_features.keys())
        importances = list(sorted_features.values())
        
        # Create horizontal bar plot
        bars = ax.barh(range(len(features)), importances)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'{model_name} - Top {top_n} Feature Importances')
        
        # Color bars
        colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Add value labels
        for i, (feature, importance) in enumerate(sorted_features.items()):
            ax.text(importance + max(importances) * 0.01, i,
                   f'{importance:.3f}', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'results/plots/{model_name.lower().replace(" ", "_")}_feature_importance.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create interactive version
        self._create_interactive_feature_importance_plot(sorted_features, model_name)
    
    def _create_interactive_feature_importance_plot(self, importance_dict: Dict[str, float],
                                                  model_name: str) -> None:
        """Create interactive feature importance plot"""
        
        features = list(importance_dict.keys())
        importances = list(importance_dict.values())
        
        fig = go.Figure(go.Bar(
            x=importances,
            y=features,
            orientation='h',
            text=[f'{imp:.3f}' for imp in importances],
            textposition='outside',
            marker=dict(
                color=importances,
                colorscale='viridis',
                showscale=True,
                colorbar=dict(title="Importance")
            )
        ))
        
        fig.update_layout(
            title=f"{model_name} - Feature Importance",
            xaxis_title="Feature Importance",
            yaxis_title="Features",
            height=max(400, len(features) * 25),
            showlegend=False
        )
        
        fig.write_html(f'results/visualizations/{model_name.lower().replace(" ", "_")}_feature_importance.html')
    
    def plot_shap_analysis(self, model: Any, X_sample: np.ndarray,
                          feature_names: List[str], model_name: str = "Model") -> None:
        """Create SHAP analysis plots"""
        self.logger.info(f"Creating SHAP analysis for {model_name}...")
        
        try:
            # Create SHAP explainer
            if hasattr(model, 'predict_proba'):
                explainer = shap.Explainer(model.predict_proba, X_sample[:100])
            else:
                explainer = shap.Explainer(model.predict, X_sample[:100])
            
            # Calculate SHAP values
            shap_values = explainer(X_sample[:500])  # Use subset for speed
            
            # Summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_sample[:500], feature_names=feature_names, show=False)
            plt.tight_layout()
            plt.savefig(f'results/plots/{model_name.lower().replace(" ", "_")}_shap_summary.png',
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Feature importance plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_sample[:500], feature_names=feature_names,
                            plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig(f'results/plots/{model_name.lower().replace(" ", "_")}_shap_importance.png',
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"SHAP analysis completed for {model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to create SHAP analysis for {model_name}: {str(e)}")
    
    def plot_credit_score_distribution(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     model_name: str = "Model") -> None:
        """Plot credit score distribution comparison"""
        self.logger.info(f"Creating credit score distribution plot for {model_name}...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Distribution comparison
        axes[0, 0].hist(y_true, bins=50, alpha=0.7, label='Actual', density=True)
        axes[0, 0].hist(y_pred, bins=50, alpha=0.7, label='Predicted', density=True)
        axes[0, 0].set_xlabel('Credit Score')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Credit Score Distribution Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Box plots
        data_to_plot = [y_true, y_pred]
        axes[0, 1].boxplot(data_to_plot, labels=['Actual', 'Predicted'])
        axes[0, 1].set_ylabel('Credit Score')
        axes[0, 1].set_title('Credit Score Box Plots')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Score bands comparison
        bands = [(300, 579), (580, 669), (670, 739), (740, 799), (800, 850)]
        band_names = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
        
        actual_counts = []
        pred_counts = []
        
        for low, high in bands:
            actual_count = np.sum((y_true >= low) & (y_true <= high))
            pred_count = np.sum((y_pred >= low) & (y_pred <= high))
            actual_counts.append(actual_count)
            pred_counts.append(pred_count)
        
        x = np.arange(len(band_names))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, actual_counts, width, label='Actual', alpha=0.8)
        axes[1, 0].bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.8)
        axes[1, 0].set_xlabel('Credit Score Bands')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Credit Score Band Distribution')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(band_names, rotation=45)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(y_true - y_pred, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot of Residuals')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'results/plots/{model_name.lower().replace(" ", "_")}_score_distribution.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_fairness_analysis(self, fairness_results: Dict[str, Dict[str, Dict[str, float]]],
                             model_name: str = "Model") -> None:
        """Plot fairness analysis results"""
        self.logger.info(f"Creating fairness analysis plots for {model_name}...")
        
        # Prepare data for plotting
        metrics_data = []
        
        for attr_name, attr_results in fairness_results.items():
            for metric_type, metrics in attr_results.items():
                if metric_type == 'demographic_parity':
                    metrics_data.append({
                        'attribute': attr_name,
                        'metric': 'Demographic Parity',
                        'value': metrics.get('demographic_parity_difference', 0)
                    })
                elif metric_type == 'equalized_odds':
                    metrics_data.append({
                        'attribute': attr_name,
                        'metric': 'Equalized Odds',
                        'value': metrics.get('equalized_odds_average_difference', 0)
                    })
                elif metric_type == 'equal_opportunity':
                    metrics_data.append({
                        'attribute': attr_name,
                        'metric': 'Equal Opportunity',
                        'value': metrics.get('equal_opportunity_difference', 0)
                    })
        
        if not metrics_data:
            self.logger.warning("No fairness data available for plotting")
            return
        
        df = pd.DataFrame(metrics_data)
        
        # Create fairness metrics plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Pivot for grouped bar plot
        pivot_df = df.pivot(index='attribute', columns='metric', values='value')
        pivot_df.plot(kind='bar', ax=ax, width=0.8)
        
        ax.set_xlabel('Protected Attributes')
        ax.set_ylabel('Fairness Violation (Lower is Better)')
        ax.set_title(f'{model_name} - Fairness Metrics by Protected Attribute')
        ax.legend(title='Fairness Metrics')
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line at 0.1 (common fairness threshold)
        ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Fairness Threshold (0.1)')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'results/plots/{model_name.lower().replace(" ", "_")}_fairness_analysis.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create interactive fairness plot
        self._create_interactive_fairness_plot(df, model_name)
    
    def _create_interactive_fairness_plot(self, df: pd.DataFrame, model_name: str) -> None:
        """Create interactive fairness analysis plot"""
        
        fig = px.bar(
            df,
            x='attribute',
            y='value',
            color='metric',
            title=f'{model_name} - Fairness Analysis',
            labels={'value': 'Fairness Violation', 'attribute': 'Protected Attributes'},
            barmode='group'
        )
        
        # Add fairness threshold line
        fig.add_hline(y=0.1, line_dash="dash", line_color="red",
                     annotation_text="Fairness Threshold (0.1)")
        
        fig.update_layout(height=600)
        fig.write_html(f'results/visualizations/{model_name.lower().replace(" ", "_")}_fairness_analysis.html')
    
    def create_model_dashboard(self, model_results: Dict[str, Any],
                             model_name: str = "Model") -> None:
        """Create comprehensive model dashboard"""
        self.logger.info(f"Creating model dashboard for {model_name}...")
        
        # This would create a comprehensive HTML dashboard
        # For now, we'll create a summary plot
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Performance Metrics', 'Feature Importance',
                          'Prediction Distribution', 'Fairness Summary'],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "histogram"}, {"type": "bar"}]]
        )
        
        # Add plots to dashboard (simplified version)
        # This would be expanded with actual data
        
        fig.update_layout(
            title=f"{model_name} - Comprehensive Dashboard",
            height=800,
            showlegend=True
        )
        
        fig.write_html(f'results/visualizations/{model_name.lower().replace(" ", "_")}_dashboard.html')
        
        self.logger.info(f"Dashboard created for {model_name}")
    
    def plot_learning_curves(self, train_scores: List[float], val_scores: List[float],
                           model_name: str = "Model") -> None:
        """Plot learning curves"""
        self.logger.info(f"Creating learning curves for {model_name}...")
        
        epochs = range(1, len(train_scores) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_scores, 'b-', label='Training Score', linewidth=2)
        plt.plot(epochs, val_scores, 'r-', label='Validation Score', linewidth=2)
        plt.xlabel('Epochs')
        plt.ylabel('Score')
        plt.title(f'{model_name} - Learning Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'results/plots/{model_name.lower().replace(" ", "_")}_learning_curves.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_all_plots_summary(self, models_results: Dict[str, Any]) -> None:
        """Save summary of all generated plots"""
        
        summary = {
            'total_models': len(models_results),
            'plots_generated': [],
            'visualizations_generated': []
        }
        
        # List all generated files
        plot_files = os.listdir('results/plots')
        viz_files = os.listdir('results/visualizations')
        
        summary['plots_generated'] = plot_files
        summary['visualizations_generated'] = viz_files
        
        # Save summary
        import json
        with open('results/visualization_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Generated {len(plot_files)} plots and {len(viz_files)} interactive visualizations")