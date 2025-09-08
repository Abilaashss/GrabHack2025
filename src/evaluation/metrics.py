"""
Project Nova - Evaluation Metrics Module
Comprehensive evaluation metrics for credit scoring models
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc,
    confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import yaml
import json
from typing import Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """Comprehensive model evaluation for credit scoring"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize ModelEvaluator with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.logger = self._setup_logger()
        self.evaluation_results = {}
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/model_evaluation.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def evaluate_regression_model(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 model_name: str = "model") -> Dict[str, float]:
        """Evaluate regression model performance"""
        self.logger.info(f"Evaluating regression model: {model_name}")
        
        metrics = {}
        
        # Basic regression metrics
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2_score'] = r2_score(y_true, y_pred)
        
        # Additional metrics
        metrics['mape'] = self._calculate_mape(y_true, y_pred)
        metrics['explained_variance'] = self._calculate_explained_variance(y_true, y_pred)
        
        # Residual analysis
        residuals = y_true - y_pred
        metrics['mean_residual'] = np.mean(residuals)
        metrics['std_residual'] = np.std(residuals)
        metrics['max_residual'] = np.max(np.abs(residuals))
        
        # Prediction intervals
        metrics['prediction_std'] = np.std(y_pred)
        metrics['prediction_range'] = np.max(y_pred) - np.min(y_pred)
        
        self.evaluation_results[model_name] = metrics
        
        return metrics
    
    def evaluate_classification_model(self, y_true: np.ndarray, y_pred: np.ndarray,
                                    y_pred_proba: np.ndarray = None,
                                    model_name: str = "model") -> Dict[str, float]:
        """Evaluate classification model performance"""
        self.logger.info(f"Evaluating classification model: {model_name}")
        
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # ROC AUC (if probabilities available)
        if y_pred_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:  # Binary classification
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:  # Multi-class
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
                
                # Precision-Recall AUC
                if len(np.unique(y_true)) == 2:
                    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
                    metrics['pr_auc'] = auc(recall, precision)
            except Exception as e:
                self.logger.warning(f"Could not calculate AUC metrics: {str(e)}")
        
        self.evaluation_results[model_name] = metrics
        
        return metrics
    
    def evaluate_credit_scoring_model(self, y_true: np.ndarray, y_pred: np.ndarray,
                                    model_name: str = "model") -> Dict[str, float]:
        """Evaluate credit scoring model with domain-specific metrics"""
        self.logger.info(f"Evaluating credit scoring model: {model_name}")
        
        # Start with regression metrics
        metrics = self.evaluate_regression_model(y_true, y_pred, model_name)
        
        # Credit scoring specific metrics
        
        # 1. Score distribution analysis
        metrics['pred_mean'] = np.mean(y_pred)
        metrics['pred_std'] = np.std(y_pred)
        metrics['pred_min'] = np.min(y_pred)
        metrics['pred_max'] = np.max(y_pred)
        
        metrics['true_mean'] = np.mean(y_true)
        metrics['true_std'] = np.std(y_true)
        metrics['true_min'] = np.min(y_true)
        metrics['true_max'] = np.max(y_true)
        
        # 2. Score band accuracy (for credit score ranges)
        metrics.update(self._calculate_score_band_accuracy(y_true, y_pred))
        
        # 3. Ranking metrics
        metrics.update(self._calculate_ranking_metrics(y_true, y_pred))
        
        # 4. Risk assessment metrics
        metrics.update(self._calculate_risk_metrics(y_true, y_pred))
        
        self.evaluation_results[model_name] = metrics
        
        return metrics
    
    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error"""
        return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    def _calculate_explained_variance(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate explained variance score"""
        return 1 - np.var(y_true - y_pred) / np.var(y_true)
    
    def _calculate_score_band_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate accuracy within credit score bands"""
        
        # Define credit score bands
        bands = {
            'poor': (300, 579),
            'fair': (580, 669),
            'good': (670, 739),
            'very_good': (740, 799),
            'excellent': (800, 850)
        }
        
        metrics = {}
        
        for band_name, (low, high) in bands.items():
            # True positives in this band
            true_in_band = (y_true >= low) & (y_true <= high)
            pred_in_band = (y_pred >= low) & (y_pred <= high)
            
            if np.sum(true_in_band) > 0:
                # Accuracy for this band
                correct_in_band = true_in_band & pred_in_band
                accuracy = np.sum(correct_in_band) / np.sum(true_in_band)
                metrics[f'{band_name}_band_accuracy'] = accuracy
            else:
                metrics[f'{band_name}_band_accuracy'] = 0.0
        
        # Overall band accuracy (within ±50 points)
        within_50_points = np.abs(y_true - y_pred) <= 50
        metrics['within_50_points_accuracy'] = np.mean(within_50_points)
        
        # Within ±25 points
        within_25_points = np.abs(y_true - y_pred) <= 25
        metrics['within_25_points_accuracy'] = np.mean(within_25_points)
        
        return metrics
    
    def _calculate_ranking_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate ranking-based metrics"""
        
        # Spearman correlation
        from scipy.stats import spearmanr, pearsonr
        
        spearman_corr, _ = spearmanr(y_true, y_pred)
        pearson_corr, _ = pearsonr(y_true, y_pred)
        
        # Kendall's tau
        from scipy.stats import kendalltau
        kendall_tau, _ = kendalltau(y_true, y_pred)
        
        return {
            'spearman_correlation': spearman_corr,
            'pearson_correlation': pearson_corr,
            'kendall_tau': kendall_tau
        }
    
    def _calculate_risk_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate risk assessment metrics"""
        
        # Define risk thresholds
        low_risk_threshold = 700
        high_risk_threshold = 600
        
        # True risk categories
        true_low_risk = y_true >= low_risk_threshold
        true_high_risk = y_true <= high_risk_threshold
        
        # Predicted risk categories
        pred_low_risk = y_pred >= low_risk_threshold
        pred_high_risk = y_pred <= high_risk_threshold
        
        metrics = {}
        
        # Low risk precision and recall
        if np.sum(pred_low_risk) > 0:
            metrics['low_risk_precision'] = np.sum(true_low_risk & pred_low_risk) / np.sum(pred_low_risk)
        else:
            metrics['low_risk_precision'] = 0.0
            
        if np.sum(true_low_risk) > 0:
            metrics['low_risk_recall'] = np.sum(true_low_risk & pred_low_risk) / np.sum(true_low_risk)
        else:
            metrics['low_risk_recall'] = 0.0
        
        # High risk precision and recall
        if np.sum(pred_high_risk) > 0:
            metrics['high_risk_precision'] = np.sum(true_high_risk & pred_high_risk) / np.sum(pred_high_risk)
        else:
            metrics['high_risk_precision'] = 0.0
            
        if np.sum(true_high_risk) > 0:
            metrics['high_risk_recall'] = np.sum(true_high_risk & pred_high_risk) / np.sum(true_high_risk)
        else:
            metrics['high_risk_recall'] = 0.0
        
        return metrics
    
    def cross_validate_model(self, model: Any, X: np.ndarray, y: np.ndarray,
                           cv_folds: int = 5, scoring: str = 'neg_mean_squared_error') -> Dict[str, float]:
        """Perform cross-validation evaluation"""
        self.logger.info(f"Performing {cv_folds}-fold cross-validation...")
        
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring, n_jobs=-1)
        
        metrics = {
            'cv_mean_score': np.mean(cv_scores),
            'cv_std_score': np.std(cv_scores),
            'cv_min_score': np.min(cv_scores),
            'cv_max_score': np.max(cv_scores)
        }
        
        return metrics
    
    def evaluate_all_models(self, models_predictions: Dict[str, np.ndarray],
                          y_true: np.ndarray) -> pd.DataFrame:
        """Evaluate all models and return comparison dataframe"""
        self.logger.info("Evaluating all models...")
        
        all_metrics = {}
        
        for model_name, y_pred in models_predictions.items():
            try:
                metrics = self.evaluate_credit_scoring_model(y_true, y_pred, model_name)
                all_metrics[model_name] = metrics
            except Exception as e:
                self.logger.error(f"Failed to evaluate {model_name}: {str(e)}")
                continue
        
        # Convert to DataFrame for easy comparison
        metrics_df = pd.DataFrame(all_metrics).T
        
        # Sort by R² score (descending)
        if 'r2_score' in metrics_df.columns:
            metrics_df = metrics_df.sort_values('r2_score', ascending=False)
        
        # Save results
        metrics_df.to_csv('results/metrics/all_models_evaluation.csv')
        
        # Convert numpy types to native Python types for JSON serialization
        json_compatible_metrics = self._convert_numpy_types(all_metrics)
        
        with open('results/metrics/all_models_evaluation.json', 'w') as f:
            json.dump(json_compatible_metrics, f, indent=2)
        
        return metrics_df
    
    def create_evaluation_report(self, model_name: str, y_true: np.ndarray,
                               y_pred: np.ndarray) -> str:
        """Create a comprehensive evaluation report"""
        
        metrics = self.evaluate_credit_scoring_model(y_true, y_pred, model_name)
        
        report = f"""
# Credit Scoring Model Evaluation Report
## Model: {model_name}

### Regression Metrics
- **R² Score**: {metrics['r2_score']:.4f}
- **RMSE**: {metrics['rmse']:.2f}
- **MAE**: {metrics['mae']:.2f}
- **MAPE**: {metrics['mape']:.2f}%

### Score Distribution
- **Predicted Mean**: {metrics['pred_mean']:.2f}
- **Predicted Std**: {metrics['pred_std']:.2f}
- **True Mean**: {metrics['true_mean']:.2f}
- **True Std**: {metrics['true_std']:.2f}

### Accuracy Metrics
- **Within ±25 points**: {metrics['within_25_points_accuracy']:.2%}
- **Within ±50 points**: {metrics['within_50_points_accuracy']:.2%}

### Correlation Metrics
- **Pearson Correlation**: {metrics['pearson_correlation']:.4f}
- **Spearman Correlation**: {metrics['spearman_correlation']:.4f}
- **Kendall Tau**: {metrics['kendall_tau']:.4f}

### Risk Assessment
- **Low Risk Precision**: {metrics['low_risk_precision']:.4f}
- **Low Risk Recall**: {metrics['low_risk_recall']:.4f}
- **High Risk Precision**: {metrics['high_risk_precision']:.4f}
- **High Risk Recall**: {metrics['high_risk_recall']:.4f}

### Score Band Accuracy
- **Poor Band (300-579)**: {metrics.get('poor_band_accuracy', 0):.2%}
- **Fair Band (580-669)**: {metrics.get('fair_band_accuracy', 0):.2%}
- **Good Band (670-739)**: {metrics.get('good_band_accuracy', 0):.2%}
- **Very Good Band (740-799)**: {metrics.get('very_good_band_accuracy', 0):.2%}
- **Excellent Band (800-850)**: {metrics.get('excellent_band_accuracy', 0):.2%}
        """
        
        # Save report
        with open(f'results/metrics/{model_name}_evaluation_report.md', 'w') as f:
            f.write(report)
        
        return report
    
    def get_model_ranking(self, metrics_df: pd.DataFrame, 
                         ranking_metric: str = 'r2_score') -> pd.DataFrame:
        """Get model ranking based on specified metric"""
        
        if ranking_metric not in metrics_df.columns:
            raise ValueError(f"Metric {ranking_metric} not found in evaluation results")
        
        # Sort by metric (ascending for error metrics, descending for score metrics)
        ascending = ranking_metric in ['mse', 'rmse', 'mae', 'mape']
        ranked_df = metrics_df.sort_values(ranking_metric, ascending=ascending)
        
        # Add rank column
        ranked_df['rank'] = range(1, len(ranked_df) + 1)
        
        return ranked_df
    
    def save_evaluation_results(self) -> None:
        """Save all evaluation results"""
        
        # Convert numpy types for JSON serialization
        json_compatible_results = self._convert_numpy_types(self.evaluation_results)
        
        # Save individual model results
        for model_name, metrics in json_compatible_results.items():
            with open(f'results/metrics/{model_name}_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
        
        # Save combined results
        with open('results/metrics/all_evaluation_results.json', 'w') as f:
            json.dump(json_compatible_results, f, indent=2)
        
        self.logger.info("Evaluation results saved")
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        import numpy as np
        
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def load_evaluation_results(self, model_name: str = None) -> Dict[str, Any]:
        """Load saved evaluation results"""
        
        if model_name:
            try:
                with open(f'results/metrics/{model_name}_metrics.json', 'r') as f:
                    return json.load(f)
            except FileNotFoundError:
                self.logger.warning(f"Evaluation results not found for {model_name}")
                return {}
        else:
            try:
                with open('results/metrics/all_evaluation_results.json', 'r') as f:
                    return json.load(f)
            except FileNotFoundError:
                self.logger.warning("Combined evaluation results not found")
                return {}