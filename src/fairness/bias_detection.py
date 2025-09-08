"""
Project Nova - Bias Detection and Fairness Analysis Module
Implements comprehensive fairness metrics and bias detection
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import yaml
import json
from typing import Dict, Any, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class BiasDetector:
    """Bias detection and fairness analysis for credit scoring models"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize BiasDetector with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.logger = self._setup_logger()
        self.fairness_results = {}
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/bias_detection.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def calculate_demographic_parity(self, y_pred: np.ndarray, 
                                   protected_attr: np.ndarray,
                                   threshold: float = 700) -> Dict[str, float]:
        """Calculate demographic parity metrics"""
        self.logger.info("Calculating demographic parity...")
        
        # Convert predictions to binary (high/low credit score)
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        # Get unique groups
        groups = np.unique(protected_attr)
        
        # Calculate positive rate for each group
        group_positive_rates = {}
        for group in groups:
            group_mask = protected_attr == group
            if np.sum(group_mask) > 0:
                positive_rate = np.mean(y_pred_binary[group_mask])
                group_positive_rates[f'group_{group}'] = positive_rate
        
        # Calculate demographic parity difference
        rates = list(group_positive_rates.values())
        dp_difference = max(rates) - min(rates) if rates else 0
        
        # Calculate demographic parity ratio
        dp_ratio = min(rates) / max(rates) if rates and max(rates) > 0 else 0
        
        metrics = {
            'demographic_parity_difference': dp_difference,
            'demographic_parity_ratio': dp_ratio,
            **group_positive_rates
        }
        
        return metrics
    
    def calculate_equalized_odds(self, y_true: np.ndarray, y_pred: np.ndarray,
                               protected_attr: np.ndarray,
                               threshold: float = 700) -> Dict[str, float]:
        """Calculate equalized odds metrics"""
        self.logger.info("Calculating equalized odds...")
        
        # Convert to binary
        y_true_binary = (y_true >= threshold).astype(int)
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        groups = np.unique(protected_attr)
        
        # Calculate TPR and FPR for each group
        group_tpr = {}
        group_fpr = {}
        
        for group in groups:
            group_mask = protected_attr == group
            
            if np.sum(group_mask) > 0:
                y_true_group = y_true_binary[group_mask]
                y_pred_group = y_pred_binary[group_mask]
                
                # True Positive Rate
                if np.sum(y_true_group == 1) > 0:
                    tpr = np.sum((y_true_group == 1) & (y_pred_group == 1)) / np.sum(y_true_group == 1)
                else:
                    tpr = 0
                
                # False Positive Rate
                if np.sum(y_true_group == 0) > 0:
                    fpr = np.sum((y_true_group == 0) & (y_pred_group == 1)) / np.sum(y_true_group == 0)
                else:
                    fpr = 0
                
                group_tpr[f'group_{group}_tpr'] = tpr
                group_fpr[f'group_{group}_fpr'] = fpr
        
        # Calculate equalized odds differences
        tpr_values = [v for k, v in group_tpr.items()]
        fpr_values = [v for k, v in group_fpr.items()]
        
        tpr_difference = max(tpr_values) - min(tpr_values) if tpr_values else 0
        fpr_difference = max(fpr_values) - min(fpr_values) if fpr_values else 0
        
        metrics = {
            'equalized_odds_tpr_difference': tpr_difference,
            'equalized_odds_fpr_difference': fpr_difference,
            'equalized_odds_average_difference': (tpr_difference + fpr_difference) / 2,
            **group_tpr,
            **group_fpr
        }
        
        return metrics
    
    def calculate_equal_opportunity(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  protected_attr: np.ndarray,
                                  threshold: float = 700) -> Dict[str, float]:
        """Calculate equal opportunity metrics"""
        self.logger.info("Calculating equal opportunity...")
        
        # Convert to binary
        y_true_binary = (y_true >= threshold).astype(int)
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        groups = np.unique(protected_attr)
        
        # Calculate TPR for each group (only for positive class)
        group_tpr = {}
        
        for group in groups:
            group_mask = protected_attr == group
            
            if np.sum(group_mask) > 0:
                y_true_group = y_true_binary[group_mask]
                y_pred_group = y_pred_binary[group_mask]
                
                # True Positive Rate for positive class only
                if np.sum(y_true_group == 1) > 0:
                    tpr = np.sum((y_true_group == 1) & (y_pred_group == 1)) / np.sum(y_true_group == 1)
                else:
                    tpr = 0
                
                group_tpr[f'group_{group}_tpr'] = tpr
        
        # Calculate equal opportunity difference
        tpr_values = list(group_tpr.values())
        eo_difference = max(tpr_values) - min(tpr_values) if tpr_values else 0
        
        metrics = {
            'equal_opportunity_difference': eo_difference,
            **group_tpr
        }
        
        return metrics
    
    def calculate_individual_fairness(self, X: np.ndarray, y_pred: np.ndarray,
                                    protected_attr: np.ndarray,
                                    k: int = 5) -> Dict[str, float]:
        """Calculate individual fairness metrics using k-nearest neighbors"""
        self.logger.info("Calculating individual fairness...")
        
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Find k-nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(X_scaled)
        distances, indices = nbrs.kneighbors(X_scaled)
        
        # Calculate individual fairness violations
        violations = []
        
        for i in range(len(X)):
            # Get neighbors (excluding self)
            neighbor_indices = indices[i][1:]
            
            # Check if any neighbors have different protected attribute
            different_group_neighbors = []
            for neighbor_idx in neighbor_indices:
                if protected_attr[neighbor_idx] != protected_attr[i]:
                    different_group_neighbors.append(neighbor_idx)
            
            if different_group_neighbors:
                # Calculate prediction difference with different-group neighbors
                pred_diffs = []
                for neighbor_idx in different_group_neighbors:
                    pred_diff = abs(y_pred[i] - y_pred[neighbor_idx])
                    pred_diffs.append(pred_diff)
                
                # Average prediction difference
                avg_pred_diff = np.mean(pred_diffs)
                violations.append(avg_pred_diff)
        
        # Individual fairness metrics
        if violations:
            individual_fairness_violation = np.mean(violations)
            individual_fairness_std = np.std(violations)
            max_violation = np.max(violations)
        else:
            individual_fairness_violation = 0
            individual_fairness_std = 0
            max_violation = 0
        
        metrics = {
            'individual_fairness_violation': individual_fairness_violation,
            'individual_fairness_std': individual_fairness_std,
            'individual_fairness_max_violation': max_violation
        }
        
        return metrics
    
    def calculate_calibration_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                    protected_attr: np.ndarray,
                                    n_bins: int = 10) -> Dict[str, float]:
        """Calculate calibration metrics across groups"""
        self.logger.info("Calculating calibration metrics...")
        
        groups = np.unique(protected_attr)
        
        # Create prediction bins
        pred_bins = np.linspace(np.min(y_pred), np.max(y_pred), n_bins + 1)
        
        group_calibration = {}
        
        for group in groups:
            group_mask = protected_attr == group
            
            if np.sum(group_mask) > 0:
                y_true_group = y_true[group_mask]
                y_pred_group = y_pred[group_mask]
                
                # Calculate calibration error
                calibration_errors = []
                
                for i in range(n_bins):
                    bin_mask = (y_pred_group >= pred_bins[i]) & (y_pred_group < pred_bins[i+1])
                    
                    if np.sum(bin_mask) > 0:
                        bin_true_mean = np.mean(y_true_group[bin_mask])
                        bin_pred_mean = np.mean(y_pred_group[bin_mask])
                        calibration_error = abs(bin_true_mean - bin_pred_mean)
                        calibration_errors.append(calibration_error)
                
                if calibration_errors:
                    group_calibration[f'group_{group}_calibration_error'] = np.mean(calibration_errors)
                else:
                    group_calibration[f'group_{group}_calibration_error'] = 0
        
        # Calculate calibration difference between groups
        calibration_values = list(group_calibration.values())
        calibration_difference = max(calibration_values) - min(calibration_values) if calibration_values else 0
        
        metrics = {
            'calibration_difference': calibration_difference,
            **group_calibration
        }
        
        return metrics
    
    def comprehensive_fairness_analysis(self, y_true: np.ndarray, y_pred: np.ndarray,
                                      X: np.ndarray, protected_attributes: Dict[str, np.ndarray],
                                      model_name: str = "model") -> Dict[str, Dict[str, float]]:
        """Perform comprehensive fairness analysis"""
        self.logger.info(f"Performing comprehensive fairness analysis for {model_name}...")
        
        fairness_results = {}
        
        for attr_name, attr_values in protected_attributes.items():
            self.logger.info(f"Analyzing fairness for attribute: {attr_name}")
            
            attr_results = {}
            
            # Demographic Parity
            try:
                dp_metrics = self.calculate_demographic_parity(y_pred, attr_values)
                attr_results['demographic_parity'] = dp_metrics
            except Exception as e:
                self.logger.error(f"Error calculating demographic parity for {attr_name}: {str(e)}")
            
            # Equalized Odds
            try:
                eo_metrics = self.calculate_equalized_odds(y_true, y_pred, attr_values)
                attr_results['equalized_odds'] = eo_metrics
            except Exception as e:
                self.logger.error(f"Error calculating equalized odds for {attr_name}: {str(e)}")
            
            # Equal Opportunity
            try:
                eop_metrics = self.calculate_equal_opportunity(y_true, y_pred, attr_values)
                attr_results['equal_opportunity'] = eop_metrics
            except Exception as e:
                self.logger.error(f"Error calculating equal opportunity for {attr_name}: {str(e)}")
            
            # Individual Fairness
            try:
                if_metrics = self.calculate_individual_fairness(X, y_pred, attr_values)
                attr_results['individual_fairness'] = if_metrics
            except Exception as e:
                self.logger.error(f"Error calculating individual fairness for {attr_name}: {str(e)}")
            
            # Calibration
            try:
                cal_metrics = self.calculate_calibration_metrics(y_true, y_pred, attr_values)
                attr_results['calibration'] = cal_metrics
            except Exception as e:
                self.logger.error(f"Error calculating calibration for {attr_name}: {str(e)}")
            
            fairness_results[attr_name] = attr_results
        
        # Store results
        self.fairness_results[model_name] = fairness_results
        
        # Save results
        self._save_fairness_results(model_name, fairness_results)
        
        return fairness_results
    
    def calculate_fairness_score(self, fairness_results: Dict[str, Dict[str, Dict[str, float]]]) -> float:
        """Calculate overall fairness score (0-1, higher is more fair)"""
        
        all_violations = []
        
        for attr_name, attr_results in fairness_results.items():
            # Demographic parity violations
            if 'demographic_parity' in attr_results:
                dp_diff = attr_results['demographic_parity'].get('demographic_parity_difference', 0)
                all_violations.append(dp_diff)
            
            # Equalized odds violations
            if 'equalized_odds' in attr_results:
                eo_diff = attr_results['equalized_odds'].get('equalized_odds_average_difference', 0)
                all_violations.append(eo_diff)
            
            # Equal opportunity violations
            if 'equal_opportunity' in attr_results:
                eop_diff = attr_results['equal_opportunity'].get('equal_opportunity_difference', 0)
                all_violations.append(eop_diff)
            
            # Individual fairness violations (normalized)
            if 'individual_fairness' in attr_results:
                if_violation = attr_results['individual_fairness'].get('individual_fairness_violation', 0)
                # Normalize by typical credit score range (550)
                if_violation_normalized = if_violation / 550
                all_violations.append(if_violation_normalized)
        
        if all_violations:
            # Fairness score: 1 - average violation (capped at 0)
            avg_violation = np.mean(all_violations)
            fairness_score = max(0, 1 - avg_violation)
        else:
            fairness_score = 1.0
        
        return fairness_score
    
    def create_fairness_report(self, model_name: str, 
                             fairness_results: Dict[str, Dict[str, Dict[str, float]]]) -> str:
        """Create comprehensive fairness report"""
        
        fairness_score = self.calculate_fairness_score(fairness_results)
        
        report = f"""
# Fairness Analysis Report
## Model: {model_name}
## Overall Fairness Score: {fairness_score:.3f}

"""
        
        for attr_name, attr_results in fairness_results.items():
            report += f"### Protected Attribute: {attr_name.upper()}\n\n"
            
            # Demographic Parity
            if 'demographic_parity' in attr_results:
                dp = attr_results['demographic_parity']
                report += f"**Demographic Parity:**\n"
                report += f"- Difference: {dp.get('demographic_parity_difference', 0):.4f}\n"
                report += f"- Ratio: {dp.get('demographic_parity_ratio', 0):.4f}\n\n"
            
            # Equalized Odds
            if 'equalized_odds' in attr_results:
                eo = attr_results['equalized_odds']
                report += f"**Equalized Odds:**\n"
                report += f"- TPR Difference: {eo.get('equalized_odds_tpr_difference', 0):.4f}\n"
                report += f"- FPR Difference: {eo.get('equalized_odds_fpr_difference', 0):.4f}\n"
                report += f"- Average Difference: {eo.get('equalized_odds_average_difference', 0):.4f}\n\n"
            
            # Equal Opportunity
            if 'equal_opportunity' in attr_results:
                eop = attr_results['equal_opportunity']
                report += f"**Equal Opportunity:**\n"
                report += f"- Difference: {eop.get('equal_opportunity_difference', 0):.4f}\n\n"
            
            # Individual Fairness
            if 'individual_fairness' in attr_results:
                if_metrics = attr_results['individual_fairness']
                report += f"**Individual Fairness:**\n"
                report += f"- Average Violation: {if_metrics.get('individual_fairness_violation', 0):.2f}\n"
                report += f"- Max Violation: {if_metrics.get('individual_fairness_max_violation', 0):.2f}\n\n"
            
            # Calibration
            if 'calibration' in attr_results:
                cal = attr_results['calibration']
                report += f"**Calibration:**\n"
                report += f"- Difference: {cal.get('calibration_difference', 0):.4f}\n\n"
        
        # Fairness interpretation
        report += "## Fairness Interpretation\n\n"
        
        if fairness_score >= 0.9:
            report += "✅ **Excellent fairness** - The model shows minimal bias across protected groups.\n"
        elif fairness_score >= 0.8:
            report += "✅ **Good fairness** - The model shows acceptable levels of bias.\n"
        elif fairness_score >= 0.7:
            report += "⚠️ **Moderate fairness** - The model shows some bias that should be addressed.\n"
        elif fairness_score >= 0.6:
            report += "⚠️ **Poor fairness** - The model shows significant bias requiring mitigation.\n"
        else:
            report += "❌ **Very poor fairness** - The model shows severe bias and should not be deployed.\n"
        
        # Save report
        with open(f'results/fairness_analysis/{model_name}_fairness_report.md', 'w') as f:
            f.write(report)
        
        return report
    
    def _save_fairness_results(self, model_name: str, 
                              fairness_results: Dict[str, Dict[str, Dict[str, float]]]) -> None:
        """Save fairness analysis results"""
        
        # Save detailed results
        with open(f'results/fairness_analysis/{model_name}_fairness_results.json', 'w') as f:
            json.dump(fairness_results, f, indent=2)
        
        # Create summary
        summary = {}
        for attr_name, attr_results in fairness_results.items():
            attr_summary = {}
            
            for metric_type, metrics in attr_results.items():
                if metric_type == 'demographic_parity':
                    attr_summary['dp_difference'] = metrics.get('demographic_parity_difference', 0)
                elif metric_type == 'equalized_odds':
                    attr_summary['eo_difference'] = metrics.get('equalized_odds_average_difference', 0)
                elif metric_type == 'equal_opportunity':
                    attr_summary['eop_difference'] = metrics.get('equal_opportunity_difference', 0)
                elif metric_type == 'individual_fairness':
                    attr_summary['if_violation'] = metrics.get('individual_fairness_violation', 0)
                elif metric_type == 'calibration':
                    attr_summary['cal_difference'] = metrics.get('calibration_difference', 0)
            
            summary[attr_name] = attr_summary
        
        # Calculate overall fairness score
        summary['overall_fairness_score'] = self.calculate_fairness_score(fairness_results)
        
        # Save summary
        with open(f'results/fairness_analysis/{model_name}_fairness_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Fairness results saved for {model_name}")
    
    def compare_model_fairness(self, models_fairness: Dict[str, Dict]) -> pd.DataFrame:
        """Compare fairness across multiple models"""
        
        comparison_data = []
        
        for model_name, fairness_results in models_fairness.items():
            fairness_score = self.calculate_fairness_score(fairness_results)
            
            row = {'model': model_name, 'overall_fairness_score': fairness_score}
            
            # Add attribute-specific scores
            for attr_name, attr_results in fairness_results.items():
                for metric_type, metrics in attr_results.items():
                    if metric_type == 'demographic_parity':
                        row[f'{attr_name}_dp_diff'] = metrics.get('demographic_parity_difference', 0)
                    elif metric_type == 'equalized_odds':
                        row[f'{attr_name}_eo_diff'] = metrics.get('equalized_odds_average_difference', 0)
                    elif metric_type == 'equal_opportunity':
                        row[f'{attr_name}_eop_diff'] = metrics.get('equal_opportunity_difference', 0)
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('overall_fairness_score', ascending=False)
        
        # Save comparison
        comparison_df.to_csv('results/fairness_analysis/model_fairness_comparison.csv', index=False)
        
        return comparison_df