#!/usr/bin/env python3
"""
Project Nova - Visualization Script
Create visualizations for already trained models
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append('src')

from preprocessing.data_loader import DataLoader
from models.ml_models import MLModels
from visualization.plots import ModelVisualizer

class VisualizationRunner:
    """Create visualizations for already trained models"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.data_loader = DataLoader()
        self.visualizer = ModelVisualizer()
        
        # Data storage
        self.drivers_data = {}
        self.merchants_data = {}
        
        # Model storage
        self.drivers_models = {}
        self.merchants_models = {}
        
    def _setup_logger(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def load_test_data(self):
        """Load and prepare test data"""
        self.logger.info("Loading test data...")
        
        try:
            # Load datasets
            drivers_df, merchants_df = self.data_loader.load_data()
            
            # Process drivers dataset
            drivers_prepared = self.data_loader.prepare_drivers_dataset(drivers_df)
            drivers_processed = self.data_loader.preprocess_features(drivers_prepared)
            X_d, y_d, feature_names_d = self.data_loader.prepare_ml_data(drivers_processed)
            X_train_d, X_val_d, X_test_d, y_train_d, y_val_d, y_test_d = self.data_loader.split_data(X_d, y_d)
            X_train_d_scaled, X_val_d_scaled, X_test_d_scaled, scaler_d = self.data_loader.scale_features(
                X_train_d, X_val_d, X_test_d
            )
            
            # Store drivers data
            self.drivers_data = {
                'X_test': X_test_d_scaled,
                'y_test': y_test_d,
                'feature_names': feature_names_d
            }
            
            # Process merchants dataset
            merchants_prepared = self.data_loader.prepare_merchants_dataset(merchants_df)
            merchants_processed = self.data_loader.preprocess_features(merchants_prepared)
            X_m, y_m, feature_names_m = self.data_loader.prepare_ml_data(merchants_processed)
            X_train_m, X_val_m, X_test_m, y_train_m, y_val_m, y_test_m = self.data_loader.split_data(X_m, y_m)
            X_train_m_scaled, X_val_m_scaled, X_test_m_scaled, scaler_m = self.data_loader.scale_features(
                X_train_m, X_val_m, X_test_m
            )
            
            # Store merchants data
            self.merchants_data = {
                'X_test': X_test_m_scaled,
                'y_test': y_test_m,
                'feature_names': feature_names_m
            }
            
            self.logger.info(f"Test data loaded - Drivers: {X_test_d.shape}, Merchants: {X_test_m.shape}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load test data: {str(e)}")
            return False
    
    def load_trained_models(self):
        """Load already trained models from disk"""
        self.logger.info("Loading trained models...")
        
        models_dir = Path("models/saved_models")
        if not models_dir.exists():
            self.logger.error("Models directory not found. Please train models first.")
            return False
        
        model_files = list(models_dir.glob("*"))
        if not model_files:
            self.logger.error("No trained models found. Please train models first.")
            return False
        
        ml_models = MLModels()
        
        # Load drivers models
        drivers_count = 0
        for model_file in model_files:
            if model_file.name.startswith("drivers_"):
                model_name = model_file.stem
                base_name = model_name.replace("drivers_", "")
                
                try:
                    model = ml_models.load_model(model_name)
                    self.drivers_models[base_name] = model
                    drivers_count += 1
                    self.logger.info(f"Loaded drivers model: {base_name}")
                except Exception as e:
                    self.logger.warning(f"Failed to load {model_name}: {str(e)}")
        
        # Load merchants models
        merchants_count = 0
        for model_file in model_files:
            if model_file.name.startswith("merchants_"):
                model_name = model_file.stem
                base_name = model_name.replace("merchants_", "")
                
                try:
                    model = ml_models.load_model(model_name)
                    self.merchants_models[base_name] = model
                    merchants_count += 1
                    self.logger.info(f"Loaded merchants model: {base_name}")
                except Exception as e:
                    self.logger.warning(f"Failed to load {model_name}: {str(e)}")
        
        self.logger.info(f"Loaded {drivers_count} drivers models and {merchants_count} merchants models")
        return drivers_count > 0 or merchants_count > 0
    
    def load_evaluation_results(self):
        """Load evaluation results if available"""
        drivers_eval = None
        merchants_eval = None
        
        try:
            if os.path.exists('results/metrics/drivers_models_evaluation.csv'):
                drivers_eval = pd.read_csv('results/metrics/drivers_models_evaluation.csv', index_col=0)
                self.logger.info("Loaded drivers evaluation results")
        except Exception as e:
            self.logger.warning(f"Could not load drivers evaluation: {str(e)}")
        
        try:
            if os.path.exists('results/metrics/merchants_models_evaluation.csv'):
                merchants_eval = pd.read_csv('results/metrics/merchants_models_evaluation.csv', index_col=0)
                self.logger.info("Loaded merchants evaluation results")
        except Exception as e:
            self.logger.warning(f"Could not load merchants evaluation: {str(e)}")
        
        return drivers_eval, merchants_eval
    
    def create_visualizations(self):
        """Create visualizations for all loaded models"""
        self.logger.info("Creating visualizations...")
        
        try:
            # Load evaluation results
            drivers_eval, merchants_eval = self.load_evaluation_results()
            
            # Create performance comparison plots
            if drivers_eval is not None:
                self.logger.info("Creating drivers performance comparison...")
                self.visualizer.plot_model_performance_comparison(drivers_eval)
            
            if merchants_eval is not None:
                self.logger.info("Creating merchants performance comparison...")
                self.visualizer.plot_model_performance_comparison(merchants_eval)
            
            # Create individual model visualizations for drivers
            if self.drivers_models:
                self.logger.info("Creating visualizations for drivers models...")
                
                for model_name, model in self.drivers_models.items():
                    try:
                        y_pred = model.predict(self.drivers_data['X_test'])
                        
                        # Prediction vs actual plots
                        self.visualizer.plot_prediction_vs_actual(
                            self.drivers_data['y_test'], y_pred, f"drivers_{model_name}"
                        )
                        
                        # Credit score distribution
                        self.visualizer.plot_credit_score_distribution(
                            self.drivers_data['y_test'], y_pred, f"drivers_{model_name}"
                        )
                        
                        # Feature importance (if available)
                        try:
                            if hasattr(model, 'feature_importances_'):
                                importance_dict = dict(zip(
                                    self.drivers_data['feature_names'], 
                                    model.feature_importances_
                                ))
                                self.visualizer.plot_feature_importance(
                                    importance_dict, f"drivers_{model_name}"
                                )
                        except Exception as e:
                            self.logger.warning(f"Could not create feature importance for drivers {model_name}: {str(e)}")
                        
                        self.logger.info(f"Created visualizations for drivers {model_name}")
                        
                    except Exception as e:
                        self.logger.error(f"Failed to create visualizations for drivers {model_name}: {str(e)}")
                        continue
            
            # Create individual model visualizations for merchants
            if self.merchants_models:
                self.logger.info("Creating visualizations for merchants models...")
                
                for model_name, model in self.merchants_models.items():
                    try:
                        y_pred = model.predict(self.merchants_data['X_test'])
                        
                        # Prediction vs actual plots
                        self.visualizer.plot_prediction_vs_actual(
                            self.merchants_data['y_test'], y_pred, f"merchants_{model_name}"
                        )
                        
                        # Credit score distribution
                        self.visualizer.plot_credit_score_distribution(
                            self.merchants_data['y_test'], y_pred, f"merchants_{model_name}"
                        )
                        
                        # Feature importance (if available)
                        try:
                            if hasattr(model, 'feature_importances_'):
                                importance_dict = dict(zip(
                                    self.merchants_data['feature_names'], 
                                    model.feature_importances_
                                ))
                                self.visualizer.plot_feature_importance(
                                    importance_dict, f"merchants_{model_name}"
                                )
                        except Exception as e:
                            self.logger.warning(f"Could not create feature importance for merchants {model_name}: {str(e)}")
                        
                        self.logger.info(f"Created visualizations for merchants {model_name}")
                        
                    except Exception as e:
                        self.logger.error(f"Failed to create visualizations for merchants {model_name}: {str(e)}")
                        continue
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create visualizations: {str(e)}")
            return False
    
    def run_visualization(self):
        """Run the complete visualization process"""
        self.logger.info("="*60)
        self.logger.info("PROJECT NOVA - VISUALIZATION CREATION")
        self.logger.info("="*60)
        
        # Step 1: Load test data
        if not self.load_test_data():
            return False
        
        # Step 2: Load trained models
        if not self.load_trained_models():
            return False
        
        # Step 3: Create visualizations
        if not self.create_visualizations():
            return False
        
        self.logger.info("="*60)
        self.logger.info("VISUALIZATION CREATION COMPLETED SUCCESSFULLY!")
        self.logger.info("="*60)
        self.logger.info("Visualizations saved in:")
        self.logger.info("  - results/plots/ (static images)")
        self.logger.info("  - results/visualizations/ (interactive HTML)")
        self.logger.info("="*60)
        
        return True

def main():
    """Main function"""
    visualizer = VisualizationRunner()
    success = visualizer.run_visualization()
    
    if not success:
        print("\n❌ Visualization creation failed. Check the logs above.")
        sys.exit(1)

if __name__ == "__main__":
    main()