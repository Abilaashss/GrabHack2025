#!/usr/bin/env python3
"""
Project Nova - Model Evaluation Script
Evaluate already trained models without retraining
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
from evaluation.metrics import ModelEvaluator

class ModelEvaluationRunner:
    """Run evaluation on already trained models"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.data_loader = DataLoader()
        self.evaluator = ModelEvaluator()
        
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
        """Load and prepare test data only"""
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
            
            # Store drivers test data
            self.drivers_data = {
                'X_test': X_test_d_scaled,
                'y_test': y_test_d,
                'feature_names': feature_names_d,
                'scaler': scaler_d
            }
            
            # Process merchants dataset
            merchants_prepared = self.data_loader.prepare_merchants_dataset(merchants_df)
            merchants_processed = self.data_loader.preprocess_features(merchants_prepared)
            X_m, y_m, feature_names_m = self.data_loader.prepare_ml_data(merchants_processed)
            X_train_m, X_val_m, X_test_m, y_train_m, y_val_m, y_test_m = self.data_loader.split_data(X_m, y_m)
            X_train_m_scaled, X_val_m_scaled, X_test_m_scaled, scaler_m = self.data_loader.scale_features(
                X_train_m, X_val_m, X_test_m
            )
            
            # Store merchants test data
            self.merchants_data = {
                'X_test': X_test_m_scaled,
                'y_test': y_test_m,
                'feature_names': feature_names_m,
                'scaler': scaler_m
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
                model_name = model_file.stem  # Remove file extension
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
                model_name = model_file.stem  # Remove file extension
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
    
    def evaluate_all_models(self):
        """Evaluate all loaded models"""
        self.logger.info("Evaluating all models...")
        
        try:
            # Evaluate drivers models
            if self.drivers_models:
                self.logger.info("Evaluating drivers models...")
                drivers_predictions = {}
                
                for model_name, model in self.drivers_models.items():
                    try:
                        y_pred = model.predict(self.drivers_data['X_test'])
                        drivers_predictions[f"drivers_{model_name}"] = y_pred
                    except Exception as e:
                        self.logger.error(f"Failed to predict with drivers {model_name}: {str(e)}")
                        continue
                
                if drivers_predictions:
                    drivers_evaluator = ModelEvaluator()
                    drivers_eval_df = drivers_evaluator.evaluate_all_models(
                        drivers_predictions, self.drivers_data['y_test']
                    )
                    
                    # Save drivers evaluation
                    drivers_eval_df.to_csv('results/metrics/drivers_models_evaluation.csv')
                    self.logger.info("Drivers evaluation saved to results/metrics/drivers_models_evaluation.csv")
                    
                    # Show best drivers model
                    best_drivers = drivers_eval_df.loc[drivers_eval_df['r2_score'].idxmax()]
                    self.logger.info(f"Best drivers model: {best_drivers.name} (R² = {best_drivers['r2_score']:.4f})")
            
            # Evaluate merchants models
            if self.merchants_models:
                self.logger.info("Evaluating merchants models...")
                merchants_predictions = {}
                
                for model_name, model in self.merchants_models.items():
                    try:
                        y_pred = model.predict(self.merchants_data['X_test'])
                        merchants_predictions[f"merchants_{model_name}"] = y_pred
                    except Exception as e:
                        self.logger.error(f"Failed to predict with merchants {model_name}: {str(e)}")
                        continue
                
                if merchants_predictions:
                    merchants_evaluator = ModelEvaluator()
                    merchants_eval_df = merchants_evaluator.evaluate_all_models(
                        merchants_predictions, self.merchants_data['y_test']
                    )
                    
                    # Save merchants evaluation
                    merchants_eval_df.to_csv('results/metrics/merchants_models_evaluation.csv')
                    self.logger.info("Merchants evaluation saved to results/metrics/merchants_models_evaluation.csv")
                    
                    # Show best merchants model
                    best_merchants = merchants_eval_df.loc[merchants_eval_df['r2_score'].idxmax()]
                    self.logger.info(f"Best merchants model: {best_merchants.name} (R² = {best_merchants['r2_score']:.4f})")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate models: {str(e)}")
            return False
    
    def run_evaluation(self):
        """Run the complete evaluation process"""
        self.logger.info("="*60)
        self.logger.info("PROJECT NOVA - MODEL EVALUATION")
        self.logger.info("="*60)
        
        # Step 1: Load test data
        if not self.load_test_data():
            return False
        
        # Step 2: Load trained models
        if not self.load_trained_models():
            return False
        
        # Step 3: Evaluate models
        if not self.evaluate_all_models():
            return False
        
        self.logger.info("="*60)
        self.logger.info("MODEL EVALUATION COMPLETED SUCCESSFULLY!")
        self.logger.info("="*60)
        self.logger.info("Results saved in:")
        self.logger.info("  - results/metrics/drivers_models_evaluation.csv")
        self.logger.info("  - results/metrics/merchants_models_evaluation.csv")
        self.logger.info("="*60)
        
        return True

def main():
    """Main function"""
    evaluator = ModelEvaluationRunner()
    success = evaluator.run_evaluation()
    
    if not success:
        print("\n❌ Evaluation failed. Check the logs above.")
        sys.exit(1)

if __name__ == "__main__":
    main()