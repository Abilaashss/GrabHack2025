#!/usr/bin/env python3
"""
Project Nova - Fairness Analysis Script
Analyze fairness of already trained models
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
from fairness.bias_detection import BiasDetector

class FairnessAnalysisRunner:
    """Run fairness analysis on already trained models"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.data_loader = DataLoader()
        self.bias_detector = BiasDetector()
        
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
        """Load and prepare test data with protected attributes"""
        self.logger.info("Loading test data with protected attributes...")
        
        try:
            # Load datasets
            drivers_df, merchants_df = self.data_loader.load_data()
            
            # Process drivers dataset
            drivers_prepared = self.data_loader.prepare_drivers_dataset(drivers_df)
            # Extract protected attributes BEFORE preprocessing (before one-hot encoding)
            protected_attrs_d = self.data_loader.get_protected_attributes_before_encoding(drivers_prepared)
            drivers_processed = self.data_loader.preprocess_features(drivers_prepared)
            X_d, y_d, feature_names_d = self.data_loader.prepare_ml_data(drivers_processed)
            X_train_d, X_val_d, X_test_d, y_train_d, y_val_d, y_test_d = self.data_loader.split_data(X_d, y_d)
            X_train_d_scaled, X_val_d_scaled, X_test_d_scaled, scaler_d = self.data_loader.scale_features(
                X_train_d, X_val_d, X_test_d
            )
            
            # Extract test set protected attributes for drivers
            test_size_d = len(y_test_d)
            total_size_d = len(y_train_d) + len(y_val_d) + len(y_test_d)
            test_start_idx_d = len(y_train_d) + len(y_val_d)
            
            test_protected_attrs_d = {}
            for attr_name, attr_values in protected_attrs_d.items():
                if len(attr_values) == total_size_d:
                    test_protected_attrs_d[attr_name] = attr_values[test_start_idx_d:]
            
            # Store drivers data
            self.drivers_data = {
                'X_test': X_test_d_scaled,
                'y_test': y_test_d,
                'feature_names': feature_names_d,
                'protected_attributes': test_protected_attrs_d
            }
            
            # Process merchants dataset
            merchants_prepared = self.data_loader.prepare_merchants_dataset(merchants_df)
            merchants_processed = self.data_loader.preprocess_features(merchants_prepared)
            protected_attrs_m = self.data_loader.get_protected_attributes(merchants_processed)
            X_m, y_m, feature_names_m = self.data_loader.prepare_ml_data(merchants_processed)
            X_train_m, X_val_m, X_test_m, y_train_m, y_val_m, y_test_m = self.data_loader.split_data(X_m, y_m)
            X_train_m_scaled, X_val_m_scaled, X_test_m_scaled, scaler_m = self.data_loader.scale_features(
                X_train_m, X_val_m, X_test_m
            )
            
            # Extract test set protected attributes for merchants
            test_size_m = len(y_test_m)
            total_size_m = len(y_train_m) + len(y_val_m) + len(y_test_m)
            test_start_idx_m = len(y_train_m) + len(y_val_m)
            
            test_protected_attrs_m = {}
            for attr_name, attr_values in protected_attrs_m.items():
                if len(attr_values) == total_size_m:
                    test_protected_attrs_m[attr_name] = attr_values[test_start_idx_m:]
            
            # Store merchants data
            self.merchants_data = {
                'X_test': X_test_m_scaled,
                'y_test': y_test_m,
                'feature_names': feature_names_m,
                'protected_attributes': test_protected_attrs_m
            }
            
            self.logger.info(f"Test data loaded with protected attributes")
            self.logger.info(f"Drivers protected attrs: {list(test_protected_attrs_d.keys())}")
            self.logger.info(f"Merchants protected attrs: {list(test_protected_attrs_m.keys())}")
            
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
    
    def analyze_fairness(self):
        """Analyze fairness for all loaded models"""
        self.logger.info("Analyzing fairness for all models...")
        
        try:
            # Analyze fairness for drivers models
            if self.drivers_models and self.drivers_data['protected_attributes']:
                self.logger.info("Analyzing fairness for drivers models...")
                
                for model_name, model in self.drivers_models.items():
                    try:
                        y_pred = model.predict(self.drivers_data['X_test'])
                        
                        fairness_results = self.bias_detector.comprehensive_fairness_analysis(
                            self.drivers_data['y_test'], y_pred, self.drivers_data['X_test'],
                            self.drivers_data['protected_attributes'], f"drivers_{model_name}"
                        )
                        
                        # Create fairness report
                        report = self.bias_detector.create_fairness_report(
                            f"drivers_{model_name}", fairness_results
                        )
                        
                        self.logger.info(f"Fairness analysis completed for drivers {model_name}")
                        
                    except Exception as e:
                        self.logger.error(f"Failed fairness analysis for drivers {model_name}: {str(e)}")
                        continue
            
            # Analyze fairness for merchants models
            if self.merchants_models and self.merchants_data['protected_attributes']:
                self.logger.info("Analyzing fairness for merchants models...")
                
                for model_name, model in self.merchants_models.items():
                    try:
                        y_pred = model.predict(self.merchants_data['X_test'])
                        
                        fairness_results = self.bias_detector.comprehensive_fairness_analysis(
                            self.merchants_data['y_test'], y_pred, self.merchants_data['X_test'],
                            self.merchants_data['protected_attributes'], f"merchants_{model_name}"
                        )
                        
                        # Create fairness report
                        report = self.bias_detector.create_fairness_report(
                            f"merchants_{model_name}", fairness_results
                        )
                        
                        self.logger.info(f"Fairness analysis completed for merchants {model_name}")
                        
                    except Exception as e:
                        self.logger.error(f"Failed fairness analysis for merchants {model_name}: {str(e)}")
                        continue
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to analyze fairness: {str(e)}")
            return False
    
    def run_fairness_analysis(self):
        """Run the complete fairness analysis process"""
        self.logger.info("="*60)
        self.logger.info("PROJECT NOVA - FAIRNESS ANALYSIS")
        self.logger.info("="*60)
        
        # Step 1: Load test data with protected attributes
        if not self.load_test_data():
            return False
        
        # Step 2: Load trained models
        if not self.load_trained_models():
            return False
        
        # Step 3: Analyze fairness
        if not self.analyze_fairness():
            return False
        
        self.logger.info("="*60)
        self.logger.info("FAIRNESS ANALYSIS COMPLETED SUCCESSFULLY!")
        self.logger.info("="*60)
        self.logger.info("Results saved in:")
        self.logger.info("  - results/fairness_analysis/")
        self.logger.info("="*60)
        
        return True

def main():
    """Main function"""
    analyzer = FairnessAnalysisRunner()
    success = analyzer.run_fairness_analysis()
    
    if not success:
        print("\n❌ Fairness analysis failed. Check the logs above.")
        sys.exit(1)

if __name__ == "__main__":
    main()