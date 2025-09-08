#!/usr/bin/env python3
"""
Project Nova - Training Only Script
Train models only without evaluation, fairness, or visualization
"""

import os
import sys
import logging
import time
from datetime import datetime

# Add src to path
sys.path.append('src')

from preprocessing.data_loader import DataLoader
from models.ml_models import MLModels

class ModelTrainingRunner:
    """Train models only"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.data_loader = DataLoader()
        
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
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/training_only.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def load_and_preprocess_data(self):
        """Load and preprocess data for both datasets"""
        self.logger.info("Loading and preprocessing data...")
        
        try:
            # Load datasets
            drivers_df, merchants_df = self.data_loader.load_data()
            
            # Process drivers dataset
            self.logger.info("Processing drivers dataset...")
            drivers_prepared = self.data_loader.prepare_drivers_dataset(drivers_df)
            drivers_processed = self.data_loader.preprocess_features(drivers_prepared)
            X_d, y_d, feature_names_d = self.data_loader.prepare_ml_data(drivers_processed)
            X_train_d, X_val_d, X_test_d, y_train_d, y_val_d, y_test_d = self.data_loader.split_data(X_d, y_d)
            X_train_d_scaled, X_val_d_scaled, X_test_d_scaled, scaler_d = self.data_loader.scale_features(
                X_train_d, X_val_d, X_test_d
            )
            
            # Store drivers data
            self.drivers_data = {
                'X_train': X_train_d_scaled, 'X_val': X_val_d_scaled, 'X_test': X_test_d_scaled,
                'y_train': y_train_d, 'y_val': y_val_d, 'y_test': y_test_d,
                'feature_names': feature_names_d, 'scaler': scaler_d
            }
            
            # Process merchants dataset
            self.logger.info("Processing merchants dataset...")
            merchants_prepared = self.data_loader.prepare_merchants_dataset(merchants_df)
            merchants_processed = self.data_loader.preprocess_features(merchants_prepared)
            X_m, y_m, feature_names_m = self.data_loader.prepare_ml_data(merchants_processed)
            X_train_m, X_val_m, X_test_m, y_train_m, y_val_m, y_test_m = self.data_loader.split_data(X_m, y_m)
            X_train_m_scaled, X_val_m_scaled, X_test_m_scaled, scaler_m = self.data_loader.scale_features(
                X_train_m, X_val_m, X_test_m
            )
            
            # Store merchants data
            self.merchants_data = {
                'X_train': X_train_m_scaled, 'X_val': X_val_m_scaled, 'X_test': X_test_m_scaled,
                'y_train': y_train_m, 'y_val': y_val_m, 'y_test': y_test_m,
                'feature_names': feature_names_m, 'scaler': scaler_m
            }
            
            self.logger.info("Data preprocessing completed successfully")
            self.logger.info(f"Drivers: Train={X_train_d.shape}, Val={X_val_d.shape}, Test={X_test_d.shape}")
            self.logger.info(f"Merchants: Train={X_train_m.shape}, Val={X_val_m.shape}, Test={X_test_m.shape}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load and preprocess data: {str(e)}")
            return False
    
    def train_models(self):
        """Train ML models for both datasets"""
        self.logger.info("Starting model training...")
        
        try:
            # Train drivers models
            self.logger.info("Training models for drivers...")
            drivers_ml = MLModels()
            drivers_ml.initialize_models()
            
            drivers_trained = drivers_ml.train_all_models(
                self.drivers_data['X_train'], self.drivers_data['y_train'],
                self.drivers_data['X_val'], self.drivers_data['y_val']
            )
            
            # Save drivers models with proper naming
            for model_name, model in drivers_trained.items():
                drivers_ml.save_model(model, f"drivers_{model_name}")
            
            self.drivers_models = drivers_trained
            
            # Train merchants models
            self.logger.info("Training models for merchants...")
            merchants_ml = MLModels()
            merchants_ml.initialize_models()
            
            merchants_trained = merchants_ml.train_all_models(
                self.merchants_data['X_train'], self.merchants_data['y_train'],
                self.merchants_data['X_val'], self.merchants_data['y_val']
            )
            
            # Save merchants models with proper naming
            for model_name, model in merchants_trained.items():
                merchants_ml.save_model(model, f"merchants_{model_name}")
            
            self.merchants_models = merchants_trained
            
            self.logger.info(f"Successfully trained {len(drivers_trained)} drivers models and {len(merchants_trained)} merchants models")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to train models: {str(e)}")
            return False
    
    def run_training(self):
        """Run the complete training process"""
        start_time = time.time()
        
        self.logger.info("="*60)
        self.logger.info("PROJECT NOVA - MODEL TRAINING ONLY")
        self.logger.info("="*60)
        
        # Step 1: Load and preprocess data
        if not self.load_and_preprocess_data():
            return False
        
        # Step 2: Train models
        if not self.train_models():
            return False
        
        # Training completion
        end_time = time.time()
        execution_time = end_time - start_time
        
        self.logger.info("="*60)
        self.logger.info("MODEL TRAINING COMPLETED SUCCESSFULLY!")
        self.logger.info(f"Total execution time: {execution_time:.2f} seconds")
        self.logger.info(f"Drivers models trained: {len(self.drivers_models)}")
        self.logger.info(f"Merchants models trained: {len(self.merchants_models)}")
        self.logger.info("="*60)
        self.logger.info("Models saved in: models/saved_models/")
        self.logger.info("Next steps:")
        self.logger.info("  - Run evaluation: python evaluate_models.py")
        self.logger.info("  - Run fairness analysis: python analyze_fairness.py")
        self.logger.info("  - Create visualizations: python create_visualizations.py")
        self.logger.info("="*60)
        
        return True

def main():
    """Main function"""
    trainer = ModelTrainingRunner()
    success = trainer.run_training()
    
    if not success:
        print("\n❌ Training failed. Check the logs above.")
        sys.exit(1)

if __name__ == "__main__":
    main()