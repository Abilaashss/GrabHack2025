"""
Project Nova - ML Models Module
Implements various ML models for credit scoring
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import logging
from typing import Dict, Any, Tuple, List
import yaml
import pickle
import os

class MLModels:
    """ML Models for credit scoring"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize ML Models with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.logger = self._setup_logger()
        self.models = {}
        self.trained_models = {}
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/ml_models.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def initialize_models(self) -> Dict[str, Any]:
        """Initialize all ML models"""
        self.logger.info("Initializing ML models...")
        
        self.models = {
            # Gradient Boosting Models
            'xgboost': xgb.XGBRegressor(
                random_state=42,
                n_jobs=-1
            ),
            'lightgbm': lgb.LGBMRegressor(
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            'catboost': cb.CatBoostRegressor(
                random_state=42,
                verbose=False
            ),
            
            # Traditional ML Models
            'random_forest': RandomForestRegressor(
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                random_state=42
            ),
            
            # Linear Models
            'linear_regression': LinearRegression(),
            'ridge': Ridge(random_state=42),
            'lasso': Lasso(random_state=42),
            'elastic_net': ElasticNet(random_state=42),
            
            # Support Vector Machine
            'svr': SVR(),
            
            # Neural Network (sklearn)
            'mlp_regressor': MLPRegressor(
                random_state=42,
                max_iter=1000
            )
        }
        
        self.logger.info(f"Initialized {len(self.models)} models")
        return self.models
    
    def create_neural_network(self, input_dim: int, architecture: List[int] = None) -> keras.Model:
        """Create a neural network model using Keras"""
        if architecture is None:
            architecture = [128, 64, 32]
        
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.BatchNormalization()
        ])
        
        # Add hidden layers
        for i, units in enumerate(architecture):
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.Dropout(0.3))
            model.add(layers.BatchNormalization())
        
        # Output layer
        model.add(layers.Dense(1, activation='linear'))
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def train_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray = None, y_val: np.ndarray = None,
                   **kwargs) -> Any:
        """Train a specific model"""
        self.logger.info(f"Training {model_name}...")
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in initialized models")
        
        model = self.models[model_name]
        
        # Handle different model types
        if model_name == 'neural_network_keras':
            # Create and train Keras model
            model = self.create_neural_network(X_train.shape[1])
            
            # Prepare callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=20,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=10,
                    min_lr=1e-7
                ),
                keras.callbacks.ModelCheckpoint(
                    f'models/checkpoints/{model_name}_best.h5',
                    monitor='val_loss',
                    save_best_only=True
                )
            ]
            
            # Train model
            if X_val is not None and y_val is not None:
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=kwargs.get('epochs', 100),
                    batch_size=kwargs.get('batch_size', 32),
                    callbacks=callbacks,
                    verbose=1
                )
            else:
                history = model.fit(
                    X_train, y_train,
                    epochs=kwargs.get('epochs', 100),
                    batch_size=kwargs.get('batch_size', 32),
                    verbose=1
                )
            
            # Save training history
            with open(f'models/checkpoints/{model_name}_history.pkl', 'wb') as f:
                pickle.dump(history.history, f)
        
        elif model_name in ['xgboost', 'lightgbm', 'catboost']:
            # Handle gradient boosting models with validation
            if X_val is not None and y_val is not None:
                if model_name == 'xgboost':
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=50,
                        verbose=False
                    )
                elif model_name == 'lightgbm':
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                    )
                elif model_name == 'catboost':
                    model.fit(
                        X_train, y_train,
                        eval_set=(X_val, y_val),
                        early_stopping_rounds=50,
                        verbose=False
                    )
            else:
                model.fit(X_train, y_train)
        
        else:
            # Standard sklearn models
            model.fit(X_train, y_train)
        
        # Store trained model
        self.trained_models[model_name] = model
        
        # Save model
        self.save_model(model, model_name)
        
        self.logger.info(f"Training completed for {model_name}")
        return model
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        """Train all initialized models"""
        self.logger.info("Training all models...")
        
        trained_models = {}
        
        for model_name in self.models.keys():
            try:
                model = self.train_model(model_name, X_train, y_train, X_val, y_val)
                trained_models[model_name] = model
                self.logger.info(f"Successfully trained {model_name}")
            except Exception as e:
                self.logger.error(f"Failed to train {model_name}: {str(e)}")
                continue
        
        # Train Keras neural network separately
        try:
            nn_model = self.train_model('neural_network_keras', X_train, y_train, X_val, y_val)
            trained_models['neural_network_keras'] = nn_model
            self.logger.info("Successfully trained neural_network_keras")
        except Exception as e:
            self.logger.error(f"Failed to train neural_network_keras: {str(e)}")
        
        self.trained_models.update(trained_models)
        self.logger.info(f"Training completed for {len(trained_models)} models")
        
        return trained_models
    
    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """Make predictions using a trained model"""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found in trained models")
        
        model = self.trained_models[model_name]
        
        if model_name == 'neural_network_keras':
            predictions = model.predict(X).flatten()
        else:
            predictions = model.predict(X)
        
        return predictions
    
    def predict_all_models(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Make predictions using all trained models"""
        predictions = {}
        
        for model_name in self.trained_models.keys():
            try:
                pred = self.predict(model_name, X)
                predictions[model_name] = pred
            except Exception as e:
                self.logger.error(f"Failed to predict with {model_name}: {str(e)}")
                continue
        
        return predictions
    
    def save_model(self, model: Any, model_name: str) -> None:
        """Save a trained model"""
        model_path = f"models/saved_models/{model_name}"
        
        # Determine model type from the actual model, not just the name
        model_type = type(model).__name__.lower()
        
        if 'keras' in model_type or hasattr(model, 'save'):
            try:
                model.save(f"{model_path}.h5")
            except:
                joblib.dump(model, f"{model_path}.pkl")
        elif 'xgb' in model_type or 'xgboost' in model_type:
            model.save_model(f"{model_path}.json")
        elif 'lgb' in model_type or 'lightgbm' in model_type:
            if hasattr(model, 'booster_'):
                model.booster_.save_model(f"{model_path}.txt")
            else:
                model.save_model(f"{model_path}.txt")
        elif 'catboost' in model_type:
            model.save_model(f"{model_path}.cbm")
        else:
            # Save sklearn models and ensembles using joblib
            joblib.dump(model, f"{model_path}.pkl")
        
        self.logger.info(f"Model {model_name} saved to {model_path}")
    
    def load_model(self, model_name: str) -> Any:
        """Load a saved model"""
        model_path = f"models/saved_models/{model_name}"
        
        if model_name == 'neural_network_keras':
            model = keras.models.load_model(f"{model_path}.h5")
        elif model_name == 'xgboost':
            model = xgb.XGBRegressor()
            model.load_model(f"{model_path}.json")
        elif model_name == 'lightgbm':
            model = lgb.Booster(model_file=f"{model_path}.txt")
        elif model_name == 'catboost':
            model = cb.CatBoostRegressor()
            model.load_model(f"{model_path}.cbm")
        else:
            model = joblib.load(f"{model_path}.pkl")
        
        self.trained_models[model_name] = model
        self.logger.info(f"Model {model_name} loaded from {model_path}")
        
        return model
    
    def create_ensemble_model(self, models_dict: Dict[str, Any], method: str = 'average') -> Any:
        """Create an ensemble model from multiple trained models"""
        self.logger.info(f"Creating ensemble model using {method} method...")
        
        class EnsembleModel:
            def __init__(self, models, method='average'):
                self.models = models
                self.method = method
                self.weights = None
                
                if method == 'weighted':
                    # Initialize equal weights
                    self.weights = np.ones(len(models)) / len(models)
            
            def predict(self, X):
                predictions = []
                for model_name, model in self.models.items():
                    if model_name == 'neural_network_keras':
                        pred = model.predict(X).flatten()
                    else:
                        pred = model.predict(X)
                    predictions.append(pred)
                
                predictions = np.array(predictions)
                
                if self.method == 'average':
                    return np.mean(predictions, axis=0)
                elif self.method == 'weighted' and self.weights is not None:
                    return np.average(predictions, axis=0, weights=self.weights)
                elif self.method == 'median':
                    return np.median(predictions, axis=0)
                else:
                    return np.mean(predictions, axis=0)
            
            def set_weights(self, weights):
                if len(weights) == len(self.models):
                    self.weights = np.array(weights)
                else:
                    raise ValueError("Number of weights must match number of models")
        
        ensemble = EnsembleModel(models_dict, method)
        
        # Save ensemble model
        joblib.dump(ensemble, f"models/saved_models/ensemble_{method}.pkl")
        
        self.logger.info(f"Ensemble model created and saved")
        return ensemble
    
    def get_feature_importance(self, model_name: str, feature_names: List[str] = None) -> Dict[str, float]:
        """Get feature importance from tree-based models"""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found in trained models")
        
        model = self.trained_models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        else:
            self.logger.warning(f"Model {model_name} does not have feature importance")
            return {}
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        importance_dict = dict(zip(feature_names, importances))
        
        # Sort by importance
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return importance_dict