"""
Project Nova - Model Utilities
Utility functions for model loading, prediction, and deployment
"""

import os
import joblib
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import logging
from tensorflow import keras
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

class ModelLoader:
    """Utility class for loading and using trained models"""
    
    def __init__(self, models_dir: str = "models/saved_models"):
        """Initialize ModelLoader"""
        self.models_dir = models_dir
        self.loaded_models = {}
        self.scaler = None
        self.feature_names = None
        
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def load_scaler(self, scaler_path: str = "models/scaler.pkl") -> None:
        """Load the feature scaler"""
        try:
            self.scaler = joblib.load(scaler_path)
            self.logger.info("Scaler loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load scaler: {str(e)}")
            raise
    
    def load_feature_names(self, feature_names_path: str = "models/feature_names.json") -> None:
        """Load feature names"""
        try:
            with open(feature_names_path, 'r') as f:
                self.feature_names = json.load(f)
            self.logger.info(f"Loaded {len(self.feature_names)} feature names")
        except Exception as e:
            self.logger.warning(f"Could not load feature names: {str(e)}")
    
    def load_model(self, model_name: str) -> Any:
        """Load a specific model"""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        model_path = os.path.join(self.models_dir, model_name)
        
        try:
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
            elif 'ensemble' in model_name:
                model = joblib.load(f"{model_path}.pkl")
            else:
                # Standard sklearn models
                model = joblib.load(f"{model_path}.pkl")
            
            self.loaded_models[model_name] = model
            self.logger.info(f"Model {model_name} loaded successfully")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise
    
    def load_all_models(self) -> Dict[str, Any]:
        """Load all available models"""
        if not os.path.exists(self.models_dir):
            raise ValueError(f"Models directory {self.models_dir} does not exist")
        
        model_files = os.listdir(self.models_dir)
        model_names = set()
        
        # Extract model names from files
        for file in model_files:
            if file.endswith(('.pkl', '.h5', '.json', '.txt', '.cbm')):
                model_name = file.split('.')[0]
                model_names.add(model_name)
        
        loaded_models = {}
        for model_name in model_names:
            try:
                model = self.load_model(model_name)
                loaded_models[model_name] = model
            except Exception as e:
                self.logger.error(f"Failed to load {model_name}: {str(e)}")
                continue
        
        self.logger.info(f"Loaded {len(loaded_models)} models")
        return loaded_models
    
    def predict_single(self, model_name: str, features: np.ndarray) -> float:
        """Make prediction for a single sample"""
        if model_name not in self.loaded_models:
            self.load_model(model_name)
        
        model = self.loaded_models[model_name]
        
        # Ensure features are 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Scale features if scaler is available
        if self.scaler is not None:
            features = self.scaler.transform(features)
        
        # Make prediction
        if model_name == 'neural_network_keras':
            prediction = model.predict(features).flatten()[0]
        elif model_name == 'lightgbm' and hasattr(model, 'predict'):
            prediction = model.predict(features)[0]
        else:
            prediction = model.predict(features)[0]
        
        return float(prediction)
    
    def predict_batch(self, model_name: str, features: np.ndarray) -> np.ndarray:
        """Make predictions for multiple samples"""
        if model_name not in self.loaded_models:
            self.load_model(model_name)
        
        model = self.loaded_models[model_name]
        
        # Scale features if scaler is available
        if self.scaler is not None:
            features = self.scaler.transform(features)
        
        # Make predictions
        if model_name == 'neural_network_keras':
            predictions = model.predict(features).flatten()
        else:
            predictions = model.predict(features)
        
        return predictions
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a model"""
        info = {
            'model_name': model_name,
            'loaded': model_name in self.loaded_models,
            'file_exists': False,
            'model_type': 'unknown'
        }
        
        # Check if model file exists
        model_path = os.path.join(self.models_dir, model_name)
        
        for ext in ['.pkl', '.h5', '.json', '.txt', '.cbm']:
            if os.path.exists(f"{model_path}{ext}"):
                info['file_exists'] = True
                info['file_extension'] = ext
                break
        
        # Determine model type
        if 'xgboost' in model_name:
            info['model_type'] = 'XGBoost'
        elif 'lightgbm' in model_name:
            info['model_type'] = 'LightGBM'
        elif 'catboost' in model_name:
            info['model_type'] = 'CatBoost'
        elif 'neural_network' in model_name:
            info['model_type'] = 'Neural Network'
        elif 'ensemble' in model_name:
            info['model_type'] = 'Ensemble'
        elif any(name in model_name for name in ['random_forest', 'gradient_boosting']):
            info['model_type'] = 'Tree-based'
        elif any(name in model_name for name in ['linear', 'ridge', 'lasso', 'elastic']):
            info['model_type'] = 'Linear'
        
        return info

class CreditScorePredictor:
    """High-level interface for credit score prediction"""
    
    def __init__(self, model_name: str = None, models_dir: str = "models/saved_models"):
        """Initialize CreditScorePredictor"""
        self.model_loader = ModelLoader(models_dir)
        self.model_name = model_name
        
        # Load scaler and feature names
        try:
            self.model_loader.load_scaler()
        except:
            pass
        
        try:
            self.model_loader.load_feature_names()
        except:
            pass
        
        # Load best model if no specific model provided
        if model_name is None:
            self.model_name = self._get_best_model()
        
        # Load the model
        if self.model_name:
            self.model_loader.load_model(self.model_name)
    
    def _get_best_model(self) -> Optional[str]:
        """Get the best performing model from evaluation results"""
        try:
            with open('results/metrics/all_models_evaluation.json', 'r') as f:
                evaluation_results = json.load(f)
            
            # Find model with highest R² score
            best_model = max(
                evaluation_results.items(),
                key=lambda x: x[1].get('r2_score', 0)
            )
            
            return best_model[0]
            
        except Exception:
            # Fallback to first available model
            try:
                models = self.model_loader.load_all_models()
                return list(models.keys())[0] if models else None
            except:
                return None
    
    def predict_from_dict(self, features_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Predict credit score from feature dictionary"""
        if not self.model_name:
            raise ValueError("No model available for prediction")
        
        # Convert dict to array (assuming feature names are available)
        if self.model_loader.feature_names:
            features_array = np.array([
                features_dict.get(name, 0) for name in self.model_loader.feature_names
            ])
        else:
            # If no feature names, assume dict values are in correct order
            features_array = np.array(list(features_dict.values()))
        
        # Make prediction
        prediction = self.model_loader.predict_single(self.model_name, features_array)
        
        # Interpret prediction
        interpretation = self._interpret_credit_score(prediction)
        
        return {
            'credit_score': round(prediction, 0),
            'risk_category': interpretation['category'],
            'risk_level': interpretation['level'],
            'description': interpretation['description'],
            'model_used': self.model_name
        }
    
    def predict_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict credit scores for a DataFrame"""
        if not self.model_name:
            raise ValueError("No model available for prediction")
        
        # Convert DataFrame to numpy array
        features_array = df.values
        
        # Make predictions
        predictions = self.model_loader.predict_batch(self.model_name, features_array)
        
        # Add predictions to DataFrame
        result_df = df.copy()
        result_df['predicted_credit_score'] = predictions
        result_df['risk_category'] = result_df['predicted_credit_score'].apply(
            lambda x: self._interpret_credit_score(x)['category']
        )
        
        return result_df
    
    def _interpret_credit_score(self, score: float) -> Dict[str, str]:
        """Interpret credit score into risk categories"""
        if score >= 800:
            return {
                'category': 'Excellent',
                'level': 'Very Low Risk',
                'description': 'Exceptional credit profile with very low default risk'
            }
        elif score >= 740:
            return {
                'category': 'Very Good',
                'level': 'Low Risk',
                'description': 'Strong credit profile with low default risk'
            }
        elif score >= 670:
            return {
                'category': 'Good',
                'level': 'Moderate Risk',
                'description': 'Good credit profile with moderate default risk'
            }
        elif score >= 580:
            return {
                'category': 'Fair',
                'level': 'High Risk',
                'description': 'Fair credit profile with elevated default risk'
            }
        else:
            return {
                'category': 'Poor',
                'level': 'Very High Risk',
                'description': 'Poor credit profile with very high default risk'
            }
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models with their information"""
        models_info = []
        
        if os.path.exists(self.model_loader.models_dir):
            model_files = os.listdir(self.model_loader.models_dir)
            model_names = set()
            
            for file in model_files:
                if file.endswith(('.pkl', '.h5', '.json', '.txt', '.cbm')):
                    model_name = file.split('.')[0]
                    model_names.add(model_name)
            
            for model_name in model_names:
                info = self.model_loader.get_model_info(model_name)
                models_info.append(info)
        
        return models_info
    
    def switch_model(self, model_name: str) -> None:
        """Switch to a different model"""
        self.model_name = model_name
        self.model_loader.load_model(model_name)

def save_feature_names(feature_names: List[str], 
                      filepath: str = "models/feature_names.json") -> None:
    """Save feature names to file"""
    with open(filepath, 'w') as f:
        json.dump(feature_names, f, indent=2)

def load_evaluation_results(filepath: str = "results/metrics/all_models_evaluation.json") -> Dict[str, Any]:
    """Load model evaluation results"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load evaluation results: {str(e)}")
        return {}

def get_model_summary() -> pd.DataFrame:
    """Get summary of all trained models"""
    try:
        evaluation_results = load_evaluation_results()
        
        if not evaluation_results:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(evaluation_results).T
        
        # Select key metrics
        key_metrics = ['r2_score', 'rmse', 'mae', 'mape', 'within_25_points_accuracy', 'within_50_points_accuracy']
        available_metrics = [m for m in key_metrics if m in df.columns]
        
        summary_df = df[available_metrics].copy()
        summary_df = summary_df.sort_values('r2_score', ascending=False)
        
        return summary_df
        
    except Exception as e:
        logging.error(f"Failed to create model summary: {str(e)}")
        return pd.DataFrame()