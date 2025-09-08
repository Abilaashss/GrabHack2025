"""
Project Nova - Hyperparameter Tuning Module
Implements various hyperparameter optimization techniques
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import optuna
from optuna.samplers import TPESampler
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from tensorflow import keras
import joblib
import logging
import yaml
import json
from typing import Dict, Any, Tuple, List, Callable
import time

class HyperparameterTuner:
    """Hyperparameter tuning for ML models"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize HyperparameterTuner with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.logger = self._setup_logger()
        self.best_params = {}
        self.tuning_results = {}
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/hyperparameter_tuning.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def tune_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray,
                     method: str = 'optuna') -> Dict[str, Any]:
        """Tune XGBoost hyperparameters"""
        self.logger.info(f"Tuning XGBoost using {method}...")
        
        if method == 'optuna':
            return self._tune_xgboost_optuna(X_train, y_train, X_val, y_val)
        elif method == 'grid_search':
            return self._tune_xgboost_grid_search(X_train, y_train)
        elif method == 'random_search':
            return self._tune_xgboost_random_search(X_train, y_train)
        else:
            raise ValueError(f"Unknown tuning method: {method}")
    
    def _tune_xgboost_optuna(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Tune XGBoost using Optuna"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': 42
            }
            
            model = xgb.XGBRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=False
            )
            
            y_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            return mse
        
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42)
        )
        
        study.optimize(
            objective,
            n_trials=self.config['hyperparameter_tuning']['n_trials'],
            timeout=self.config['hyperparameter_tuning']['timeout']
        )
        
        best_params = study.best_params
        best_score = study.best_value
        
        self.best_params['xgboost'] = best_params
        self.tuning_results['xgboost'] = {
            'best_params': best_params,
            'best_score': best_score,
            'method': 'optuna',
            'n_trials': len(study.trials)
        }
        
        # Save results
        self._save_tuning_results('xgboost', study)
        
        return best_params
    
    def _tune_xgboost_grid_search(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Tune XGBoost using Grid Search"""
        
        param_grid = {
            'n_estimators': self.config['models']['xgboost']['n_estimators'][:2],  # Limit for speed
            'max_depth': self.config['models']['xgboost']['max_depth'][:2],
            'learning_rate': self.config['models']['xgboost']['learning_rate'][:2]
        }
        
        model = xgb.XGBRegressor(random_state=42)
        
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=self.config['evaluation']['cross_validation']['cv_folds'],
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        best_params = grid_search.best_params_
        best_score = -grid_search.best_score_
        
        self.best_params['xgboost'] = best_params
        self.tuning_results['xgboost'] = {
            'best_params': best_params,
            'best_score': best_score,
            'method': 'grid_search'
        }
        
        return best_params
    
    def tune_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray,
                      method: str = 'optuna') -> Dict[str, Any]:
        """Tune LightGBM hyperparameters"""
        self.logger.info(f"Tuning LightGBM using {method}...")
        
        if method == 'optuna':
            return self._tune_lightgbm_optuna(X_train, y_train, X_val, y_val)
        else:
            raise ValueError(f"Method {method} not implemented for LightGBM")
    
    def _tune_lightgbm_optuna(self, X_train: np.ndarray, y_train: np.ndarray,
                             X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Tune LightGBM using Optuna"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'random_state': 42,
                'verbose': -1
            }
            
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            
            y_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            return mse
        
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42)
        )
        
        study.optimize(
            objective,
            n_trials=self.config['hyperparameter_tuning']['n_trials'],
            timeout=self.config['hyperparameter_tuning']['timeout']
        )
        
        best_params = study.best_params
        best_score = study.best_value
        
        self.best_params['lightgbm'] = best_params
        self.tuning_results['lightgbm'] = {
            'best_params': best_params,
            'best_score': best_score,
            'method': 'optuna',
            'n_trials': len(study.trials)
        }
        
        # Save results
        self._save_tuning_results('lightgbm', study)
        
        return best_params
    
    def tune_catboost(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray,
                      method: str = 'optuna') -> Dict[str, Any]:
        """Tune CatBoost hyperparameters"""
        self.logger.info(f"Tuning CatBoost using {method}...")
        
        if method == 'optuna':
            return self._tune_catboost_optuna(X_train, y_train, X_val, y_val)
        else:
            raise ValueError(f"Method {method} not implemented for CatBoost")
    
    def _tune_catboost_optuna(self, X_train: np.ndarray, y_train: np.ndarray,
                             X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Tune CatBoost using Optuna"""
        
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 100, 1000),
                'depth': trial.suggest_int('depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'random_state': 42,
                'verbose': False
            }
            
            model = cb.CatBoostRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                early_stopping_rounds=50,
                verbose=False
            )
            
            y_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            return mse
        
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42)
        )
        
        study.optimize(
            objective,
            n_trials=self.config['hyperparameter_tuning']['n_trials'],
            timeout=self.config['hyperparameter_tuning']['timeout']
        )
        
        best_params = study.best_params
        best_score = study.best_value
        
        self.best_params['catboost'] = best_params
        self.tuning_results['catboost'] = {
            'best_params': best_params,
            'best_score': best_score,
            'method': 'optuna',
            'n_trials': len(study.trials)
        }
        
        # Save results
        self._save_tuning_results('catboost', study)
        
        return best_params
    
    def tune_neural_network(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray,
                           method: str = 'optuna') -> Dict[str, Any]:
        """Tune Neural Network hyperparameters"""
        self.logger.info(f"Tuning Neural Network using {method}...")
        
        if method == 'optuna':
            return self._tune_neural_network_optuna(X_train, y_train, X_val, y_val)
        else:
            raise ValueError(f"Method {method} not implemented for Neural Network")
    
    def _tune_neural_network_optuna(self, X_train: np.ndarray, y_train: np.ndarray,
                                   X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Tune Neural Network using Optuna"""
        
        def objective(trial):
            # Architecture parameters
            n_layers = trial.suggest_int('n_layers', 2, 5)
            layers_config = []
            
            for i in range(n_layers):
                n_units = trial.suggest_int(f'n_units_l{i}', 32, 512)
                layers_config.append(n_units)
            
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            
            # Create model
            model = keras.Sequential([
                keras.layers.Input(shape=(X_train.shape[1],)),
                keras.layers.BatchNormalization()
            ])
            
            for units in layers_config:
                model.add(keras.layers.Dense(units, activation='relu'))
                model.add(keras.layers.Dropout(dropout_rate))
                model.add(keras.layers.BatchNormalization())
            
            model.add(keras.layers.Dense(1, activation='linear'))
            
            # Compile model
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss='mse',
                metrics=['mae']
            )
            
            # Train model
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=batch_size,
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Get best validation loss
            best_val_loss = min(history.history['val_loss'])
            return best_val_loss
        
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42)
        )
        
        study.optimize(
            objective,
            n_trials=50,  # Reduced for neural networks
            timeout=self.config['hyperparameter_tuning']['timeout']
        )
        
        best_params = study.best_params
        best_score = study.best_value
        
        self.best_params['neural_network'] = best_params
        self.tuning_results['neural_network'] = {
            'best_params': best_params,
            'best_score': best_score,
            'method': 'optuna',
            'n_trials': len(study.trials)
        }
        
        # Save results
        self._save_tuning_results('neural_network', study)
        
        return best_params
    
    def tune_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Tune hyperparameters for all models"""
        self.logger.info("Starting hyperparameter tuning for all models...")
        
        models_to_tune = ['xgboost', 'lightgbm', 'catboost', 'neural_network']
        all_best_params = {}
        
        for model_name in models_to_tune:
            try:
                self.logger.info(f"Tuning {model_name}...")
                start_time = time.time()
                
                if model_name == 'xgboost':
                    best_params = self.tune_xgboost(X_train, y_train, X_val, y_val)
                elif model_name == 'lightgbm':
                    best_params = self.tune_lightgbm(X_train, y_train, X_val, y_val)
                elif model_name == 'catboost':
                    best_params = self.tune_catboost(X_train, y_train, X_val, y_val)
                elif model_name == 'neural_network':
                    best_params = self.tune_neural_network(X_train, y_train, X_val, y_val)
                
                end_time = time.time()
                tuning_time = end_time - start_time
                
                all_best_params[model_name] = best_params
                self.tuning_results[model_name]['tuning_time'] = tuning_time
                
                self.logger.info(f"Completed tuning {model_name} in {tuning_time:.2f} seconds")
                
            except Exception as e:
                self.logger.error(f"Failed to tune {model_name}: {str(e)}")
                continue
        
        # Save all results
        self._save_all_tuning_results()
        
        return all_best_params
    
    def _save_tuning_results(self, model_name: str, study: optuna.Study) -> None:
        """Save individual model tuning results"""
        
        # Save study object
        joblib.dump(study, f'models/hyperparameters/{model_name}_study.pkl')
        
        # Save best parameters
        with open(f'models/hyperparameters/{model_name}_best_params.json', 'w') as f:
            json.dump(study.best_params, f, indent=2)
        
        # Save trials dataframe
        trials_df = study.trials_dataframe()
        trials_df.to_csv(f'models/hyperparameters/{model_name}_trials.csv', index=False)
        
        self.logger.info(f"Tuning results saved for {model_name}")
    
    def _save_all_tuning_results(self) -> None:
        """Save all tuning results summary"""
        
        # Save best parameters for all models
        with open('models/hyperparameters/all_best_params.json', 'w') as f:
            json.dump(self.best_params, f, indent=2)
        
        # Save tuning results summary
        with open('models/hyperparameters/tuning_summary.json', 'w') as f:
            json.dump(self.tuning_results, f, indent=2)
        
        # Create summary dataframe
        summary_data = []
        for model_name, results in self.tuning_results.items():
            summary_data.append({
                'model': model_name,
                'best_score': results['best_score'],
                'method': results['method'],
                'n_trials': results.get('n_trials', 'N/A'),
                'tuning_time': results.get('tuning_time', 'N/A')
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('models/hyperparameters/tuning_summary.csv', index=False)
        
        self.logger.info("All tuning results saved")
    
    def load_best_params(self, model_name: str) -> Dict[str, Any]:
        """Load best parameters for a model"""
        try:
            with open(f'models/hyperparameters/{model_name}_best_params.json', 'r') as f:
                best_params = json.load(f)
            return best_params
        except FileNotFoundError:
            self.logger.warning(f"Best parameters not found for {model_name}")
            return {}
    
    def get_tuning_summary(self) -> pd.DataFrame:
        """Get summary of all tuning results"""
        try:
            summary_df = pd.read_csv('models/hyperparameters/tuning_summary.csv')
            return summary_df
        except FileNotFoundError:
            self.logger.warning("Tuning summary not found")
            return pd.DataFrame()