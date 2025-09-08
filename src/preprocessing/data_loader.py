"""
Project Nova - Data Loading and Preprocessing Module
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import logging
from typing import Tuple, Dict, List
import yaml

class DataLoader:
    """Data loading and preprocessing for credit scoring"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize DataLoader with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.logger = self._setup_logger()
        self.preprocessor = None
        self.label_encoders = {}
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/data_preprocessing.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load drivers and merchants datasets"""
        self.logger.info("Loading datasets...")
        
        drivers_df = pd.read_csv(self.config['data']['drivers_path'])
        merchants_df = pd.read_csv(self.config['data']['merchants_path'])
        
        self.logger.info(f"Drivers dataset shape: {drivers_df.shape}")
        self.logger.info(f"Merchants dataset shape: {merchants_df.shape}")
        
        return drivers_df, merchants_df
    
    def prepare_drivers_dataset(self, drivers_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare drivers dataset for ML"""
        self.logger.info("Preparing drivers dataset...")
        
        # Create a copy to avoid modifying original
        processed_df = drivers_df.copy()
        
        # Rename columns for consistency
        processed_df = processed_df.rename(columns={
            'monthly_earnings': 'monthly_revenue',
            'earnings_growth_4w': 'revenue_growth_4w'
        })
        
        self.logger.info(f"Drivers dataset shape: {processed_df.shape}")
        return processed_df
    
    def prepare_merchants_dataset(self, merchants_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare merchants dataset for ML"""
        self.logger.info("Preparing merchants dataset...")
        
        # Create a copy to avoid modifying original
        processed_df = merchants_df.copy()
        
        # Rename columns for consistency
        processed_df = processed_df.rename(columns={
            'monthly_sales': 'monthly_revenue', 
            'sales_growth_4w': 'revenue_growth_4w'
        })
        
        self.logger.info(f"Merchants dataset shape: {processed_df.shape}")
        return processed_df
    
    def preprocess_features(self, df: pd.DataFrame, extract_protected_attrs: bool = False) -> pd.DataFrame:
        """Preprocess features for ML models"""
        self.logger.info("Preprocessing features...")
        
        # Create a copy to avoid modifying original
        processed_df = df.copy()
        
        # Convert categorical columns to strings to avoid pandas categorical issues
        processed_df = self._convert_categorical_to_string(processed_df)
        
        # Handle missing values
        processed_df = self._handle_missing_values(processed_df)
        
        # Feature engineering
        processed_df = self._engineer_features(processed_df)
        
        # Extract protected attributes BEFORE one-hot encoding if requested
        if extract_protected_attrs:
            self.protected_attributes = self.get_protected_attributes_before_encoding(processed_df)
        
        # Encode categorical variables
        processed_df = self._encode_categorical_features(processed_df)
        
        return processed_df
    
    def _convert_categorical_to_string(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert pandas categorical columns to strings to avoid encoding issues"""
        self.logger.info("Converting categorical columns to strings...")
        
        for col in df.columns:
            if df[col].dtype.name == 'category':
                df[col] = df[col].astype(str)
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        self.logger.info("Handling missing values...")
        
        # Numerical columns - fill with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        # Categorical columns - fill with mode or 'Unknown'
        categorical_cols = df.select_dtypes(include=['object', 'bool', 'category']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                # Convert categorical to string first to avoid category issues
                if df[col].dtype.name == 'category':
                    df[col] = df[col].astype(str)
                
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val[0])
                else:
                    df[col] = df[col].fillna('Unknown')
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer new features"""
        self.logger.info("Engineering features...")
        
        # Revenue per rating point
        df['revenue_per_rating'] = df['monthly_revenue'] / (df['average_rating'] + 1e-8)
        
        # Complaint to compliment ratio
        df['complaint_compliment_ratio'] = df['customer_complaints'] / (df['customer_compliments'] + 1)
        
        # Tenure categories
        df['tenure_category'] = pd.cut(df['tenure_months'], 
                                     bins=[0, 12, 36, 60, float('inf')], 
                                     labels=['New', 'Intermediate', 'Experienced', 'Veteran']).astype(str)
        
        # Revenue stability categories
        df['stability_category'] = pd.cut(df['earnings_stability'],
                                        bins=[0, 0.3, 0.7, 1.0],
                                        labels=['Low', 'Medium', 'High']).astype(str)
        
        # Digital adoption score
        df['digital_adoption_score'] = df['digital_payment_ratio']
        
        # Performance score (combine multiple metrics)
        if 'completion_rate' in df.columns:
            # Driver-specific performance score
            df['performance_score'] = (df['completion_rate'] * 0.4 + 
                                     df['acceptance_rate'] * 0.3 + 
                                     (1 - df['cancel_rate']) * 0.3)
        elif 'order_acceptance_rate' in df.columns:
            # Merchant-specific performance score
            df['performance_score'] = (df['order_acceptance_rate'] * 0.5 + 
                                     (1 - df['order_error_rate']) * 0.5)
        
        return df
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features"""
        self.logger.info("Encoding categorical features...")
        
        # Get categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove target column if present
        if self.config['data']['target_column'] in categorical_cols:
            categorical_cols.remove(self.config['data']['target_column'])
        
        # Label encode ordinal features
        ordinal_features = ['education_level', 'tenure_category', 'stability_category']
        ordinal_mappings = {
            'education_level': {'High School': 1, 'Diploma': 2, 'Bachelors': 3, 'Masters': 4, 'PhD': 5},
            'tenure_category': {'New': 1, 'Intermediate': 2, 'Experienced': 3, 'Veteran': 4},
            'stability_category': {'Low': 1, 'Medium': 2, 'High': 3}
        }
        
        for feature in ordinal_features:
            if feature in df.columns:
                # Convert to string first to avoid categorical issues
                df[feature] = df[feature].astype(str)
                # Map values and fill NaN with 0
                mapped_values = df[feature].map(ordinal_mappings[feature])
                df[feature] = mapped_values.fillna(0).astype(int)
        
        # One-hot encode nominal features
        nominal_features = [col for col in categorical_cols if col not in ordinal_features]
        
        for feature in nominal_features:
            if feature in df.columns:
                # Convert to string first to avoid categorical issues
                df[feature] = df[feature].astype(str)
                # Create dummy variables
                dummies = pd.get_dummies(df[feature], prefix=feature, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(feature, axis=1)
        
        return df
    
    def prepare_ml_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare data for ML models"""
        self.logger.info("Preparing data for ML models...")
        
        # Separate features and target
        target_col = self.config['data']['target_column']
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")
        
        X = df.drop([target_col, 'partner_id'], axis=1, errors='ignore')
        y = df[target_col]
        
        # Get feature names
        feature_names = X.columns.tolist()
        
        # Convert to numpy arrays
        X = X.values
        y = y.values
        
        self.logger.info(f"Features shape: {X.shape}")
        self.logger.info(f"Target shape: {y.shape}")
        
        return X, y, feature_names
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Split data into train, validation, and test sets"""
        self.logger.info("Splitting data...")
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=self.config['data']['test_size'],
            random_state=self.config['data']['random_state'],
            stratify=None  # For regression
        )
        
        # Second split: train vs val
        val_size = self.config['data']['validation_size'] / (1 - self.config['data']['test_size'])
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            random_state=self.config['data']['random_state']
        )
        
        self.logger.info(f"Train set: {X_train.shape}")
        self.logger.info(f"Validation set: {X_val.shape}")
        self.logger.info(f"Test set: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scale_features(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Scale features using StandardScaler"""
        self.logger.info("Scaling features...")
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Save scaler
        joblib.dump(scaler, 'models/scaler.pkl')
        self.logger.info("Scaler saved to models/scaler.pkl")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, scaler
    
    def get_protected_attributes(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract protected attributes for fairness analysis"""
        self.logger.info("Extracting protected attributes...")
        
        protected_attrs = {}
        for attr in self.config['fairness']['protected_attributes']:
            if attr in df.columns:
                # Encode categorical protected attributes
                le = LabelEncoder()
                protected_attrs[attr] = le.fit_transform(df[attr].astype(str))
                
                # Save label encoder
                joblib.dump(le, f'models/{attr}_label_encoder.pkl')
                self.logger.info(f"Extracted protected attribute: {attr}")
            else:
                self.logger.warning(f"Protected attribute '{attr}' not found in dataset")
        
        self.logger.info(f"Extracted {len(protected_attrs)} protected attributes")
        return protected_attrs
    
    def get_protected_attributes_before_encoding(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract protected attributes before one-hot encoding"""
        self.logger.info("Extracting protected attributes before encoding...")
        
        protected_attrs = {}
        for attr in self.config['fairness']['protected_attributes']:
            if attr in df.columns:
                # Convert to string and encode
                le = LabelEncoder()
                attr_values = df[attr].astype(str)
                protected_attrs[attr] = le.fit_transform(attr_values)
                
                # Save label encoder
                joblib.dump(le, f'models/{attr}_label_encoder.pkl')
                self.logger.info(f"Extracted protected attribute: {attr} ({len(le.classes_)} categories)")
            else:
                self.logger.warning(f"Protected attribute '{attr}' not found in dataset")
        
        return protected_attrs