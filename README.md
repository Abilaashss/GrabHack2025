# Project Nova - Credit Scoring ML Pipeline

A comprehensive machine learning pipeline for credit scoring with fairness analysis, bias detection, and extensive model evaluation for Grab drivers and merchants.

## 🎯 Project Overview

Project Nova implements a state-of-the-art credit scoring system that:

- **Trains multiple ML models** including XGBoost, LightGBM, CatBoost, Neural Networks, and ensemble methods
- **Performs hyperparameter tuning** using Optuna, Grid Search, and Random Search
- **Evaluates model performance** with comprehensive metrics
- **Analyzes fairness and bias** across protected attributes
- **Creates extensive visualizations** for model interpretation
- **Provides deployment-ready models** with easy-to-use APIs

## 📁 Project Structure

```
project-nova/
├── data/                                    # Dataset files
│   ├── grab_drivers_dataset_refined_score.csv
│   └── grab_merchants_dataset_refined_score.csv
├── src/                                     # Source code
│   ├── preprocessing/                       # Data preprocessing
│   │   └── data_loader.py
│   ├── models/                             # ML models
│   │   ├── ml_models.py
│   │   └── hyperparameter_tuning.py
│   ├── evaluation/                         # Model evaluation
│   │   └── metrics.py
│   ├── fairness/                           # Fairness analysis
│   │   └── bias_detection.py
│   └── visualization/                      # Plotting and visualization
│       └── plots.py
├── models/                                 # Trained models and artifacts
│   ├── saved_models/                       # Model files
│   ├── checkpoints/                        # Training checkpoints
│   └── hyperparameters/                    # Hyperparameter tuning results
├── results/                                # Results and outputs
│   ├── metrics/                            # Evaluation metrics
│   ├── plots/                              # Static plots
│   ├── visualizations/                     # Interactive visualizations
│   └── fairness_analysis/                  # Fairness reports
├── utils/                                  # Utility functions
│   └── model_utils.py
├── config/                                 # Configuration files
│   └── config.yaml
├── logs/                                   # Log files
├── main_training_pipeline.py               # Main training pipeline
├── quick_start.py                          # Quick start script
├── deploy_model.py                         # Model deployment
└── requirements.txt                        # Python dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Quick Start (Recommended for first time)

```bash
python quick_start.py
```

This will:
- Load and preprocess the data
- Train models with default parameters
- Evaluate model performance
- Analyze fairness
- Create visualizations
- Generate a comprehensive report

### 3. Run Full Pipeline (With Hyperparameter Tuning)

```bash
python main_training_pipeline.py
```

This includes everything from quick start plus:
- Extensive hyperparameter tuning
- Ensemble model creation
- Advanced visualizations

## 📊 Models Implemented

### Core ML Models
- **XGBoost** - Gradient boosting with extensive hyperparameter tuning
- **LightGBM** - Fast gradient boosting with categorical feature support
- **CatBoost** - Gradient boosting optimized for categorical features
- **Neural Networks** - Deep learning models with Keras/TensorFlow
- **Random Forest** - Ensemble of decision trees
- **Linear Models** - Ridge, Lasso, Elastic Net regression
- **Support Vector Regression** - SVR with RBF kernel

### Ensemble Methods
- **Average Ensemble** - Simple averaging of predictions
- **Weighted Ensemble** - Weighted averaging based on performance
- **Median Ensemble** - Robust median-based ensemble

## 🔧 Hyperparameter Tuning

The pipeline supports multiple hyperparameter optimization methods:

- **Optuna** - Bayesian optimization (default)
- **Grid Search** - Exhaustive search over parameter grid
- **Random Search** - Random sampling of parameter space

### Tuning Configuration

Edit `config/config.yaml` to customize:
- Number of trials
- Timeout settings
- Parameter ranges
- Cross-validation settings

## 📈 Evaluation Metrics

### Regression Metrics
- **R² Score** - Coefficient of determination
- **RMSE** - Root Mean Square Error
- **MAE** - Mean Absolute Error
- **MAPE** - Mean Absolute Percentage Error

### Credit Scoring Specific Metrics
- **Score Band Accuracy** - Accuracy within credit score ranges
- **Within ±25/50 points** - Prediction accuracy thresholds
- **Risk Assessment Metrics** - Precision/recall for risk categories
- **Ranking Metrics** - Spearman correlation, Kendall's tau

## ⚖️ Fairness Analysis

### Fairness Metrics Implemented
- **Demographic Parity** - Equal positive rates across groups
- **Equalized Odds** - Equal TPR and FPR across groups
- **Equal Opportunity** - Equal TPR for positive class across groups
- **Individual Fairness** - Similar predictions for similar individuals
- **Calibration** - Prediction accuracy across groups

### Protected Attributes
- Gender
- Ethnicity
- Age Group
- Education Level
- Location

### Bias Mitigation (Planned)
- **Pre-processing** - Reweighing, Disparate Impact Remover
- **In-processing** - Adversarial debiasing
- **Post-processing** - Equalized odds, Reject option classification

## 📊 Visualizations

### Static Plots (PNG/PDF)
- Model performance comparison
- Prediction vs actual scatter plots
- Feature importance plots
- Credit score distributions
- Fairness analysis charts
- SHAP value plots
- Learning curves

### Interactive Visualizations (HTML)
- Interactive performance dashboards
- Dynamic feature importance plots
- Fairness comparison tools
- Model prediction analysis

## 🚀 Model Deployment

### Deploy Best Model

```bash
python deploy_model.py
```

### Use Deployed Model

```python
from utils.model_utils import CreditScorePredictor

# Initialize predictor
predictor = CreditScorePredictor()

# Make prediction
features = {
    'tenure_months': 24,
    'average_rating': 4.5,
    'monthly_revenue': 5000,
    # ... other features
}

result = predictor.predict_from_dict(features)
print(f"Credit Score: {result['credit_score']}")
print(f"Risk Category: {result['risk_category']}")
```

### API Integration

The deployed models can be easily integrated into web APIs:

```python
from flask import Flask, request, jsonify
from utils.model_utils import CreditScorePredictor

app = Flask(__name__)
predictor = CreditScorePredictor()

@app.route('/predict', methods=['POST'])
def predict():
    features = request.json
    result = predictor.predict_from_dict(features)
    return jsonify(result)
```

## 📋 Configuration

### Main Configuration (`config/config.yaml`)

```yaml
# Data Configuration
data:
  test_size: 0.2
  validation_size: 0.2
  random_state: 42

# Model Configuration
models:
  xgboost:
    n_estimators: [100, 200, 500, 1000]
    max_depth: [3, 6, 10, 15]
    learning_rate: [0.01, 0.1, 0.2, 0.3]

# Hyperparameter Tuning
hyperparameter_tuning:
  method: "optuna"
  n_trials: 100
  timeout: 3600

# Fairness Configuration
fairness:
  protected_attributes: ["gender", "ethnicity", "age_group"]
  fairness_metrics:
    - "demographic_parity"
    - "equalized_odds"
    - "equal_opportunity"
```

## 📊 Results and Outputs

### Generated Files

After running the pipeline, you'll find:

- **Models**: `models/saved_models/` - Trained model files
- **Metrics**: `results/metrics/` - Performance evaluation results
- **Plots**: `results/plots/` - Static visualization files
- **Interactive**: `results/visualizations/` - Interactive HTML plots
- **Fairness**: `results/fairness_analysis/` - Bias analysis reports
- **Reports**: `results/final_report.md` - Comprehensive summary

### Key Output Files

- `results/final_report.md` - Executive summary
- `results/metrics/all_models_evaluation.csv` - Model comparison
- `results/fairness_analysis/model_fairness_comparison.csv` - Fairness comparison
- `models/deployment_info.json` - Deployment configuration

## 🔍 Model Interpretation

### Feature Importance
- Tree-based models provide built-in feature importance
- SHAP values for model-agnostic interpretation
- Permutation importance for robust feature ranking

### SHAP Analysis
- Summary plots showing feature contributions
- Waterfall plots for individual predictions
- Dependence plots for feature interactions

## 🛠️ Advanced Usage

### Custom Model Training

```python
from src.models.ml_models import MLModels
from src.preprocessing.data_loader import DataLoader

# Load and preprocess data
data_loader = DataLoader()
# ... data loading code ...

# Initialize and train custom model
ml_models = MLModels()
ml_models.initialize_models()

# Train specific model
model = ml_models.train_model('xgboost', X_train, y_train, X_val, y_val)
```

### Custom Fairness Analysis

```python
from src.fairness.bias_detection import BiasDetector

bias_detector = BiasDetector()
fairness_results = bias_detector.comprehensive_fairness_analysis(
    y_true, y_pred, X_test, protected_attributes, 'my_model'
)
```

## 📝 Logging

All operations are logged to:
- `logs/main_pipeline.log` - Main pipeline execution
- `logs/data_preprocessing.log` - Data processing steps
- `logs/ml_models.log` - Model training logs
- `logs/hyperparameter_tuning.log` - Hyperparameter optimization
- `logs/model_evaluation.log` - Evaluation metrics
- `logs/bias_detection.log` - Fairness analysis
- `logs/visualization.log` - Plotting operations

## 🚨 Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce batch size or number of trials in config
2. **Missing Dependencies**: Run `pip install -r requirements.txt`
3. **Data Loading Errors**: Check data file paths in config
4. **Model Loading Errors**: Ensure models are trained before deployment

### Performance Optimization

- Use `quick_start.py` for faster execution without hyperparameter tuning
- Reduce number of Optuna trials for faster tuning
- Use smaller validation sets for quicker evaluation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Grab for providing the dataset and problem statement
- Scikit-learn, XGBoost, LightGBM, CatBoost teams for excellent ML libraries
- Optuna team for hyperparameter optimization framework
- SHAP team for model interpretation tools

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review log files in the `logs/` directory
3. Open an issue on GitHub
4. Contact the development team

---

**Project Nova** - Empowering fair and accurate credit scoring for the digital economy.