# Project Nova - Separate Scripts Guide

## 🎯 **Problem Solved**

You now have **separate, independent scripts** for each pipeline stage, so you can:
- ✅ **Avoid redundant training** when models are already trained
- ✅ **Run evaluation independently** after training
- ✅ **Run fairness analysis separately** 
- ✅ **Create visualizations independently**
- ✅ **Control each stage individually**

## 📋 **Available Scripts**

### **1. `train_models_only.py`**
**Purpose**: Train models only, no evaluation or analysis
```bash
python train_models_only.py
```
**What it does**:
- Loads and preprocesses data
- Trains all ML models for drivers and merchants
- Saves models to `models/saved_models/`
- **Does NOT** run evaluation, fairness, or visualization

### **2. `evaluate_models.py`** ✅ **WORKING**
**Purpose**: Evaluate already trained models
```bash
python evaluate_models.py
```
**What it does**:
- Loads trained models from disk
- Loads test data
- Evaluates all models
- Saves results to `results/metrics/`
- **Requires**: Trained models in `models/saved_models/`

### **3. `analyze_fairness.py`**
**Purpose**: Analyze fairness of trained models
```bash
python analyze_fairness.py
```
**What it does**:
- Loads trained models from disk
- Loads test data with protected attributes
- Performs comprehensive fairness analysis
- Saves reports to `results/fairness_analysis/`
- **Requires**: Trained models in `models/saved_models/`

### **4. `create_visualizations.py`**
**Purpose**: Create visualizations for trained models
```bash
python create_visualizations.py
```
**What it does**:
- Loads trained models from disk
- Loads test data
- Creates performance plots, prediction plots, feature importance
- Saves to `results/plots/` and `results/visualizations/`
- **Requires**: Trained models in `models/saved_models/`

### **5. `pipeline_controller.py`**
**Purpose**: Interactive menu to control all stages
```bash
python pipeline_controller.py
```
**What it does**:
- Shows interactive menu
- Checks pipeline status
- Runs individual stages
- Manages prerequisites automatically

## 🚀 **Current Status**

Based on your recent run:

### ✅ **Models Trained**
- **8 Drivers Models**: gradient_boosting, elastic_net, linear_regression, random_forest, lasso, ridge, mlp_regressor, svr
- **8 Merchants Models**: linear_regression, elastic_net, mlp_regressor, gradient_boosting, random_forest, ridge, lasso, svr

### ✅ **Evaluation Completed**
- **Best Drivers Model**: `drivers_mlp_regressor` (R² = 1.0000)
- **Best Merchants Model**: `merchants_mlp_regressor` (R² = 1.0000)
- **Results Saved**: 
  - `results/metrics/drivers_models_evaluation.csv`
  - `results/metrics/merchants_models_evaluation.csv`

### 🔄 **Next Steps Available**
- **Fairness Analysis**: `python analyze_fairness.py`
- **Visualizations**: `python create_visualizations.py`

## 📊 **Usage Examples**

### **Scenario 1: You have trained models, want evaluation only**
```bash
python evaluate_models.py
```

### **Scenario 2: You want fairness analysis after training**
```bash
python analyze_fairness.py
```

### **Scenario 3: You want to create plots**
```bash
python create_visualizations.py
```

### **Scenario 4: Interactive control**
```bash
python pipeline_controller.py
# Choose option 2, 3, or 4 from menu
```

### **Scenario 5: Start fresh with training only**
```bash
python train_models_only.py
```

## 🔧 **Error Fix Applied**

The **JSON serialization error** has been fixed in `src/evaluation/metrics.py`:
- ✅ Added `_convert_numpy_types()` method
- ✅ Converts numpy int64/float64 to native Python types
- ✅ All evaluation scripts now work without JSON errors

## 📁 **File Structure**

```
project-nova/
├── train_models_only.py          # Training only
├── evaluate_models.py             # Evaluation only ✅ WORKING
├── analyze_fairness.py            # Fairness only
├── create_visualizations.py       # Visualization only
├── pipeline_controller.py         # Interactive controller
├── main_training_pipeline.py      # Full pipeline (original)
├── quick_start.py                 # Quick test
└── models/saved_models/           # Your trained models ✅
    ├── drivers_mlp_regressor.pkl
    ├── merchants_mlp_regressor.pkl
    └── ... (14 more models)
```

## 🎯 **Benefits**

1. **No Redundant Training**: Run evaluation/analysis without retraining
2. **Faster Iteration**: Test different analysis without waiting for training
3. **Modular Development**: Work on specific pipeline stages independently
4. **Error Recovery**: If one stage fails, others still work
5. **Resource Efficiency**: Don't waste compute on unnecessary steps

## 🚀 **Ready to Use!**

Your models are trained and evaluation is complete. You can now run:

```bash
# For fairness analysis
python analyze_fairness.py

# For visualizations  
python create_visualizations.py

# For interactive control
python pipeline_controller.py
```

**All scripts are independent and won't retrain your models!** 🎉