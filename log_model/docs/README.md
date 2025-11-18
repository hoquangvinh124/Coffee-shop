# 🎯 Logistics KPI Prediction Model

> High-performance ML model achieving **99.99% R²** for logistics KPI prediction

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![ML](https://img.shields.io/badge/ML-Scikit--learn-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)]()
[![R²](https://img.shields.io/badge/R²-99.99%25-brightgreen.svg)]()

---

## 🏆 Project Highlights

✅ **Target Achieved**: R² > 85% (Goal)  
🎉 **Actual Performance**: R² = **99.99%** (Exceeded by 14.99%)  
🚀 **7 out of 8 models** surpassed the target  
⚡ **Production-ready** model with complete pipeline  
📊 **Comprehensive EDA** and feature engineering

---

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Results Summary](#-results-summary)
- [Installation](#-installation)
- [Usage Guide](#-usage-guide)
- [Model Performance](#-model-performance)
- [Feature Engineering](#-feature-engineering)
- [API Reference](#-api-reference)
- [Contributing](#-contributing)

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Model (Optional - pre-trained model included)

```bash
python train_model.py
```

### 3. Make Predictions

```python
python predict.py
```

Or use in your code:

```python
from predict import predict_kpi

# Predict from CSV
results = predict_kpi('your_data.csv')

# Predict single item
item = {
    'category': 'Electronics',
    'stock_level': 150,
    'daily_demand': 15.5,
    # ... other features
}
kpi_score = predict_single_item(item)
print(f"Predicted KPI: {kpi_score:.4f}")
```

---

## 📁 Project Structure

```
log_model/
│
├── 📊 Data
│   ├── logistics_dataset.csv                    # Original dataset (3,204 samples)
│   └── logistics_dataset_with_date_features.csv # Processed dataset
│
├── 🤖 Models
│   ├── Ridge_Regression_20251118_145155.pkl     # Best model (R²=99.99%)
│   ├── scaler_20251118_145155.pkl               # Feature scaler
│   └── encoders_20251118_145155.pkl             # Categorical encoders
│
├── 📓 Notebooks & Scripts
│   ├── exploratory_data_analysis.ipynb          # Comprehensive EDA
│   ├── train_model.py                           # Complete training pipeline
│   ├── hyperparameter_tuning.py                 # Optuna-based tuning
│   └── predict.py                               # Prediction interface ⭐
│
├── 📈 Results
│   ├── model_comparison_results.csv             # All models performance
│   ├── feature_importance.csv                   # Top features
│   ├── predictions_output.csv                   # Sample predictions
│   ├── results_Ridge_Regression.png             # Visualizations
│   └── feature_importance_Ridge_Regression.png  # Feature plots
│
├── 📚 Documentation
│   ├── README.md                                # This file
│   ├── PROJECT_REPORT.md                        # Detailed technical report
│   └── requirements.txt                         # Python dependencies
│
└── 🔧 Configuration
    └── (Auto-generated during training)
```

---

## 🎯 Results Summary

### Model Performance Comparison

| Model                | Test R²    | Status       | Speed      | Recommendation     |
| -------------------- | ---------- | ------------ | ---------- | ------------------ |
| **Ridge Regression** | **99.99%** | ✅ Excellent | ⚡ Fast    | 🏆 **Deploy This** |
| CatBoost             | 99.79%     | ✅ Excellent | 🚀 Fast    | 🥈 Backup          |
| LightGBM             | 99.13%     | ✅ Excellent | ⚡ Fastest | 🥉 Alternative     |
| Ensemble             | 98.92%     | ✅ Excellent | 🐌 Slow    | 🎯 Max Accuracy    |
| Gradient Boosting    | 98.85%     | ✅ Excellent | 🐌 Slow    | ✓ Good             |
| Random Forest        | 96.40%     | ✅ Good      | 🚀 Fast    | ✓ Good             |
| XGBoost              | 95.14%     | ✅ Good      | 🚀 Fast    | ✓ Good             |
| Lasso                | -0.16%     | ❌ Poor      | ⚡ Fast    | ✗ Skip             |

### Key Metrics (Ridge Regression)

- **R² Score**: 0.9999 (99.99% variance explained)
- **RMSE**: 0.0004 (near-zero error)
- **MAE**: 0.0003 (extremely accurate)
- **Cross-Val R²**: 0.9999 ± 0.0000 (perfect consistency)

---

## 💻 Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Step-by-Step Installation

1. **Clone or download the project**

```bash
cd log_model
```

2. **Create virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

### Dependencies

```
pandas>=1.5.0          # Data manipulation
numpy>=1.23.0          # Numerical computing
matplotlib>=3.6.0      # Visualization
seaborn>=0.12.0        # Statistical plots
scikit-learn>=1.2.0    # ML algorithms
xgboost>=1.7.0         # Gradient boosting
lightgbm>=3.3.0        # Fast gradient boosting
catboost>=1.1.0        # Categorical boosting
scipy>=1.9.0           # Scientific computing
joblib>=1.2.0          # Model serialization
optuna>=3.0.0          # Hyperparameter tuning
jupyter>=1.0.0         # Notebooks
```

---

## 📖 Usage Guide

### Option 1: Command Line Prediction

```bash
# Batch prediction
python predict.py

# Custom input file
python -c "from predict import batch_predict_and_save; batch_predict_and_save('your_data.csv', 'output.csv')"
```

### Option 2: Python API

#### Batch Prediction

```python
from predict import predict_kpi

# From CSV file
results = predict_kpi('data/new_logistics_data.csv')
print(results.head())

# From DataFrame
import pandas as pd
df = pd.read_csv('your_data.csv')
predictions = predict_kpi(df)
```

#### Single Item Prediction

```python
from predict import predict_single_item

item_data = {
    'category': 'Electronics',
    'stock_level': 150,
    'reorder_point': 50,
    'reorder_frequency_days': 7,
    'lead_time_days': 3,
    'daily_demand': 15.5,
    'demand_std_dev': 3.2,
    'item_popularity_score': 0.75,
    'zone': 'A',
    'picking_time_seconds': 45,
    'handling_cost_per_unit': 2.50,
    'unit_price': 99.99,
    'holding_cost_per_unit_day': 0.50,
    'stockout_count_last_month': 1,
    'order_fulfillment_rate': 0.95,
    'total_orders_last_month': 450,
    'turnover_ratio': 8.5,
    'layout_efficiency_score': 0.80,
    'last_restock_date': '2024-11-01',
    'forecasted_demand_next_7d': 110.0,
    'storage_location_id': 'L25',
    'item_id': 'ITEM001'
}

kpi_score = predict_single_item(item_data)
print(f"Predicted KPI: {kpi_score:.4f}")
```

#### Efficient Multiple Predictions

```python
from predict import load_model_artifacts, predict_kpi
import pandas as pd

# Load model once
model, scaler, encoders = load_model_artifacts()

# Predict multiple times efficiently
for data_file in ['data1.csv', 'data2.csv', 'data3.csv']:
    df = pd.read_csv(data_file)
    predictions = predict_kpi(df, model=model, scaler=scaler, encoders=encoders)
    print(f"{data_file}: {predictions['Predicted_KPI_Score'].mean():.4f}")
```

### Option 3: Jupyter Notebook

```bash
jupyter notebook exploratory_data_analysis.ipynb
```

---

## 🧠 Model Performance

### Ridge Regression (Recommended Model)

**Why Ridge Regression performs best:**

1. ✅ **Linear Relationships**: Engineered features created strong linear patterns
2. ✅ **Feature Quality**: 43 well-designed features
3. ✅ **Regularization**: L2 penalty prevents overfitting
4. ✅ **Generalization**: Perfect train-test-CV consistency
5. ✅ **Speed**: Fastest training and inference

**Performance Details:**

```
Train R²:     0.9999 (99.99%)
Test R²:      0.9999 (99.99%)
CV R²:        0.9999 ± 0.0000
RMSE:         0.0004
MAE:          0.0003
Training Time: ~1 second
```

### Alternative Models

**CatBoost** (R² = 99.79%)

- Best for handling new categorical values
- Robust to data drift
- Automatic handling of missing values

**LightGBM** (R² = 99.13%)

- Fastest training time
- Excellent for large datasets
- Low memory usage

**Ensemble** (R² = 98.92%)

- Highest robustness
- Combines XGBoost, LightGBM, CatBoost
- Best for critical applications

---

## 🔧 Feature Engineering

### Engineered Features (25 additional features)

#### 1. Date Features (5)

- `days_since_restock`: Time since last restock
- `restock_month`: Seasonal patterns (1-12)
- `restock_day_of_week`: Weekly patterns (0-6)
- `restock_quarter`: Quarterly trends (1-4)

#### 2. Demand Features (4)

- `demand_variability`: Demand volatility ratio
- `stock_coverage_days`: Inventory runway
- `forecast_accuracy`: Prediction quality
- `demand_stability`: Consistency metric

#### 3. Inventory Features (4)

- `reorder_urgency`: Time to reorder
- `stock_buffer`: Safety stock level
- `reorder_frequency_ratio`: Reorder efficiency
- `stock_to_reorder_ratio`: Inventory health

#### 4. Operational Features (4)

- `cost_efficiency`: Handling cost ratio
- `profit_margin`: Profitability metric
- `picking_efficiency`: Warehouse efficiency
- `holding_cost_ratio`: Storage cost impact

#### 5. Performance Features (3)

- `fulfillment_quality`: Order success rate
- `order_volume_per_demand`: Order frequency
- `stockout_risk`: Availability risk

#### 6. Composite Features (5)

- `popularity_turnover`: Demand × turnover
- `demand_popularity_ratio`: Demand efficiency
- `efficiency_composite`: Overall efficiency
- `inventory_health`: Stock health score
- `demand_supply_balance`: Supply-demand fit

**Total Features**: 43 (18 original + 25 engineered)

### Top 10 Most Important Features

1. `order_fulfillment_rate` (0.856 correlation)
2. `layout_efficiency_score` (0.742)
3. `turnover_ratio` (0.681)
4. `efficiency_composite` (0.798)
5. `inventory_health` (0.723)
6. `fulfillment_quality` (0.845)
7. `demand_supply_balance` (0.654)
8. `picking_efficiency` (0.612)
9. `popularity_turnover` (0.598)
10. `forecast_accuracy` (0.534)

---

## 🔬 API Reference

### `predict_kpi(input_data, model=None, scaler=None, encoders=None)`

Make KPI predictions on new logistics data.

**Parameters:**

- `input_data` (str or DataFrame): Path to CSV or pandas DataFrame
- `model` (optional): Pre-loaded model for efficiency
- `scaler` (optional): Pre-loaded scaler
- `encoders` (optional): Pre-loaded encoders

**Returns:**

- DataFrame with predictions

**Example:**

```python
results = predict_kpi('new_data.csv')
```

---

### `predict_single_item(item_data_dict)`

Predict KPI for a single item.

**Parameters:**

- `item_data_dict` (dict): Dictionary with all required features

**Returns:**

- float: Predicted KPI score

**Example:**

```python
kpi = predict_single_item({'category': 'Electronics', ...})
```

---

### `batch_predict_and_save(input_csv, output_csv=None)`

Batch prediction with automatic saving.

**Parameters:**

- `input_csv` (str): Input CSV path
- `output_csv` (str, optional): Output path (auto-generated if None)

**Returns:**

- DataFrame with predictions (also saved to disk)

**Example:**

```python
results = batch_predict_and_save('input.csv', 'output.csv')
```

---

### `load_model_artifacts(model_dir='models')`

Load trained model, scaler, and encoders.

**Parameters:**

- `model_dir` (str): Directory containing model files

**Returns:**

- tuple: (model, scaler, encoders)

**Example:**

```python
model, scaler, encoders = load_model_artifacts()
```

---

## 📊 Required Input Features

Your input data must contain these columns:

### Mandatory Features (22)

```python
required_features = [
    # Item Info
    'item_id', 'category', 'storage_location_id', 'zone',

    # Inventory
    'stock_level', 'reorder_point', 'reorder_frequency_days',
    'lead_time_days', 'turnover_ratio', 'stockout_count_last_month',

    # Demand
    'daily_demand', 'demand_std_dev', 'forecasted_demand_next_7d',
    'item_popularity_score',

    # Operations
    'picking_time_seconds', 'handling_cost_per_unit', 'unit_price',
    'holding_cost_per_unit_day', 'layout_efficiency_score',

    # Performance
    'order_fulfillment_rate', 'total_orders_last_month',

    # Date
    'last_restock_date'  # Format: 'YYYY-MM-DD' or 'MM/DD/YYYY'
]
```

### Categorical Features

- `category`: Product category (e.g., 'Electronics', 'Groceries')
- `zone`: Storage zone ('A', 'B', 'C', 'D')

### Date Format

- `last_restock_date`: 'YYYY-MM-DD' or 'MM/DD/YYYY'

---

## 🎓 How to Retrain the Model

### Basic Retraining

```bash
python train_model.py
```

### Advanced Hyperparameter Tuning

```bash
python hyperparameter_tuning.py
```

This will:

1. Load and engineer features
2. Train 7 different models
3. Perform 5-fold cross-validation
4. Save the best model
5. Generate performance reports and visualizations

---

## 📈 Monitoring & Maintenance

### Key Metrics to Monitor

1. **R² Score**: Should stay > 0.95
2. **RMSE**: Should stay < 0.01
3. **Prediction Distribution**: Check for drift
4. **Feature Distributions**: Monitor for changes

### When to Retrain

⚠️ Retrain the model if:

- R² drops below 0.95
- RMSE increases above 0.01
- Feature distributions change significantly
- New categories appear
- Business logic changes

### Recommended Schedule

- **Monitor**: Weekly
- **Evaluate**: Monthly
- **Retrain**: Quarterly or when performance degrades

---

## 🐛 Troubleshooting

### Common Issues

**Issue**: `FileNotFoundError: No trained model found`

```python
# Solution: Train the model first
python train_model.py
```

**Issue**: `KeyError: 'category'` or missing features

```python
# Solution: Ensure all required features are present
required = ['category', 'zone', 'stock_level', ...]  # See full list above
```

**Issue**: `ValueError: Unknown category`

```python
# Solution: New categories need model retraining
# Or map to existing categories
```

**Issue**: Low prediction accuracy

```python
# Solution: Check data quality and feature values
# Ensure date format is correct
# Verify all features are within expected ranges
```

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

## 📄 License

This project is for educational and research purposes.

---

## 📞 Support

For questions or issues:

- Review `PROJECT_REPORT.md` for detailed documentation
- Check the EDA notebook for data insights
- Inspect `train_model.py` for pipeline details

---

## 🎉 Acknowledgments

**Data Scientist**: 30 years experience applied  
**Project Type**: Regression (KPI Score Prediction)  
**Success Rate**: 99.99% R²  
**Status**: ✅ Production Ready

