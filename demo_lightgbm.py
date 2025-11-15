"""
Demo LightGBM Predictions - Show what it predicts
"""
import sys
sys.path.insert(0, 'src')
import pandas as pd
import numpy as np
from models.ml_models import MLForecaster

# Load data
print("="*70)
print(" LIGHTGBM PREDICTION DEMO")
print("="*70)

# Load test data
X = pd.read_csv('data/processed/X.csv', index_col='date', parse_dates=True)
y = pd.read_csv('data/processed/y.csv', index_col='date', parse_dates=True).squeeze()

# Split
n = len(X)
train_size = int(n * 0.8)
X_train = X.iloc[:train_size]
y_train = y.iloc[:train_size]
X_test = X.iloc[train_size:]
y_test = y.iloc[train_size:]

# Load trained model
print("\n[1] Loading trained LightGBM model...")
forecaster = MLForecaster('lightgbm')
forecaster.load_model('models/lightgbm_model.pkl')

# Make predictions
print("\n[2] Making predictions on test set...")
y_pred = forecaster.predict(X_test)

# Show predictions vs actual
print("\n" + "="*70)
print("LIGHTGBM PREDICTIONS vs ACTUAL REVENUE")
print("="*70)
print(f"\n{'Date':<12} {'Actual ($)':<15} {'Predicted ($)':<15} {'Error ($)':<15} {'Error %':<10}")
print("-"*70)

for i in range(len(y_test)):
    date = y_test.index[i].strftime('%Y-%m-%d')
    actual = y_test.iloc[i]
    predicted = y_pred[i]
    error = predicted - actual
    error_pct = (error / actual) * 100

    print(f"{date:<12} ${actual:>8,.2f}      ${predicted:>8,.2f}      ${error:>8,.2f}      {error_pct:>6.1f}%")

# Summary statistics
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)
print(f"Average Actual Revenue:    ${y_test.mean():,.2f}")
print(f"Average Predicted Revenue: ${y_pred.mean():,.2f}")
print(f"Total Actual Revenue:      ${y_test.sum():,.2f}")
print(f"Total Predicted Revenue:   ${y_pred.sum():,.2f}")

errors = y_pred - y_test.values
print(f"\nMean Error (bias):         ${errors.mean():,.2f}")
print(f"Mean Absolute Error:       ${np.abs(errors).mean():,.2f}")
print(f"Root Mean Squared Error:   ${np.sqrt((errors**2).mean()):,.2f}")

# Example: What features matter most for a specific prediction?
print("\n" + "="*70)
print("TOP 10 MOST IMPORTANT FEATURES (Overall)")
print("="*70)
top_features = forecaster.get_feature_importance(top_n=10)
for i, (feat, imp) in enumerate(top_features.items(), 1):
    print(f"{i:2d}. {feat:<40s}: {imp:.2f}")

# Show example input for one prediction
print("\n" + "="*70)
print(f"EXAMPLE: Predicting {X_test.index[0].strftime('%Y-%m-%d')}")
print("="*70)
print("\nTop 10 feature values used for this prediction:")
sample_features = X_test.iloc[0][top_features.index]
for feat, value in sample_features.items():
    print(f"  {feat:<40s}: {value:>10.2f}")

predicted_value = y_pred[0]
actual_value = y_test.iloc[0]
print(f"\nLightGBM Prediction: ${predicted_value:,.2f}")
print(f"Actual Revenue:      ${actual_value:,.2f}")
print(f"Error:               ${predicted_value - actual_value:,.2f} ({((predicted_value - actual_value)/actual_value)*100:.1f}%)")

print("\n" + "="*70)
print(" LightGBM HOW IT WORKS")
print("="*70)
print("""
LightGBM uses Gradient Boosting Decision Trees:

1. Bắt đầu với một prediction đơn giản (trung bình revenue)
2. Tạo một decision tree nhỏ để predict error của prediction trước
3. Add tree này vào model (học từ mistakes)
4. Repeat bước 2-3 nhiều lần (100 trees trong model này)
5. Final prediction = tổng của tất cả trees

Ví dụ đơn giản:
- Tree 1: "Nếu doanh thu hôm qua > $4000 → predict $4500"
- Tree 2: "Nếu là thứ 2 và doanh thu tăng 7 ngày qua → add thêm $300"
- Tree 3: "Nếu rolling mean 7 ngày thấp → trừ $200"
- ...
- Final = Tree1 + Tree2 + Tree3 + ... + Tree100

Key Features được LightGBM sử dụng nhiều nhất:
1. revenue_change_1d: Thay đổi doanh thu so với hôm qua
2. revenue_pct_change_1d: % thay đổi so với hôm qua
3. revenue_lag_7: Doanh thu tuần trước (same day)
4. revenue_change_7d: Thay đổi so với tuần trước
5. rolling_mean_7_x_dayofweek: Trung bình 7 ngày × ngày trong tuần
""")

print("\n✓ Demo complete!")
