"""
Compare ML Regression vs LSTM for Future Predictions
"""

import subprocess
import re
import pandas as pd

print("=" * 80)
print("MODEL COMPARISON: ML REGRESSION vs LSTM")
print("=" * 80)

test_dates = [
    ("2023-07-15", 15, "Short-term"),
    ("2023-08-01", 32, "Short-term"),
    ("2023-12-25", 178, "Medium-term"),
    ("2024-01-01", 185, "Medium-term"),
    ("2024-06-15", 351, "Long-term"),
]

print("\nTraining data ends: 2023-06-30")
print(f"Testing {len(test_dates)} dates\n")

print("-" * 80)
print(f"{'Date':<15} {'Days':<10} {'Range':<15} {'ML Regression':<20} {'LSTM':<20}")
print("-" * 80)

for date, days, range_type in test_dates:
    # Get ML Regression prediction
    result_ml = subprocess.run(
        ["python", "predict_future.py", date],
        capture_output=True,
        text=True
    )

    ml_pred = None
    for line in result_ml.stdout.split('\n'):
        if '💰' in line or 'Doanh thu dự đoán' in line:
            match = re.search(r'\$([0-9,]+\.\d{2})', line)
            if match:
                ml_pred = match.group(1)
                break

    # Get LSTM prediction
    result_lstm = subprocess.run(
        ["python", "predict_lstm.py", date],
        capture_output=True,
        text=True,
        timeout=60
    )

    lstm_pred = None
    for line in result_lstm.stdout.split('\n'):
        if 'Doanh thu dự đoán (LSTM)' in line:
            match = re.search(r'\$([0-9,]+\.\d{2})', line)
            if match:
                lstm_pred = match.group(1)
                break

    if ml_pred and lstm_pred:
        print(f"{date:<15} {days:<10} {range_type:<15} ${ml_pred:<19} ${lstm_pred:<20}")
    else:
        status_ml = ml_pred if ml_pred else "FAILED"
        status_lstm = lstm_pred if lstm_pred else "FAILED"
        print(f"{date:<15} {days:<10} {range_type:<15} {status_ml:<20} {status_lstm:<20}")

print("-" * 80)

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

print("""
📊 ML REGRESSION (Random Forest/LightGBM):
   ✅ Advantages:
      - Stable predictions even for distant future
      - No error accumulation
      - Fast inference
      - Works with synthetic future data

   ❌ Disadvantages:
      - Uses rolling average for future dates (same value)
      - Predictions converge to a constant for far future
      - Not true time series forecasting
      - Cannot capture new trends

📊 LSTM (Recurrent Neural Network):
   ✅ Advantages:
      - Learns temporal patterns
      - True time series forecasting approach
      - Can capture sequential dependencies
      - Good for short-term predictions

   ❌ Disadvantages:
      - ERROR ACCUMULATION for long-term predictions
      - Predictions can EXPLODE for distant future
      - Slower inference (iterative)
      - Requires more data for training
      - Numerical instability

💡 RECOMMENDATION:

   For SHORT-TERM (< 1 month):
      ✅ LSTM or ML Regression - both work

   For MEDIUM-TERM (1-6 months):
      ✅ ML Regression (more stable)
      ⚠️  LSTM (may accumulate errors)

   For LONG-TERM (> 6 months):
      ✅ ML Regression (stable but constant)
      ❌ LSTM (will EXPLODE - DO NOT USE)

   BEST APPROACH:
      1. Use HYBRID: LSTM for next 7-30 days, ML for beyond
      2. Or use PROPER time series: SARIMA, Prophet
      3. Or RETRAIN regularly with new data
""")

print("=" * 80)
