"""
LSTM Future Prediction Script
Predict revenue for future dates using trained LSTM model
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import pickle
import sys
from datetime import datetime, timedelta

def load_lstm_model():
    """Load trained LSTM model and scaler"""
    model = keras.models.load_model('models/lstm_model.keras')
    with open('models/lstm_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

def predict_future(model, scaler, historical_data, target_date, lookback=30):
    """
    Predict revenue for a future date using LSTM

    Args:
        model: Trained LSTM model
        scaler: MinMaxScaler used during training
        historical_data: DataFrame with date and revenue columns
        target_date: Date to predict (datetime or string)
        lookback: Number of previous days to use (default: 30)
    """
    target_date = pd.to_datetime(target_date)

    # Get last date in historical data
    last_date = historical_data['date'].max()

    # Calculate how many days to predict ahead
    days_ahead = (target_date - last_date).days

    if days_ahead < 0:
        # Target date is in the past - use actual historical data
        lookback_start = target_date - timedelta(days=lookback)
        historical_window = historical_data[
            (historical_data['date'] >= lookback_start) &
            (historical_data['date'] < target_date)
        ]

        if len(historical_window) < lookback:
            print(f"⚠️  Not enough historical data. Need {lookback} days before {target_date}")
            return None

        # Use actual historical values
        sequence = historical_window['revenue'].values[-lookback:]
        sequence_scaled = scaler.transform(sequence.reshape(-1, 1))

        # Predict
        X = sequence_scaled.reshape(1, lookback, 1)
        prediction_scaled = model.predict(X, verbose=0)
        prediction = scaler.inverse_transform(prediction_scaled)[0, 0]

        return {
            'date': target_date,
            'prediction': prediction,
            'method': 'historical_lookback',
            'days_ahead': 0
        }

    else:
        # Target date is in the future - need to predict iteratively
        # Start with last 30 days of historical data
        sequence = historical_data['revenue'].values[-lookback:]
        sequence_scaled = scaler.transform(sequence.reshape(-1, 1)).flatten()

        predictions = []
        current_date = last_date

        # Predict day by day until target date
        for i in range(days_ahead):
            # Prepare input
            X = sequence_scaled[-lookback:].reshape(1, lookback, 1)

            # Predict next day
            prediction_scaled = model.predict(X, verbose=0)[0, 0]
            predictions.append(prediction_scaled)

            # Add prediction to sequence for next iteration
            sequence_scaled = np.append(sequence_scaled, prediction_scaled)

            current_date += timedelta(days=1)

        # Inverse transform final prediction
        final_prediction_scaled = predictions[-1]
        final_prediction = scaler.inverse_transform([[final_prediction_scaled]])[0, 0]

        return {
            'date': target_date,
            'prediction': final_prediction,
            'method': 'iterative_forecast',
            'days_ahead': days_ahead,
            'all_predictions': scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        }

def main():
    if len(sys.argv) < 2:
        print("=" * 70)
        print("🔮 LSTM REVENUE PREDICTION - COFFEE SHOP")
        print("=" * 70)
        print("\nCách sử dụng:")
        print("  python predict_lstm.py 2023-07-15")
        print("  python predict_lstm.py 2024-01-01")
        print("  python predict_lstm.py 2025-01-01")
        print("  python predict_lstm.py 2026-01-01")
        print("\nLSTM có thể dự đoán cho BẤT KỲ ngày tương lai nào!")
        print()

        # Show data range
        daily_revenue = pd.read_csv('data/processed/daily_revenue.csv')
        daily_revenue['date'] = pd.to_datetime(daily_revenue['date'])
        print(f"📊 Dữ liệu training: {daily_revenue['date'].min().strftime('%Y-%m-%d')} "
              f"đến {daily_revenue['date'].max().strftime('%Y-%m-%d')}")
        print(f"💡 LSTM sẽ dự đoán iteratively cho các ngày trong tương lai")
        print()
        sys.exit(1)

    target_date_str = sys.argv[1]

    print("=" * 70)
    print("🔮 LSTM REVENUE PREDICTION")
    print("=" * 70)
    print()

    # Load model and data
    print("🔄 Loading LSTM model...")
    model, scaler = load_lstm_model()
    print("✓ Model loaded")

    print("📊 Loading historical data...")
    daily_revenue = pd.read_csv('data/processed/daily_revenue.csv')
    daily_revenue['date'] = pd.to_datetime(daily_revenue['date'])
    daily_revenue = daily_revenue.sort_values('date').reset_index(drop=True)
    print(f"✓ Loaded {len(daily_revenue)} days of data")

    # Predict
    print(f"🔮 Predicting for {target_date_str}...")
    result = predict_future(model, scaler, daily_revenue, target_date_str)

    if result:
        print()
        print("=" * 70)
        print(f"📅 Ngày: {result['date'].strftime('%A, %Y-%m-%d')}")
        print(f"💰 Doanh thu dự đoán (LSTM): ${result['prediction']:,.2f}")

        if result['days_ahead'] > 0:
            print(f"📆 Dự đoán {result['days_ahead']} ngày trước (iterative forecasting)")

        # Check if we have actual data
        actual_row = daily_revenue[daily_revenue['date'] == result['date']]
        if not actual_row.empty:
            actual = actual_row['revenue'].values[0]
            error = abs(result['prediction'] - actual)
            mape = (error / actual) * 100

            print()
            print(f"✅ Doanh thu thực tế: ${actual:,.2f}")
            print(f"📊 Sai số: ${error:,.2f} ({mape:.2f}%)")

            if mape < 5:
                print("🎯 Dự đoán RẤT CHÍNH XÁC!")
            elif mape < 10:
                print("👍 Dự đoán TỐT!")
            elif mape < 15:
                print("✓ Dự đoán CHẤP NHẬN ĐƯỢC")
            else:
                print("⚠️  Dự đoán chưa chính xác lắm")
        else:
            print()
            print("ℹ️  Ngày trong tương lai - không có data thực tế")
            if result['days_ahead'] > 180:
                print("⚠️  Dự đoán quá xa có thể không chính xác do error accumulation")

        print("=" * 70)
        print()

if __name__ == "__main__":
    main()
