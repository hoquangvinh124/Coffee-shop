"""
Simple Prediction Script - Predict revenue for a specific date
Usage: python predict_date.py 2023-06-15
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from features.feature_engineering import TimeSeriesFeatureEngine

def train_model():
    """Train and return the LightGBM model"""
    print("🔄 Loading data...")
    X = pd.read_csv('data/processed/X.csv', index_col=0)
    y = pd.read_csv('data/processed/y.csv', index_col=0)

    print("🔄 Training model...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y['revenue'], test_size=0.1, random_state=42, shuffle=True
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1, random_state=42, shuffle=True
    )

    model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=31,
        random_state=42,
        verbose=-1
    )

    model.fit(X_train, y_train)
    print("✅ Model trained successfully!\n")

    return model

def predict_for_date(model, target_date_str, historical_data):
    """Predict revenue for a given date"""
    try:
        # Parse date
        target_date = pd.to_datetime(target_date_str)

        # Check if date is too old (need historical data)
        min_date = historical_data['date'].min() + timedelta(days=28)
        max_date = historical_data['date'].max()

        if target_date < min_date:
            print(f"⚠️  Ngày quá xa trong quá khứ! Cần ít nhất 28 ngày dữ liệu trước đó.")
            print(f"   Vui lòng chọn ngày từ {min_date.strftime('%Y-%m-%d')} trở đi")
            return None

        # Create features for target date
        feature_engine = TimeSeriesFeatureEngine()

        # Keep only date and revenue columns
        hist_df = historical_data[['date', 'revenue']].copy()

        # Add target date as a row with NaN revenue
        temp_df = pd.concat([
            hist_df,
            pd.DataFrame({'date': [target_date], 'revenue': [np.nan]})
        ]).sort_values('date').reset_index(drop=True)

        # Set date as index (required by feature_engineering)
        temp_df = temp_df.set_index('date')

        # Create features
        features_df = feature_engine.create_all_features(temp_df)

        # Get features for target date (date is now in index)
        target_row = features_df[features_df.index == target_date]

        if target_row.empty:
            print("❌ Không thể tạo features cho ngày này")
            return None

        # Drop revenue column for prediction (date is already index)
        X_predict = target_row.drop(['revenue'], axis=1, errors='ignore')

        # Debug: check for NaN values
        nan_count = X_predict.isna().sum().sum()
        if nan_count > 0:
            print(f"⚠️  Warning: {nan_count} NaN values in features")
            # Fill NaN with 0 (same as training)
            X_predict = X_predict.fillna(0)

        # Predict
        prediction = model.predict(X_predict)[0]

        return prediction, target_row

    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    if len(sys.argv) < 2:
        print("=" * 60)
        print("🔮 COFFEE SHOP REVENUE PREDICTION")
        print("=" * 60)
        print("\nCách sử dụng:")
        print("  python predict_date.py 2023-06-15")
        print("  python predict_date.py 2023-07-01")
        print("\nVí dụ:")
        print("  python predict_date.py 2023-05-20")
        print()

        # Load data to show date range
        daily_revenue = pd.read_csv('data/processed/daily_revenue.csv')
        daily_revenue['date'] = pd.to_datetime(daily_revenue['date'])
        min_date = daily_revenue['date'].min() + timedelta(days=28)
        max_date = daily_revenue['date'].max()

        print(f"📊 Dữ liệu có sẵn: {daily_revenue['date'].min().strftime('%Y-%m-%d')} đến {max_date.strftime('%Y-%m-%d')}")
        print(f"📅 Bạn có thể dự đoán cho ngày từ: {min_date.strftime('%Y-%m-%d')} trở đi")
        print()
        sys.exit(1)

    target_date_str = sys.argv[1]

    print("=" * 60)
    print("🔮 COFFEE SHOP REVENUE PREDICTION")
    print("=" * 60)
    print()

    # Train model
    model = train_model()

    # Load historical data
    daily_revenue = pd.read_csv('data/processed/daily_revenue.csv')
    daily_revenue['date'] = pd.to_datetime(daily_revenue['date'])

    # Predict
    result = predict_for_date(model, target_date_str, daily_revenue)

    if result:
        prediction, target_row = result
        target_date = pd.to_datetime(target_date_str)

        print("=" * 60)
        print(f"📅 Ngày dự đoán: {target_date.strftime('%A, %Y-%m-%d')}")
        print(f"💰 Doanh thu dự đoán: ${prediction:,.2f}")

        # Check if we have actual data
        actual_data = daily_revenue[daily_revenue['date'] == target_date]
        if not actual_data.empty:
            actual_revenue = actual_data['revenue'].values[0]
            error = abs(prediction - actual_revenue)
            mape = (error / actual_revenue) * 100
            print()
            print(f"✅ Doanh thu thực tế: ${actual_revenue:,.2f}")
            print(f"📊 Sai số: ${error:,.2f} ({mape:.2f}%)")

            if mape < 5:
                print("🎯 Dự đoán RẤT CHÍNH XÁC!")
            elif mape < 10:
                print("👍 Dự đoán TỐT!")
            else:
                print("⚠️  Dự đoán chưa chính xác lắm")
        else:
            print()
            print("ℹ️  Không có dữ liệu thực tế để so sánh (ngày trong tương lai)")

        print("=" * 60)
        print()

if __name__ == "__main__":
    main()
