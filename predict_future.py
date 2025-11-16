"""
Predict future revenue - Uses saved model
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import sys
import os
import pickle

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from features.feature_engineering import TimeSeriesFeatureEngine

MODEL_PATH = 'models/lightgbm_model.pkl'

def train_and_save_model():
    """Train and save the model"""
    print("🔄 Training new model...")
    X = pd.read_csv('data/processed/X.csv', index_col=0)
    y = pd.read_csv('data/processed/y.csv', index_col=0)

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

    # Save model
    os.makedirs('models', exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    print(f"✅ Model saved to {MODEL_PATH}")
    return model

def load_or_train_model():
    """Load model if exists, otherwise train new one"""
    if os.path.exists(MODEL_PATH):
        print(f"✅ Loading model from {MODEL_PATH}")
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    else:
        return train_and_save_model()

def create_features_for_future_date(target_date, historical_data):
    """
    Create features for a future date using historical data
    """
    target_date = pd.to_datetime(target_date)

    # We need historical revenue data up to the day before target_date
    # Generate synthetic future data by forward-filling from last known value
    last_date = historical_data.index[-1]
    last_revenue = historical_data['revenue'].iloc[-1]

    # Create date range from last known date to target date
    future_dates = pd.date_range(start=last_date + timedelta(days=1), end=target_date, freq='D')

    if len(future_dates) == 0:
        # Target date is in the past (within historical data range)
        # Just use historical data
        combined_df = historical_data.copy()
    else:
        # Create placeholder data for future dates
        # Use rolling average of last 7 days as estimate
        rolling_avg = historical_data['revenue'].tail(7).mean()

        future_df = pd.DataFrame({
            'revenue': [rolling_avg] * len(future_dates)
        }, index=future_dates)

        # Combine historical and future data
        combined_df = pd.concat([historical_data, future_df])

    # Create features using the feature engine
    feature_engine = TimeSeriesFeatureEngine()
    features_df, _ = feature_engine.create_all_features(combined_df)  # Returns (df, feature_list)

    # Get features for target date
    target_features = features_df.loc[[target_date]].drop(['revenue'], axis=1, errors='ignore')

    return target_features

def predict_future_date(target_date_str):
    """Predict revenue for a future date"""
    try:
        target_date = pd.to_datetime(target_date_str)

        # Load model
        model = load_or_train_model()

        # Load historical data
        print("📊 Loading historical data...")
        daily_revenue = pd.read_csv('data/processed/daily_revenue.csv')
        daily_revenue['date'] = pd.to_datetime(daily_revenue['date'])
        daily_revenue = daily_revenue.set_index('date')[['revenue']]

        min_predictable_date = daily_revenue.index[0] + timedelta(days=28)

        if target_date < min_predictable_date:
            print(f"⚠️  Ngày quá xa trong quá khứ!")
            print(f"   Cần ít nhất 28 ngày historical data")
            print(f"   Vui lòng chọn ngày từ {min_predictable_date.strftime('%Y-%m-%d')} trở đi")
            return None

        print(f"🔮 Creating features for {target_date.strftime('%Y-%m-%d')}...")

        # Create features
        X_predict = create_features_for_future_date(target_date, daily_revenue)

        # Check for NaN
        nan_count = X_predict.isna().sum().sum()
        if nan_count > 0:
            print(f"⚠️  Warning: {nan_count} NaN values found, filling with 0")
            X_predict = X_predict.fillna(0)

        # Predict
        prediction = model.predict(X_predict)[0]

        return {
            'date': target_date,
            'prediction': prediction,
            'has_actual': target_date in daily_revenue.index,
            'actual': daily_revenue.loc[target_date, 'revenue'] if target_date in daily_revenue.index else None
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    if len(sys.argv) < 2:
        print("=" * 70)
        print("🔮 COFFEE SHOP REVENUE PREDICTION - FUTURE DATES")
        print("=" * 70)
        print("\nCách sử dụng:")
        print("  python predict_future.py 2023-07-15")
        print("  python predict_future.py 2023-08-01")
        print("  python predict_future.py 2024-01-01")
        print("\nScript này dự đoán cho BẤT KỲ ngày nào trong tương lai!")
        print()

        # Load historical data range
        daily_revenue = pd.read_csv('data/processed/daily_revenue.csv')
        daily_revenue['date'] = pd.to_datetime(daily_revenue['date'])
        min_date = daily_revenue['date'].min() + timedelta(days=28)
        max_date = daily_revenue['date'].max()

        print(f"📊 Dữ liệu historical: {daily_revenue['date'].min().strftime('%Y-%m-%d')} đến {max_date.strftime('%Y-%m-%d')}")
        print(f"📅 Có thể dự đoán từ: {min_date.strftime('%Y-%m-%d')} trở đi")
        print(f"💡 Dự đoán cho ngày sau {max_date.strftime('%Y-%m-%d')} sẽ ước lượng future revenue")
        print()
        sys.exit(1)

    target_date_str = sys.argv[1]

    print("=" * 70)
    print("🔮 COFFEE SHOP REVENUE PREDICTION")
    print("=" * 70)
    print()

    result = predict_future_date(target_date_str)

    if result:
        print()
        print("=" * 70)
        print(f"📅 Ngày: {result['date'].strftime('%A, %Y-%m-%d')}")
        print(f"💰 Doanh thu dự đoán: ${result['prediction']:,.2f}")
        print()

        if result['has_actual']:
            actual = result['actual']
            error = abs(result['prediction'] - actual)
            mape = (error / actual) * 100
            print(f"✅ Doanh thu thực tế: ${actual:,.2f}")
            print(f"📊 Sai số: ${error:,.2f} ({mape:.2f}%)")
            print()

            if mape < 5:
                print("🎯 Dự đoán RẤT CHÍNH XÁC!")
            elif mape < 10:
                print("👍 Dự đoán TỐT!")
            elif mape < 15:
                print("✓ Dự đoán CHẤP NHẬN ĐƯỢC")
            else:
                print("⚠️  Dự đoán chưa chính xác lắm")
        else:
            print("ℹ️  Ngày trong tương lai - không có dữ liệu thực tế để so sánh")
            print("💡 Dự đoán dựa trên trend và patterns từ historical data")

        print("=" * 70)
        print()

if __name__ == "__main__":
    main()
