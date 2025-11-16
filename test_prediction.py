"""
Simple test script - Predict for dates that already have features
This uses the pre-computed features from X.csv
"""

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import sys

def main():
    if len(sys.argv) < 2:
        print("=" * 70)
        print("🔮 COFFEE SHOP REVENUE PREDICTION - QUICK TEST")
        print("=" * 70)
        print("\nCách sử dụng:")
        print("  python test_prediction.py 2023-05-20")
        print("\nVí dụ:")
        print("  python test_prediction.py 2023-03-15")
        print("  python test_prediction.py 2023-04-20")
        print("  python test_prediction.py 2023-06-10")
        print()

        # Show available date range
        y = pd.read_csv('data/processed/y.csv')
        y['date'] = pd.to_datetime(y['date'])
        print(f"📅 Ngày khả dụng: {y['date'].min().strftime('%Y-%m-%d')} đến {y['date'].max().strftime('%Y-%m-%d')}")
        print(f"📊 Tổng số ngày: {len(y)}")
        print()
        sys.exit(1)

    target_date = sys.argv[1]

    print("=" * 70)
    print("🔮 COFFEE SHOP REVENUE PREDICTION")
    print("=" * 70)
    print()

    # Load data
    print("🔄 Loading data...")
    X = pd.read_csv('data/processed/X.csv')
    y_df = pd.read_csv('data/processed/y.csv')
    y_df['date'] = pd.to_datetime(y_df['date'])

    # Drop date from X if exists
    if 'date' in X.columns:
        dates = X['date']
        X = X.drop('date', axis=1)
    else:
        dates = y_df['date']

    # Find the target date
    target_idx = y_df[y_df['date'] == target_date].index

    if len(target_idx) == 0:
        print(f"❌ Ngày {target_date} không tồn tại trong dataset")
        print(f"\n📅 Vui lòng chọn ngày từ {y_df['date'].min().strftime('%Y-%m-%d')} đến {y_df['date'].max().strftime('%Y-%m-%d')}")
        sys.exit(1)

    target_idx = target_idx[0]

    # Split data - remove the target date from training
    X_target = X.iloc[target_idx:target_idx+1]
    y_target = y_df.iloc[target_idx]['revenue']

    # Train on all other data
    X_train = pd.concat([X.iloc[:target_idx], X.iloc[target_idx+1:]])
    y_train = pd.concat([y_df.iloc[:target_idx]['revenue'], y_df.iloc[target_idx+1:]['revenue']])

    print(f"✓ Training on {len(X_train)} days")
    print(f"✓ Predicting for 1 day")
    print()

    print("🔄 Training model...")
    model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=31,
        random_state=42,
        verbose=-1
    )

    model.fit(X_train, y_train)
    print("✅ Model trained!\n")

    # Predict
    prediction = model.predict(X_target)[0]

    # Display results
    print("=" * 70)
    target_datetime = pd.to_datetime(target_date)
    print(f"📅 Ngày dự đoán: {target_datetime.strftime('%A, %Y-%m-%d')}")
    print(f"💰 Doanh thu dự đoán: ${prediction:,.2f}")
    print()
    print(f"✅ Doanh thu thực tế: ${y_target:,.2f}")

    error = abs(prediction - y_target)
    mape = (error / y_target) * 100
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

    print("=" * 70)
    print()

if __name__ == "__main__":
    main()
