"""
Run feature engineering pipeline
"""
import sys
sys.path.insert(0, 'src')
import pandas as pd
from features.feature_engineering import TimeSeriesFeatureEngine


def main():
    print("="*70)
    print(" FEATURE ENGINEERING PIPELINE")
    print("="*70 + "\n")

    # Load processed data
    print("[1/3] Loading processed daily revenue data...")
    df = pd.read_csv('data/processed/daily_revenue.csv',
                     index_col='date', parse_dates=True)
    print(f"✓ Loaded: {df.shape}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")

    # Initialize feature engine
    print("\n[2/3] Creating features...")
    fe = TimeSeriesFeatureEngine(lookback_days=28)

    # Create all features
    df_features, feature_cols = fe.create_all_features(df, target_col='revenue')

    # Prepare for modeling
    X, y = fe.prepare_for_modeling(df_features, target_col='revenue')

    print(f"\n{'='*70}")
    print("FEATURE ENGINEERING COMPLETE")
    print(f"{'='*70}")
    print(f"\nFinal dataset:")
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Target shape: {y.shape}")
    print(f"  Date range: {X.index.min()} to {X.index.max()}")

    # Feature groups
    print(f"\nFeature groups breakdown:")
    groups = fe.get_feature_importance_groups()
    for group_name, features in groups.items():
        print(f"  {group_name:15s}: {len(features):3d} features")

    print(f"\nTotal features: {len(feature_cols)}")

    # Save
    print(f"\n[3/3] Saving features...")
    df_features.to_csv('data/processed/daily_revenue_with_features.csv')
    print("✓ Full dataset with features saved to: data/processed/daily_revenue_with_features.csv")

    # Save feature names
    with open('data/processed/feature_names.txt', 'w') as f:
        for feat in feature_cols:
            f.write(feat + '\n')
    print("✓ Feature names saved to: data/processed/feature_names.txt")

    # Save X, y for modeling
    X.to_csv('data/processed/X.csv')
    y.to_csv('data/processed/y.csv')
    print("✓ X, y matrices saved")

    print(f"\n{'='*70}")
    print(" READY FOR MODELING")
    print(f"{'='*70}")

    return X, y, fe


if __name__ == "__main__":
    X, y, fe = main()
