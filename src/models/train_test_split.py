"""
Time Series Train/Val/Test Split
CRITICAL: Temporal split only - NO SHUFFLING!
"""
import pandas as pd
import numpy as np


def create_time_series_split(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Create temporal train/validation/test split for time series

    IMPORTANT: Uses temporal ordering - NO SHUFFLING!

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with datetime index
    train_ratio : float, default 0.8
        Proportion of data for training
    val_ratio : float, default 0.1
        Proportion of data for validation
    test_ratio : float, default 0.1
        Proportion of data for testing

    Returns:
    --------
    train, val, test : pd.DataFrame
        Train, validation, and test sets
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    n = len(df)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    # Temporal split
    train = df.iloc[:train_size]
    val = df.iloc[train_size:train_size + val_size]
    test = df.iloc[train_size + val_size:]

    print(f"\nTime Series Split Summary:")
    print(f"{'='*70}")
    print(f"Total samples: {n}")
    print(f"\nTrain set: {len(train)} samples ({len(train)/n*100:.1f}%)")
    print(f"  Date range: {train.index.min()} to {train.index.max()}")
    print(f"\nValidation set: {len(val)} samples ({len(val)/n*100:.1f}%)")
    print(f"  Date range: {val.index.min()} to {val.index.max()}")
    print(f"\nTest set: {len(test)} samples ({len(test)/n*100:.1f}%)")
    print(f"  Date range: {test.index.min()} to {test.index.max()}")
    print(f"{'='*70}")

    return train, val, test


def get_X_y_split(X, y, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split features and target into train/val/test sets

    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix with datetime index
    y : pd.Series
        Target variable with datetime index
    train_ratio, val_ratio, test_ratio : float
        Split ratios

    Returns:
    --------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Ensure indices match
    assert (X.index == y.index).all(), "X and y indices must match"

    # Create combined dataframe for splitting
    combined = X.copy()
    combined['__target__'] = y

    # Split
    train, val, test = create_time_series_split(
        combined, train_ratio, val_ratio, test_ratio
    )

    # Separate features and target
    X_train = train.drop('__target__', axis=1)
    y_train = train['__target__']

    X_val = val.drop('__target__', axis=1)
    y_val = val['__target__']

    X_test = test.drop('__target__', axis=1)
    y_test = test['__target__']

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    # Test split
    import pandas as pd

    # Load data
    df = pd.read_csv('../../data/processed/daily_revenue.csv',
                     index_col='date', parse_dates=True)

    print("Testing time series split...")
    train, val, test = create_time_series_split(df)

    print(f"\n✓ Split successful")
    print(f"Train shape: {train.shape}")
    print(f"Val shape: {val.shape}")
    print(f"Test shape: {test.shape}")
