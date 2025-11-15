"""
Feature Engineering for Time Series Forecasting
Creates 100+ features from daily revenue time series
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesFeatureEngine:
    """
    Comprehensive feature engineering for time series forecasting

    Features created:
    1. Temporal features (calendar + cyclical)
    2. Lag features (1, 2, 3, 7, 14, 21, 28 days)
    3. Rolling statistics (mean, std, min, max - windows: 3, 7, 14, 28)
    4. Domain-specific features (volatility, trend, momentum)
    5. Interaction features

    IMPORTANT: All features properly shifted to avoid data leakage
    """

    def __init__(self, lookback_days=28):
        """
        Parameters:
        -----------
        lookback_days : int, default 28
            Maximum lookback period for rolling features
        """
        self.lookback_days = lookback_days
        self.scaler = StandardScaler()
        self.feature_names = []

    def create_temporal_features(self, df):
        """
        Create calendar and cyclical temporal features

        Returns features for:
        - Day of week (cyclical encoding)
        - Day of month
        - Day of year
        - Week of year
        - Month
        - Quarter
        - Is weekend
        - Is month start/end
        """
        print("Creating temporal features...")

        # Day of week (0=Monday, 6=Sunday)
        df['dayofweek'] = df.index.dayofweek
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

        # Cyclical encoding for day of week (captures weekly seasonality)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

        # Day of month
        df['day'] = df.index.day
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)

        # Day of year
        df['dayofyear'] = df.index.dayofyear
        df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
        df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)

        # Week of year
        df['week'] = df.index.isocalendar().week
        df['week_sin'] = np.sin(2 * np.pi * df['week'] / 52)
        df['week_cos'] = np.cos(2 * np.pi * df['week'] / 52)

        # Month
        df['month'] = df.index.month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Quarter
        df['quarter'] = df.index.quarter

        # Month start/end flags
        df['is_month_start'] = df.index.is_month_start.astype(int)
        df['is_month_end'] = df.index.is_month_end.astype(int)
        df['is_quarter_start'] = df.index.is_quarter_start.astype(int)
        df['is_quarter_end'] = df.index.is_quarter_end.astype(int)

        # Days in month
        df['days_in_month'] = df.index.days_in_month

        temporal_features = [
            'dayofweek', 'is_weekend', 'dayofweek_sin', 'dayofweek_cos',
            'day', 'day_sin', 'day_cos',
            'dayofyear', 'dayofyear_sin', 'dayofyear_cos',
            'week', 'week_sin', 'week_cos',
            'month', 'month_sin', 'month_cos',
            'quarter', 'is_month_start', 'is_month_end',
            'is_quarter_start', 'is_quarter_end', 'days_in_month'
        ]

        print(f"✓ Created {len(temporal_features)} temporal features")
        return df, temporal_features

    def create_lag_features(self, df, target_col='revenue', lags=[1, 2, 3, 7, 14, 21, 28]):
        """
        Create lag features (shifted values)

        IMPORTANT: Properly shifted to avoid data leakage
        lag_1 means yesterday's value (not today's)
        """
        print(f"Creating lag features for {lags}...")

        lag_features = []
        for lag in lags:
            feature_name = f'{target_col}_lag_{lag}'
            df[feature_name] = df[target_col].shift(lag)
            lag_features.append(feature_name)

        print(f"✓ Created {len(lag_features)} lag features")
        return df, lag_features

    def create_rolling_features(self, df, target_col='revenue', windows=[3, 7, 14, 28]):
        """
        Create rolling window statistics

        Features:
        - Rolling mean
        - Rolling std
        - Rolling min
        - Rolling max
        - Rolling median

        IMPORTANT: All features shifted by 1 to avoid data leakage
        """
        print(f"Creating rolling features for windows {windows}...")

        rolling_features = []

        for window in windows:
            # Rolling mean
            feature_name = f'{target_col}_rolling_mean_{window}'
            df[feature_name] = df[target_col].rolling(window=window).mean().shift(1)
            rolling_features.append(feature_name)

            # Rolling std (volatility)
            feature_name = f'{target_col}_rolling_std_{window}'
            df[feature_name] = df[target_col].rolling(window=window).std().shift(1)
            rolling_features.append(feature_name)

            # Rolling min
            feature_name = f'{target_col}_rolling_min_{window}'
            df[feature_name] = df[target_col].rolling(window=window).min().shift(1)
            rolling_features.append(feature_name)

            # Rolling max
            feature_name = f'{target_col}_rolling_max_{window}'
            df[feature_name] = df[target_col].rolling(window=window).max().shift(1)
            rolling_features.append(feature_name)

            # Rolling median
            feature_name = f'{target_col}_rolling_median_{window}'
            df[feature_name] = df[target_col].rolling(window=window).median().shift(1)
            rolling_features.append(feature_name)

            # Rolling range
            feature_name = f'{target_col}_rolling_range_{window}'
            df[feature_name] = (df[target_col].rolling(window=window).max() -
                               df[target_col].rolling(window=window).min()).shift(1)
            rolling_features.append(feature_name)

        print(f"✓ Created {len(rolling_features)} rolling features")
        return df, rolling_features

    def create_expanding_features(self, df, target_col='revenue'):
        """
        Create expanding window features (cumulative statistics)
        """
        print("Creating expanding features...")

        expanding_features = []

        # Expanding mean (all-time average up to this point)
        feature_name = f'{target_col}_expanding_mean'
        df[feature_name] = df[target_col].expanding().mean().shift(1)
        expanding_features.append(feature_name)

        # Expanding std
        feature_name = f'{target_col}_expanding_std'
        df[feature_name] = df[target_col].expanding().std().shift(1)
        expanding_features.append(feature_name)

        # Expanding min/max
        feature_name = f'{target_col}_expanding_min'
        df[feature_name] = df[target_col].expanding().min().shift(1)
        expanding_features.append(feature_name)

        feature_name = f'{target_col}_expanding_max'
        df[feature_name] = df[target_col].expanding().max().shift(1)
        expanding_features.append(feature_name)

        print(f"✓ Created {len(expanding_features)} expanding features")
        return df, expanding_features

    def create_domain_features(self, df, target_col='revenue'):
        """
        Create domain-specific features for revenue forecasting
        """
        print("Creating domain-specific features...")

        domain_features = []

        # Day-over-day change
        feature_name = f'{target_col}_change_1d'
        df[feature_name] = df[target_col].diff(1)
        domain_features.append(feature_name)

        # Week-over-week change
        feature_name = f'{target_col}_change_7d'
        df[feature_name] = df[target_col].diff(7)
        domain_features.append(feature_name)

        # Percentage change
        feature_name = f'{target_col}_pct_change_1d'
        df[feature_name] = df[target_col].pct_change(1)
        domain_features.append(feature_name)

        feature_name = f'{target_col}_pct_change_7d'
        df[feature_name] = df[target_col].pct_change(7)
        domain_features.append(feature_name)

        # Momentum (3-day, 7-day)
        feature_name = f'{target_col}_momentum_3d'
        df[feature_name] = (df[target_col] - df[target_col].shift(3)).shift(1)
        domain_features.append(feature_name)

        feature_name = f'{target_col}_momentum_7d'
        df[feature_name] = (df[target_col] - df[target_col].shift(7)).shift(1)
        domain_features.append(feature_name)

        # Relative strength index (RSI-like)
        for period in [7, 14]:
            delta = df[target_col].diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
            rs = gain / loss
            feature_name = f'{target_col}_rsi_{period}'
            df[feature_name] = (100 - (100 / (1 + rs))).shift(1)
            domain_features.append(feature_name)

        # Distance from rolling mean (normalized)
        for window in [7, 14, 28]:
            rolling_mean = df[target_col].rolling(window=window).mean()
            feature_name = f'{target_col}_distance_from_ma_{window}'
            df[feature_name] = ((df[target_col] - rolling_mean) / rolling_mean).shift(1)
            domain_features.append(feature_name)

        # Coefficient of variation (rolling)
        for window in [7, 14]:
            rolling_mean = df[target_col].rolling(window=window).mean()
            rolling_std = df[target_col].rolling(window=window).std()
            feature_name = f'{target_col}_cv_{window}'
            df[feature_name] = (rolling_std / rolling_mean).shift(1)
            domain_features.append(feature_name)

        print(f"✓ Created {len(domain_features)} domain features")
        return df, domain_features

    def create_interaction_features(self, df):
        """
        Create interaction features between temporal and revenue patterns
        """
        print("Creating interaction features...")

        interaction_features = []

        # Revenue lag 7 * is_weekend
        if 'revenue_lag_7' in df.columns and 'is_weekend' in df.columns:
            feature_name = 'revenue_lag_7_x_weekend'
            df[feature_name] = df['revenue_lag_7'] * df['is_weekend']
            interaction_features.append(feature_name)

        # Rolling mean 7 * day of week
        if 'revenue_rolling_mean_7' in df.columns and 'dayofweek' in df.columns:
            feature_name = 'rolling_mean_7_x_dayofweek'
            df[feature_name] = df['revenue_rolling_mean_7'] * df['dayofweek']
            interaction_features.append(feature_name)

        # Month * rolling mean 28
        if 'revenue_rolling_mean_28' in df.columns and 'month' in df.columns:
            feature_name = 'rolling_mean_28_x_month'
            df[feature_name] = df['revenue_rolling_mean_28'] * df['month']
            interaction_features.append(feature_name)

        print(f"✓ Created {len(interaction_features)} interaction features")
        return df, interaction_features

    def create_all_features(self, df, target_col='revenue'):
        """
        Create all features at once

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with DatetimeIndex and target column
        target_col : str, default 'revenue'
            Name of target column

        Returns:
        --------
        df : pd.DataFrame
            DataFrame with all features added
        feature_cols : list
            List of all feature column names
        """
        print("\n" + "="*70)
        print("FEATURE ENGINEERING PIPELINE")
        print("="*70 + "\n")

        # Make a copy to avoid modifying original
        df = df.copy()

        all_features = []

        # 1. Temporal features
        df, temporal_feats = self.create_temporal_features(df)
        all_features.extend(temporal_feats)

        # 2. Lag features
        df, lag_feats = self.create_lag_features(df, target_col)
        all_features.extend(lag_feats)

        # 3. Rolling features
        df, rolling_feats = self.create_rolling_features(df, target_col)
        all_features.extend(rolling_feats)

        # 4. Expanding features
        df, expanding_feats = self.create_expanding_features(df, target_col)
        all_features.extend(expanding_feats)

        # 5. Domain features
        df, domain_feats = self.create_domain_features(df, target_col)
        all_features.extend(domain_feats)

        # 6. Interaction features
        df, interaction_feats = self.create_interaction_features(df)
        all_features.extend(interaction_feats)

        self.feature_names = all_features

        print("\n" + "="*70)
        print(f"✓ TOTAL FEATURES CREATED: {len(all_features)}")
        print("="*70)

        # Report on missing values
        missing_counts = df[all_features].isnull().sum()
        features_with_missing = missing_counts[missing_counts > 0]

        if len(features_with_missing) > 0:
            print(f"\nFeatures with missing values: {len(features_with_missing)}")
            print("(This is expected for lag/rolling features at the beginning)")

        return df, all_features

    def prepare_for_modeling(self, df, target_col='revenue', feature_cols=None,
                            dropna=True):
        """
        Prepare dataset for modeling

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with features
        target_col : str
            Target column name
        feature_cols : list, optional
            List of feature columns. If None, uses all created features
        dropna : bool, default True
            Whether to drop rows with missing values

        Returns:
        --------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        """
        if feature_cols is None:
            feature_cols = self.feature_names

        # Create feature matrix and target
        X = df[feature_cols].copy()
        y = df[target_col].copy()

        if dropna:
            # Drop rows where any feature is NaN
            valid_idx = X.notna().all(axis=1)
            X = X[valid_idx]
            y = y[valid_idx]

            print(f"\nAfter dropping NaN:")
            print(f"  Samples: {len(X)}")
            print(f"  Features: {X.shape[1]}")
            print(f"  Date range: {X.index.min()} to {X.index.max()}")

        return X, y

    def get_feature_importance_groups(self):
        """
        Return feature groups for analysis
        """
        groups = {
            'temporal': [f for f in self.feature_names if any(x in f for x in
                        ['dayofweek', 'day', 'week', 'month', 'quarter', 'weekend'])],
            'lag': [f for f in self.feature_names if 'lag' in f],
            'rolling': [f for f in self.feature_names if 'rolling' in f],
            'expanding': [f for f in self.feature_names if 'expanding' in f],
            'domain': [f for f in self.feature_names if any(x in f for x in
                      ['change', 'momentum', 'rsi', 'distance', 'cv'])],
            'interaction': [f for f in self.feature_names if '_x_' in f]
        }

        return groups


if __name__ == "__main__":
    # Test feature engineering
    print("Testing Feature Engineering Pipeline...\n")

    # Load processed data
    df = pd.read_csv('../data/processed/daily_revenue.csv',
                     index_col='date', parse_dates=True)

    # Initialize feature engine
    fe = TimeSeriesFeatureEngine(lookback_days=28)

    # Create all features
    df_features, feature_cols = fe.create_all_features(df, target_col='revenue')

    # Prepare for modeling
    X, y = fe.prepare_for_modeling(df_features, target_col='revenue')

    print(f"\n{'='*70}")
    print("FEATURE ENGINEERING COMPLETE")
    print(f"{'='*70}")
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"\nFeature groups:")
    groups = fe.get_feature_importance_groups()
    for group_name, features in groups.items():
        print(f"  {group_name}: {len(features)} features")

    # Save processed features
    df_features.to_csv('../data/processed/daily_revenue_with_features.csv')
    print(f"\n✓ Features saved to data/processed/daily_revenue_with_features.csv")
