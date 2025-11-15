"""
Data loading and preprocessing module
"""
import pandas as pd
import numpy as np
from pathlib import Path


class CoffeeShopDataLoader:
    """Load and preprocess coffee shop sales data"""

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.daily_revenue = None

    def load_raw_data(self):
        """Load raw Excel data"""
        print(f"Loading data from {self.file_path}...")
        self.df = pd.read_excel(self.file_path)
        print(f"Data loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self.df

    def explore_data(self):
        """Initial data exploration"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_raw_data() first.")

        print("\n" + "="*70)
        print("INITIAL DATA EXPLORATION")
        print("="*70)

        print(f"\nDataset shape: {self.df.shape}")
        print(f"\nColumn names:")
        for i, col in enumerate(self.df.columns, 1):
            print(f"  {i}. {col}")

        print(f"\nData types:")
        print(self.df.dtypes)

        print(f"\nMissing values:")
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("  ✓ No missing values!")
        else:
            print(missing[missing > 0])

        print(f"\nDuplicate rows: {self.df.duplicated().sum()}")

        print(f"\nFirst 5 rows:")
        print(self.df.head())

        print(f"\nBasic statistics:")
        print(self.df.describe())

        return self.df.info()

    def preprocess_datetime(self):
        """Parse and create datetime features"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_raw_data() first.")

        print("\n" + "="*70)
        print("DATETIME PREPROCESSING & REVENUE CALCULATION")
        print("="*70)

        # Calculate revenue (transaction_qty * unit_price)
        if 'transaction_qty' in self.df.columns and 'unit_price' in self.df.columns:
            self.df['revenue'] = self.df['transaction_qty'] * self.df['unit_price']
            print(f"\n✓ Revenue calculated: transaction_qty × unit_price")
            print(f"Total revenue: ${self.df['revenue'].sum():,.2f}")
            print(f"Average transaction revenue: ${self.df['revenue'].mean():.2f}")
        else:
            print("Warning: Could not calculate revenue - columns not found")

        # Identify datetime columns (adjust based on actual column names)
        # Common patterns: transaction_date, transaction_time, date, time, etc.
        date_cols = [col for col in self.df.columns if 'date' in col.lower()]
        time_cols = [col for col in self.df.columns if 'time' in col.lower()]

        print(f"\nDetected date columns: {date_cols}")
        print(f"Detected time columns: {time_cols}")

        # This will be adjusted based on actual column names
        if len(date_cols) > 0:
            date_col = date_cols[0]
            self.df['date'] = pd.to_datetime(self.df[date_col])

            # Extract temporal features
            self.df['year'] = self.df['date'].dt.year
            self.df['month'] = self.df['date'].dt.month
            self.df['day'] = self.df['date'].dt.day
            self.df['day_of_week'] = self.df['date'].dt.dayofweek
            self.df['day_name'] = self.df['date'].dt.day_name()
            self.df['week'] = self.df['date'].dt.isocalendar().week
            self.df['quarter'] = self.df['date'].dt.quarter
            self.df['is_weekend'] = self.df['day_of_week'].isin([5, 6]).astype(int)

            print(f"\nDate range: {self.df['date'].min()} to {self.df['date'].max()}")
            print(f"Total days: {self.df['date'].nunique()}")

        if len(time_cols) > 0:
            time_col = time_cols[0]
            # Handle time parsing
            try:
                self.df['time'] = pd.to_datetime(self.df[time_col], format='%H:%M:%S').dt.time
                self.df['hour'] = pd.to_datetime(self.df[time_col], format='%H:%M:%S').dt.hour
                self.df['minute'] = pd.to_datetime(self.df[time_col], format='%H:%M:%S').dt.minute
                print(f"Hour range: {self.df['hour'].min()} to {self.df['hour'].max()}")
            except:
                print(f"Warning: Could not parse time column {time_col}")

        return self.df

    def create_daily_revenue_series(self, revenue_col=None, date_col='date'):
        """
        Aggregate transactions to daily revenue time series

        Parameters:
        -----------
        revenue_col : str, optional
            Name of revenue/sales column. If None, will auto-detect
        date_col : str, default 'date'
            Name of date column
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_raw_data() first.")

        print("\n" + "="*70)
        print("CREATING DAILY REVENUE TIME SERIES")
        print("="*70)

        # Auto-detect revenue column if not specified
        if revenue_col is None:
            revenue_patterns = ['revenue', 'sales', 'amount', 'total', 'price']
            for pattern in revenue_patterns:
                matches = [col for col in self.df.columns if pattern in col.lower()]
                if matches:
                    revenue_col = matches[0]
                    break

        if revenue_col is None:
            raise ValueError("Could not auto-detect revenue column. Please specify revenue_col parameter.")

        print(f"Using revenue column: {revenue_col}")
        print(f"Using date column: {date_col}")

        # Aggregate to daily level
        self.daily_revenue = self.df.groupby(date_col).agg({
            revenue_col: 'sum'
        }).reset_index()

        self.daily_revenue.columns = ['date', 'revenue']
        self.daily_revenue = self.daily_revenue.set_index('date').sort_index()

        # Add transaction count
        transaction_count = self.df.groupby(date_col).size()
        self.daily_revenue['transaction_count'] = transaction_count

        # Calculate revenue per transaction
        self.daily_revenue['revenue_per_transaction'] = (
            self.daily_revenue['revenue'] / self.daily_revenue['transaction_count']
        )

        print(f"\nDaily revenue series created!")
        print(f"Date range: {self.daily_revenue.index.min()} to {self.daily_revenue.index.max()}")
        print(f"Total days: {len(self.daily_revenue)}")
        print(f"\nDaily Revenue Statistics:")
        print(self.daily_revenue.describe())

        # Check for missing dates
        date_range = pd.date_range(
            start=self.daily_revenue.index.min(),
            end=self.daily_revenue.index.max(),
            freq='D'
        )
        missing_dates = date_range.difference(self.daily_revenue.index)
        if len(missing_dates) > 0:
            print(f"\nWarning: {len(missing_dates)} missing dates detected:")
            print(missing_dates[:10])
        else:
            print("\n✓ No missing dates - continuous time series!")

        return self.daily_revenue

    def get_store_level_series(self, store_col=None, revenue_col=None, date_col='date'):
        """Create separate time series for each store"""
        if self.df is None:
            raise ValueError("Data not loaded.")

        # Auto-detect store column
        if store_col is None:
            store_patterns = ['store', 'location', 'branch', 'shop']
            for pattern in store_patterns:
                matches = [col for col in self.df.columns if pattern in col.lower()]
                if matches:
                    store_col = matches[0]
                    break

        if store_col is None:
            print("No store column found. Skipping store-level analysis.")
            return None

        print(f"\nStores found: {self.df[store_col].unique()}")

        # Auto-detect revenue column
        if revenue_col is None:
            revenue_patterns = ['revenue', 'sales', 'amount', 'total']
            for pattern in revenue_patterns:
                matches = [col for col in self.df.columns if pattern in col.lower()]
                if matches:
                    revenue_col = matches[0]
                    break

        store_series = self.df.groupby([date_col, store_col])[revenue_col].sum().unstack(fill_value=0)

        return store_series

    def save_processed_data(self, output_dir='../data/processed'):
        """Save processed daily revenue data"""
        if self.daily_revenue is None:
            raise ValueError("Daily revenue not created. Call create_daily_revenue_series() first.")

        output_path = Path(output_dir) / 'daily_revenue.csv'
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.daily_revenue.to_csv(output_path)
        print(f"\n✓ Daily revenue saved to: {output_path}")

        return output_path


if __name__ == "__main__":
    # Test the data loader
    data_path = "../data/raw/Coffee Shop Sales.xlsx"

    loader = CoffeeShopDataLoader(data_path)
    df = loader.load_raw_data()
    loader.explore_data()
    df = loader.preprocess_datetime()
    daily_revenue = loader.create_daily_revenue_series()
    loader.save_processed_data()
