"""
Exploratory Data Analysis for Coffee Shop Time Series
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesEDA:
    """Comprehensive EDA for time series data"""

    def __init__(self, daily_revenue, df=None):
        """
        Parameters:
        -----------
        daily_revenue : pd.DataFrame
            Daily aggregated revenue with date index
        df : pd.DataFrame, optional
            Transaction-level data for deeper analysis
        """
        self.daily_revenue = daily_revenue
        self.df = df
        self.decomposition = None

    def plot_time_series(self, save_path=None):
        """Plot main time series with trend"""
        fig, ax = plt.subplots(figsize=(15, 6))

        # Plot revenue
        ax.plot(self.daily_revenue.index, self.daily_revenue['revenue'],
                label='Daily Revenue', linewidth=2, alpha=0.7)

        # Add 7-day moving average
        ma7 = self.daily_revenue['revenue'].rolling(window=7).mean()
        ax.plot(self.daily_revenue.index, ma7,
                label='7-Day Moving Average', linewidth=2, color='red')

        # Add 30-day moving average
        ma30 = self.daily_revenue['revenue'].rolling(window=30).mean()
        ax.plot(self.daily_revenue.index, ma30,
                label='30-Day Moving Average', linewidth=2, color='green')

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Revenue ($)', fontsize=12)
        ax.set_title('Daily Coffee Shop Revenue - Time Series', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")
        plt.show()

        return fig

    def seasonal_decomposition(self, period=7, model='additive', save_path=None):
        """
        Decompose time series into trend, seasonal, and residual components

        Parameters:
        -----------
        period : int, default 7
            Period of seasonality (7 for weekly)
        model : str, default 'additive'
            Type of decomposition ('additive' or 'multiplicative')
        """
        print(f"\n{'='*70}")
        print(f"SEASONAL DECOMPOSITION (period={period}, model={model})")
        print(f"{'='*70}")

        self.decomposition = seasonal_decompose(
            self.daily_revenue['revenue'],
            model=model,
            period=period
        )

        # Plot decomposition
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))

        self.decomposition.observed.plot(ax=axes[0], color='blue')
        axes[0].set_ylabel('Observed', fontsize=11)
        axes[0].set_title('Time Series Decomposition', fontsize=14, fontweight='bold')
        axes[0].grid(alpha=0.3)

        self.decomposition.trend.plot(ax=axes[1], color='red')
        axes[1].set_ylabel('Trend', fontsize=11)
        axes[1].grid(alpha=0.3)

        self.decomposition.seasonal.plot(ax=axes[2], color='green')
        axes[2].set_ylabel('Seasonal', fontsize=11)
        axes[2].grid(alpha=0.3)

        self.decomposition.resid.plot(ax=axes[3], color='purple')
        axes[3].set_ylabel('Residual', fontsize=11)
        axes[3].set_xlabel('Date', fontsize=11)
        axes[3].grid(alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Decomposition plot saved to {save_path}")
        plt.show()

        # Calculate strength of trend and seasonality
        trend_strength = max(0, 1 - (self.decomposition.resid.var() /
                                      (self.decomposition.resid + self.decomposition.trend).var()))
        seasonal_strength = max(0, 1 - (self.decomposition.resid.var() /
                                         (self.decomposition.resid + self.decomposition.seasonal).var()))

        print(f"\nTrend Strength: {trend_strength:.4f}")
        print(f"Seasonal Strength: {seasonal_strength:.4f}")

        if trend_strength > 0.6:
            print("✓ Strong trend component detected")
        if seasonal_strength > 0.6:
            print("✓ Strong seasonal component detected")

        return self.decomposition

    def test_stationarity(self):
        """Perform stationarity tests (ADF and KPSS)"""
        print(f"\n{'='*70}")
        print("STATIONARITY TESTS")
        print(f"{'='*70}")

        series = self.daily_revenue['revenue'].dropna()

        # Augmented Dickey-Fuller test
        print('\n1. Augmented Dickey-Fuller Test (ADF):')
        print('   H0: Series has a unit root (non-stationary)')
        print('   H1: Series is stationary')
        adf_result = adfuller(series, autolag='AIC')
        print(f'\n   ADF Statistic: {adf_result[0]:.4f}')
        print(f'   p-value: {adf_result[1]:.4f}')
        print(f'   Lags used: {adf_result[2]}')
        print(f'   Critical Values:')
        for key, value in adf_result[4].items():
            print(f'      {key}: {value:.4f}')

        if adf_result[1] <= 0.05:
            print(f"\n   ✓ RESULT: Series is STATIONARY (p-value = {adf_result[1]:.4f} ≤ 0.05)")
            print("   We reject H0 - the series does not have a unit root")
        else:
            print(f"\n   ✗ RESULT: Series is NON-STATIONARY (p-value = {adf_result[1]:.4f} > 0.05)")
            print("   We fail to reject H0 - the series has a unit root")

        # KPSS test
        print('\n2. KPSS Test (Kwiatkowski-Phillips-Schmidt-Shin):')
        print('   H0: Series is trend-stationary')
        print('   H1: Series has a unit root (non-stationary)')
        kpss_result = kpss(series, regression='ct', nlags='auto')
        print(f'\n   KPSS Statistic: {kpss_result[0]:.4f}')
        print(f'   p-value: {kpss_result[1]:.4f}')
        print(f'   Lags used: {kpss_result[2]}')
        print(f'   Critical Values:')
        for key, value in kpss_result[3].items():
            print(f'      {key}: {value:.4f}')

        if kpss_result[1] >= 0.05:
            print(f"\n   ✓ RESULT: Series is STATIONARY (p-value = {kpss_result[1]:.4f} ≥ 0.05)")
            print("   We fail to reject H0 - the series is trend-stationary")
        else:
            print(f"\n   ✗ RESULT: Series is NON-STATIONARY (p-value = {kpss_result[1]:.4f} < 0.05)")
            print("   We reject H0 - the series has a unit root")

        # Combined interpretation
        print(f"\n{'='*70}")
        print("COMBINED INTERPRETATION:")
        if adf_result[1] <= 0.05 and kpss_result[1] >= 0.05:
            print("✓ Both tests agree: Series is STATIONARY")
            recommendation = "No differencing needed for modeling"
        elif adf_result[1] > 0.05 and kpss_result[1] < 0.05:
            print("✗ Both tests agree: Series is NON-STATIONARY")
            recommendation = "Differencing recommended (d=1 for ARIMA)"
        elif adf_result[1] <= 0.05 and kpss_result[1] < 0.05:
            print("⚠ Tests disagree: Difference-stationary")
            recommendation = "Consider differencing or trend removal"
        else:
            print("⚠ Tests disagree: Needs further investigation")
            recommendation = "Try transformations (log, sqrt) or differencing"

        print(f"RECOMMENDATION: {recommendation}")
        print(f"{'='*70}")

        return {'adf': adf_result, 'kpss': kpss_result}

    def plot_acf_pacf(self, lags=40, save_path=None):
        """Plot ACF and PACF for determining ARIMA parameters"""
        fig, axes = plt.subplots(2, 1, figsize=(15, 8))

        series = self.daily_revenue['revenue'].dropna()

        # ACF plot
        plot_acf(series, lags=lags, ax=axes[0], alpha=0.05)
        axes[0].set_title('Autocorrelation Function (ACF)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Lag', fontsize=11)
        axes[0].set_ylabel('Correlation', fontsize=11)
        axes[0].grid(alpha=0.3)

        # PACF plot
        plot_pacf(series, lags=lags, ax=axes[1], alpha=0.05, method='ywm')
        axes[1].set_title('Partial Autocorrelation Function (PACF)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Lag', fontsize=11)
        axes[1].set_ylabel('Correlation', fontsize=11)
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ ACF/PACF plot saved to {save_path}")
        plt.show()

        # Interpretation guide
        print(f"\n{'='*70}")
        print("ACF/PACF INTERPRETATION GUIDE:")
        print(f"{'='*70}")
        print("\nFor ARIMA(p,d,q) parameter selection:")
        print("  - ACF: Determines q (MA order)")
        print("    • Sharp cutoff at lag q → MA(q)")
        print("    • Gradual decay → AR component present")
        print("\n  - PACF: Determines p (AR order)")
        print("    • Sharp cutoff at lag p → AR(p)")
        print("    • Gradual decay → MA component present")
        print("\n  - d: Differencing order")
        print("    • Determined by stationarity tests")
        print("    • If non-stationary, start with d=1")
        print(f"{'='*70}")

        return fig

    def analyze_patterns(self, save_path_prefix=None):
        """Analyze temporal patterns (hourly, day of week, monthly)"""
        if self.df is None:
            print("Transaction-level data not provided. Skipping pattern analysis.")
            return None

        print(f"\n{'='*70}")
        print("TEMPORAL PATTERN ANALYSIS")
        print(f"{'='*70}")

        # Hourly pattern
        if 'hour' in self.df.columns:
            print("\n1. Hourly Revenue Pattern:")
            hourly = self.df.groupby('hour')['revenue'].agg(['sum', 'mean', 'count'])
            hourly.columns = ['Total Revenue', 'Avg Revenue', 'Transaction Count']
            print(hourly)

            # Plot hourly pattern
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))

            axes[0].bar(hourly.index, hourly['Total Revenue'], color='steelblue', alpha=0.7)
            axes[0].set_xlabel('Hour of Day', fontsize=11)
            axes[0].set_ylabel('Total Revenue ($)', fontsize=11)
            axes[0].set_title('Revenue by Hour of Day', fontsize=12, fontweight='bold')
            axes[0].grid(alpha=0.3, axis='y')
            axes[0].set_xticks(range(24))

            axes[1].bar(hourly.index, hourly['Transaction Count'], color='coral', alpha=0.7)
            axes[1].set_xlabel('Hour of Day', fontsize=11)
            axes[1].set_ylabel('Transaction Count', fontsize=11)
            axes[1].set_title('Transactions by Hour of Day', fontsize=12, fontweight='bold')
            axes[1].grid(alpha=0.3, axis='y')
            axes[1].set_xticks(range(24))

            plt.tight_layout()
            if save_path_prefix:
                path = f"{save_path_prefix}_hourly_pattern.png"
                plt.savefig(path, dpi=300, bbox_inches='tight')
                print(f"✓ Hourly pattern plot saved to {path}")
            plt.show()

            peak_hour = hourly['Total Revenue'].idxmax()
            print(f"\n   Peak hour: {peak_hour}:00 with ${hourly.loc[peak_hour, 'Total Revenue']:,.2f}")

        # Day of week pattern
        if 'day_name' in self.df.columns:
            print("\n2. Day of Week Revenue Pattern:")
            dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            dow = self.df.groupby('day_name')['revenue'].agg(['sum', 'mean', 'count'])
            dow = dow.reindex(dow_order)
            dow.columns = ['Total Revenue', 'Avg Revenue', 'Transaction Count']
            print(dow)

            # Plot day of week pattern
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))

            axes[0].bar(range(7), dow['Total Revenue'], color='seagreen', alpha=0.7)
            axes[0].set_xlabel('Day of Week', fontsize=11)
            axes[0].set_ylabel('Total Revenue ($)', fontsize=11)
            axes[0].set_title('Revenue by Day of Week', fontsize=12, fontweight='bold')
            axes[0].set_xticks(range(7))
            axes[0].set_xticklabels(dow_order, rotation=45, ha='right')
            axes[0].grid(alpha=0.3, axis='y')

            axes[1].bar(range(7), dow['Transaction Count'], color='mediumpurple', alpha=0.7)
            axes[1].set_xlabel('Day of Week', fontsize=11)
            axes[1].set_ylabel('Transaction Count', fontsize=11)
            axes[1].set_title('Transactions by Day of Week', fontsize=12, fontweight='bold')
            axes[1].set_xticks(range(7))
            axes[1].set_xticklabels(dow_order, rotation=45, ha='right')
            axes[1].grid(alpha=0.3, axis='y')

            plt.tight_layout()
            if save_path_prefix:
                path = f"{save_path_prefix}_dayofweek_pattern.png"
                plt.savefig(path, dpi=300, bbox_inches='tight')
                print(f"✓ Day of week pattern plot saved to {path}")
            plt.show()

            peak_day = dow['Total Revenue'].idxmax()
            print(f"\n   Peak day: {peak_day} with ${dow.loc[peak_day, 'Total Revenue']:,.2f}")

        # Monthly trend
        if 'month' in self.df.columns:
            print("\n3. Monthly Revenue Trend:")
            monthly = self.df.groupby('month')['revenue'].agg(['sum', 'mean', 'count'])
            monthly.columns = ['Total Revenue', 'Avg Revenue', 'Transaction Count']
            monthly.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'][:len(monthly)]
            print(monthly)

            # Calculate growth rate
            if len(monthly) > 1:
                total_growth = ((monthly['Total Revenue'].iloc[-1] - monthly['Total Revenue'].iloc[0]) /
                               monthly['Total Revenue'].iloc[0] * 100)
                print(f"\n   Total growth from {monthly.index[0]} to {monthly.index[-1]}: {total_growth:+.1f}%")

        return {'hourly': hourly if 'hour' in self.df.columns else None,
                'day_of_week': dow if 'day_name' in self.df.columns else None,
                'monthly': monthly if 'month' in self.df.columns else None}

    def analyze_stores(self, save_path=None):
        """Analyze revenue by store location"""
        if self.df is None or 'store_location' not in self.df.columns:
            print("Store location data not available.")
            return None

        print(f"\n{'='*70}")
        print("STORE-LEVEL ANALYSIS")
        print(f"{'='*70}")

        store_stats = self.df.groupby('store_location')['revenue'].agg([
            'sum', 'mean', 'count', 'std'
        ])
        store_stats.columns = ['Total Revenue', 'Avg Revenue', 'Transactions', 'Std Dev']
        store_stats['Revenue Share %'] = (store_stats['Total Revenue'] /
                                           store_stats['Total Revenue'].sum() * 100)

        print("\nStore Performance Summary:")
        print(store_stats.sort_values('Total Revenue', ascending=False))

        # Plot store comparison
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Total revenue by store
        store_stats['Total Revenue'].sort_values(ascending=False).plot(
            kind='barh', ax=axes[0], color='steelblue', alpha=0.7
        )
        axes[0].set_xlabel('Total Revenue ($)', fontsize=11)
        axes[0].set_title('Total Revenue by Store', fontsize=12, fontweight='bold')
        axes[0].grid(alpha=0.3, axis='x')

        # Revenue share
        axes[1].pie(store_stats['Total Revenue'], labels=store_stats.index, autopct='%1.1f%%',
                   startangle=90, colors=['steelblue', 'coral', 'seagreen'])
        axes[1].set_title('Revenue Share by Store', fontsize=12, fontweight='bold')

        # Average transaction revenue
        store_stats['Avg Revenue'].sort_values(ascending=False).plot(
            kind='barh', ax=axes[2], color='coral', alpha=0.7
        )
        axes[2].set_xlabel('Avg Revenue per Transaction ($)', fontsize=11)
        axes[2].set_title('Avg Transaction Value by Store', fontsize=12, fontweight='bold')
        axes[2].grid(alpha=0.3, axis='x')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Store analysis plot saved to {save_path}")
        plt.show()

        # Time series by store
        if 'date' in self.df.columns:
            store_daily = self.df.groupby(['date', 'store_location'])['revenue'].sum().unstack()

            fig, ax = plt.subplots(figsize=(15, 6))
            for store in store_daily.columns:
                ax.plot(store_daily.index, store_daily[store], label=store, linewidth=2, alpha=0.7)

            ax.set_xlabel('Date', fontsize=11)
            ax.set_ylabel('Daily Revenue ($)', fontsize=11)
            ax.set_title('Daily Revenue by Store Location', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)

            plt.tight_layout()
            if save_path:
                path = save_path.replace('.png', '_timeseries.png')
                plt.savefig(path, dpi=300, bbox_inches='tight')
                print(f"✓ Store time series plot saved to {path}")
            plt.show()

        return store_stats

    def analyze_products(self, save_path=None):
        """Analyze revenue by product category"""
        if self.df is None or 'product_category' not in self.df.columns:
            print("Product category data not available.")
            return None

        print(f"\n{'='*70}")
        print("PRODUCT CATEGORY ANALYSIS")
        print(f"{'='*70}")

        product_stats = self.df.groupby('product_category')['revenue'].agg([
            'sum', 'mean', 'count'
        ])
        product_stats.columns = ['Total Revenue', 'Avg Revenue', 'Transactions']
        product_stats['Revenue Share %'] = (product_stats['Total Revenue'] /
                                            product_stats['Total Revenue'].sum() * 100)
        product_stats = product_stats.sort_values('Total Revenue', ascending=False)

        print("\nProduct Category Performance:")
        print(product_stats)

        # Plot product analysis
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # Total revenue
        product_stats['Total Revenue'].plot(kind='barh', ax=axes[0, 0],
                                           color='steelblue', alpha=0.7)
        axes[0, 0].set_xlabel('Total Revenue ($)', fontsize=11)
        axes[0, 0].set_title('Revenue by Product Category', fontsize=12, fontweight='bold')
        axes[0, 0].grid(alpha=0.3, axis='x')

        # Transaction count
        product_stats['Transactions'].plot(kind='barh', ax=axes[0, 1],
                                          color='coral', alpha=0.7)
        axes[0, 1].set_xlabel('Transaction Count', fontsize=11)
        axes[0, 1].set_title('Transactions by Product Category', fontsize=12, fontweight='bold')
        axes[0, 1].grid(alpha=0.3, axis='x')

        # Revenue share pie chart
        axes[1, 0].pie(product_stats['Total Revenue'], labels=product_stats.index,
                      autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('Revenue Share by Category', fontsize=12, fontweight='bold')

        # Average transaction value
        product_stats['Avg Revenue'].plot(kind='barh', ax=axes[1, 1],
                                         color='seagreen', alpha=0.7)
        axes[1, 1].set_xlabel('Avg Revenue per Transaction ($)', fontsize=11)
        axes[1, 1].set_title('Avg Transaction Value by Category', fontsize=12, fontweight='bold')
        axes[1, 1].grid(alpha=0.3, axis='x')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Product analysis plot saved to {save_path}")
        plt.show()

        return product_stats

    def generate_summary_report(self):
        """Generate comprehensive summary of EDA findings"""
        print(f"\n{'='*70}")
        print("EDA SUMMARY REPORT")
        print(f"{'='*70}")

        print(f"\n📊 DATASET OVERVIEW:")
        print(f"  • Date range: {self.daily_revenue.index.min().date()} to {self.daily_revenue.index.max().date()}")
        print(f"  • Total days: {len(self.daily_revenue)}")
        print(f"  • Total revenue: ${self.daily_revenue['revenue'].sum():,.2f}")
        print(f"  • Daily average: ${self.daily_revenue['revenue'].mean():,.2f}")
        print(f"  • Daily std dev: ${self.daily_revenue['revenue'].std():,.2f}")
        print(f"  • Min daily revenue: ${self.daily_revenue['revenue'].min():,.2f}")
        print(f"  • Max daily revenue: ${self.daily_revenue['revenue'].max():,.2f}")

        if self.df is not None:
            print(f"\n  • Total transactions: {len(self.df):,}")
            print(f"  • Avg transactions/day: {len(self.df) / len(self.daily_revenue):.0f}")
            print(f"  • Avg revenue/transaction: ${self.df['revenue'].mean():.2f}")

        # Trend analysis
        print(f"\n📈 TREND ANALYSIS:")
        first_week_avg = self.daily_revenue['revenue'].iloc[:7].mean()
        last_week_avg = self.daily_revenue['revenue'].iloc[-7:].mean()
        growth = ((last_week_avg - first_week_avg) / first_week_avg * 100)
        print(f"  • First week average: ${first_week_avg:,.2f}")
        print(f"  • Last week average: ${last_week_avg:,.2f}")
        print(f"  • Growth: {growth:+.1f}%")

        # Volatility
        cv = (self.daily_revenue['revenue'].std() / self.daily_revenue['revenue'].mean()) * 100
        print(f"\n📉 VOLATILITY:")
        print(f"  • Coefficient of Variation: {cv:.1f}%")
        if cv < 20:
            print("  • Assessment: Low volatility - stable revenue")
        elif cv < 40:
            print("  • Assessment: Moderate volatility")
        else:
            print("  • Assessment: High volatility - unstable revenue")

        print(f"\n{'='*70}")

        return {
            'total_days': len(self.daily_revenue),
            'total_revenue': self.daily_revenue['revenue'].sum(),
            'daily_mean': self.daily_revenue['revenue'].mean(),
            'daily_std': self.daily_revenue['revenue'].std(),
            'growth_rate': growth,
            'coefficient_of_variation': cv
        }


if __name__ == "__main__":
    # Load processed data
    daily_revenue = pd.read_csv('../data/processed/daily_revenue.csv', index_col='date', parse_dates=True)

    # Initialize EDA
    eda = TimeSeriesEDA(daily_revenue)

    # Run analyses
    eda.plot_time_series(save_path='../results/timeseries_plot.png')
    eda.seasonal_decomposition(period=7, save_path='../results/decomposition.png')
    eda.test_stationarity()
    eda.plot_acf_pacf(lags=40, save_path='../results/acf_pacf.png')
    summary = eda.generate_summary_report()
