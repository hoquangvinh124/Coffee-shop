"""
Baseline Time Series Models
- Naive (persistence)
- Moving Average
- Seasonal Naive
- ARIMA
- SARIMA
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class BaselineModels:
    """
    Collection of baseline time series forecasting models
    """

    def __init__(self):
        self.models = {}
        self.forecasts = {}
        self.metrics = {}

    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """
        Calculate forecasting metrics

        Returns:
        --------
        dict with RMSE, MAE, MAPE, R2, MBD
        """
        # Remove NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0:
            return {
                'RMSE': np.nan,
                'MAE': np.nan,
                'MAPE': np.nan,
                'R2': np.nan,
                'MBD': np.nan
            }

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        # R-squared
        r2 = r2_score(y_true, y_pred)

        # MBD (Mean Bias Deviation) - systematic over/under-prediction
        mbd = np.mean(y_pred - y_true)

        return {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2,
            'MBD': mbd
        }

    def naive_forecast(self, train, test):
        """
        Naive (Persistence) Model: Tomorrow = Today

        Parameters:
        -----------
        train : pd.Series
            Training data
        test : pd.Series
            Test data (for evaluation dates)

        Returns:
        --------
        forecast : pd.Series
            Forecasted values
        """
        print("\n[Naive Model] Training...")

        # Forecast: Use the last value from train for all test periods
        last_value = train.iloc[-1]
        forecast = pd.Series([last_value] * len(test), index=test.index)

        self.forecasts['naive'] = forecast
        metrics = self.calculate_metrics(test.values, forecast.values)
        self.metrics['naive'] = metrics

        print(f"  RMSE: ${metrics['RMSE']:.2f}")
        print(f"  MAE: ${metrics['MAE']:.2f}")
        print(f"  MAPE: {metrics['MAPE']:.2f}%")
        print(f"  R²: {metrics['R2']:.4f}")

        return forecast

    def moving_average_forecast(self, train, test, window=7):
        """
        Moving Average Model

        Parameters:
        -----------
        window : int
            Window size for moving average
        """
        print(f"\n[Moving Average {window}] Training...")

        # Calculate moving average from training data
        ma = train.rolling(window=window).mean()
        last_ma = ma.iloc[-1]

        # Forecast: Use last MA value for all test periods
        forecast = pd.Series([last_ma] * len(test), index=test.index)

        model_name = f'ma_{window}'
        self.forecasts[model_name] = forecast
        metrics = self.calculate_metrics(test.values, forecast.values)
        self.metrics[model_name] = metrics

        print(f"  RMSE: ${metrics['RMSE']:.2f}")
        print(f"  MAE: ${metrics['MAE']:.2f}")
        print(f"  MAPE: {metrics['MAPE']:.2f}%")
        print(f"  R²: {metrics['R2']:.4f}")

        return forecast

    def seasonal_naive_forecast(self, train, test, season_length=7):
        """
        Seasonal Naive: Tomorrow = Same day last week

        Parameters:
        -----------
        season_length : int, default 7
            Seasonal period (7 for weekly seasonality)
        """
        print(f"\n[Seasonal Naive (period={season_length})] Training...")

        # Forecast: Use values from season_length days ago
        forecast = []
        for i in range(len(test)):
            # Get value from season_length days before test period
            idx = -(season_length - (i % season_length))
            if abs(idx) <= len(train):
                forecast.append(train.iloc[idx])
            else:
                forecast.append(train.iloc[-1])

        forecast = pd.Series(forecast, index=test.index)

        self.forecasts['seasonal_naive'] = forecast
        metrics = self.calculate_metrics(test.values, forecast.values)
        self.metrics['seasonal_naive'] = metrics

        print(f"  RMSE: ${metrics['RMSE']:.2f}")
        print(f"  MAE: ${metrics['MAE']:.2f}")
        print(f"  MAPE: {metrics['MAPE']:.2f}%")
        print(f"  R²: {metrics['R2']:.4f}")

        return forecast

    def arima_forecast(self, train, test, order=(1, 1, 1)):
        """
        ARIMA Model

        Parameters:
        -----------
        order : tuple (p, d, q)
            ARIMA order
        """
        print(f"\n[ARIMA{order}] Training...")

        try:
            # Fit ARIMA model
            model = ARIMA(train, order=order)
            fitted_model = model.fit()
            self.models['arima'] = fitted_model

            # Forecast
            forecast = fitted_model.forecast(steps=len(test))
            forecast = pd.Series(forecast, index=test.index)

            self.forecasts['arima'] = forecast
            metrics = self.calculate_metrics(test.values, forecast.values)
            self.metrics['arima'] = metrics

            print(f"  RMSE: ${metrics['RMSE']:.2f}")
            print(f"  MAE: ${metrics['MAE']:.2f}")
            print(f"  MAPE: {metrics['MAPE']:.2f}%")
            print(f"  R²: {metrics['R2']:.4f}")

            return forecast

        except Exception as e:
            print(f"  ✗ ARIMA failed: {e}")
            return None

    def sarima_forecast(self, train, test, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
        """
        SARIMA Model (ARIMA with seasonality)

        Parameters:
        -----------
        order : tuple (p, d, q)
            ARIMA order
        seasonal_order : tuple (P, D, Q, s)
            Seasonal order
        """
        print(f"\n[SARIMA{order}x{seasonal_order}] Training...")

        try:
            # Fit SARIMA model
            model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
            fitted_model = model.fit(disp=False)
            self.models['sarima'] = fitted_model

            # Forecast
            forecast = fitted_model.forecast(steps=len(test))
            forecast = pd.Series(forecast, index=test.index)

            self.forecasts['sarima'] = forecast
            metrics = self.calculate_metrics(test.values, forecast.values)
            self.metrics['sarima'] = metrics

            print(f"  RMSE: ${metrics['RMSE']:.2f}")
            print(f"  MAE: ${metrics['MAE']:.2f}")
            print(f"  MAPE: {metrics['MAPE']:.2f}%")
            print(f"  R²: {metrics['R2']:.4f}")

            return forecast

        except Exception as e:
            print(f"  ✗ SARIMA failed: {e}")
            return None

    def train_all_baselines(self, train, test):
        """
        Train all baseline models

        Parameters:
        -----------
        train : pd.Series
            Training data
        test : pd.Series
            Test data

        Returns:
        --------
        results : pd.DataFrame
            Comparison of all models
        """
        print("\n" + "="*70)
        print("TRAINING BASELINE MODELS")
        print("="*70)

        # 1. Naive
        self.naive_forecast(train, test)

        # 2. Moving Averages (different windows)
        for window in [3, 7, 14, 28]:
            self.moving_average_forecast(train, test, window=window)

        # 3. Seasonal Naive
        self.seasonal_naive_forecast(train, test, season_length=7)

        # 4. ARIMA
        self.arima_forecast(train, test, order=(1, 1, 1))

        # 5. SARIMA
        self.sarima_forecast(train, test, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))

        # Compare results
        results = pd.DataFrame(self.metrics).T
        results = results.sort_values('MAPE')

        print("\n" + "="*70)
        print("BASELINE MODEL COMPARISON")
        print("="*70)
        print(results.to_string())
        print("\n" + "="*70)

        # Find best model
        best_model = results['MAPE'].idxmin()
        print(f"\n🏆 Best Model: {best_model.upper()}")
        print(f"   MAPE: {results.loc[best_model, 'MAPE']:.2f}%")
        print(f"   RMSE: ${results.loc[best_model, 'RMSE']:.2f}")
        print("="*70)

        return results

    def get_forecast(self, model_name):
        """Get forecast for a specific model"""
        return self.forecasts.get(model_name)

    def get_metrics(self, model_name):
        """Get metrics for a specific model"""
        return self.metrics.get(model_name)


if __name__ == "__main__":
    # Test baseline models
    print("Testing Baseline Models...\n")

    # Load data
    df = pd.read_csv('../../data/processed/daily_revenue.csv',
                     index_col='date', parse_dates=True)

    revenue = df['revenue']

    # Split data (80/20 for quick test)
    n = len(revenue)
    train_size = int(n * 0.8)
    train = revenue.iloc[:train_size]
    test = revenue.iloc[train_size:]

    print(f"Train: {len(train)} samples")
    print(f"Test: {len(test)} samples")

    # Train all baselines
    baselines = BaselineModels()
    results = baselines.train_all_baselines(train, test)
