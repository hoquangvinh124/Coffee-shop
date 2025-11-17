"""
Revenue Prediction Module
Direct model loading and prediction without API
"""
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import os


class RevenuePredictor:
    """Revenue predictor for stores"""

    def __init__(self):
        """Initialize predictor"""
        base_dir = Path(__file__).parent
        self.models_dir = base_dir / 'ml-models' / 'store_models'
        self.metadata_file = self.models_dir / 'stores_metadata.csv'
        self.overall_model_path = base_dir / 'ml-models' / 'revenue_prediction.pkl'

        # Load metadata
        self.metadata = pd.read_csv(self.metadata_file)
        self.loaded_models = {}
        self.overall_model = None

    def load_overall_model(self):
        """Load overall system model"""
        if self.overall_model is None:
            with open(self.overall_model_path, 'rb') as f:
                self.overall_model = pickle.load(f)
        return self.overall_model

    def load_store_model(self, store_nbr):
        """Load model for specific store"""
        if store_nbr in self.loaded_models:
            return self.loaded_models[store_nbr]

        store_data = self.metadata[self.metadata['store_nbr'] == store_nbr]
        if len(store_data) == 0:
            raise ValueError(f"Store {store_nbr} not found")

        model_file = store_data.iloc[0]['model_file']
        with open(model_file, 'rb') as f:
            model = pickle.load(f)

        self.loaded_models[store_nbr] = model
        return model

    def get_all_stores(self):
        """Get all stores info"""
        stores = []
        for _, row in self.metadata.iterrows():
            stores.append({
                'store_nbr': int(row['store_nbr']),
                'city': row['city'],
                'type': row['type'],
                'state': row['state'],
                'cluster': int(row['cluster']),
                'historical_avg_daily': float(row['historical_avg_daily']),
                'forecast_avg_daily': float(row['forecast_avg_daily']),
                'growth_percent': float(row['growth_percent'])
            })
        return stores

    def get_store_info(self, store_nbr):
        """Get specific store info"""
        store_data = self.metadata[self.metadata['store_nbr'] == store_nbr]
        if len(store_data) == 0:
            raise ValueError(f"Store {store_nbr} not found")

        row = store_data.iloc[0]
        return {
            'store_nbr': int(row['store_nbr']),
            'city': row['city'],
            'type': row['type'],
            'state': row['state'],
            'cluster': int(row['cluster']),
            'historical_avg_daily': float(row['historical_avg_daily']),
            'forecast_avg_daily': float(row['forecast_avg_daily']),
            'growth_percent': float(row['growth_percent']),
            'date_from': row['date_from'],
            'date_to': row['date_to']
        }

    def predict_overall(self, days):
        """Predict overall system revenue"""
        model = self.load_overall_model()

        start_date = datetime.now()
        future_dates = pd.date_range(start=start_date, periods=days, freq='D')
        future_df = pd.DataFrame({'ds': future_dates})

        forecast = model.predict(future_df)

        forecasts = []
        for _, row in forecast.iterrows():
            forecasts.append({
                'date': row['ds'].strftime("%Y-%m-%d"),
                'forecast': float(row['yhat']),
                'lower_bound': float(row['yhat_lower']),
                'upper_bound': float(row['yhat_upper'])
            })

        summary = {
            'avg_daily_forecast': float(forecast['yhat'].mean()),
            'total_forecast': float(forecast['yhat'].sum()),
            'min_forecast': float(forecast['yhat'].min()),
            'max_forecast': float(forecast['yhat'].max()),
            'std_forecast': float(forecast['yhat'].std())
        }

        return {
            'forecasts': forecasts,
            'summary': summary,
            'forecast_start': forecasts[0]['date'],
            'forecast_end': forecasts[-1]['date'],
            'total_days': len(forecasts)
        }

    def predict_store(self, store_nbr, days):
        """Predict store revenue"""
        model = self.load_store_model(store_nbr)
        store_info = self.get_store_info(store_nbr)

        # Create future dataframe
        future = model.make_future_dataframe(periods=days, freq='D')
        forecast = model.predict(future)

        # Get only future predictions
        last_date = pd.to_datetime(store_info['date_to'])
        future_forecast = forecast[forecast['ds'] > last_date].copy()

        forecasts = []
        for _, row in future_forecast.iterrows():
            forecasts.append({
                'date': row['ds'].strftime("%Y-%m-%d"),
                'forecast': float(row['yhat']),
                'lower_bound': float(row['yhat_lower']),
                'upper_bound': float(row['yhat_upper'])
            })

        avg_forecast = float(future_forecast['yhat'].mean())
        total_forecast = float(future_forecast['yhat'].sum())
        historical_avg = store_info['historical_avg_daily']
        growth = ((avg_forecast - historical_avg) / historical_avg * 100)

        return {
            'store_nbr': store_nbr,
            'city': store_info['city'],
            'type': store_info['type'],
            'forecasts': forecasts,
            'forecast_avg_daily': avg_forecast,
            'total_forecast': total_forecast,
            'historical_avg_daily': historical_avg,
            'growth_percent': float(growth),
            'forecast_start': forecasts[0]['date'] if forecasts else None,
            'forecast_end': forecasts[-1]['date'] if forecasts else None
        }

    def get_top_stores(self, n=10):
        """Get top N stores by forecast revenue"""
        stores = self.metadata.sort_values('forecast_avg_daily', ascending=False).head(n)
        result = []
        for _, row in stores.iterrows():
            result.append({
                'store_nbr': int(row['store_nbr']),
                'city': row['city'],
                'type': row['type'],
                'forecast_avg_daily': float(row['forecast_avg_daily']),
                'historical_avg_daily': float(row['historical_avg_daily']),
                'growth_percent': float(row['growth_percent'])
            })
        return result

    def get_bottom_stores(self, n=10):
        """Get bottom N stores by forecast revenue"""
        stores = self.metadata.sort_values('forecast_avg_daily', ascending=True).head(n)
        result = []
        for _, row in stores.iterrows():
            result.append({
                'store_nbr': int(row['store_nbr']),
                'city': row['city'],
                'type': row['type'],
                'forecast_avg_daily': float(row['forecast_avg_daily']),
                'historical_avg_daily': float(row['historical_avg_daily']),
                'growth_percent': float(row['growth_percent'])
            })
        return result


# Global instance
_predictor = None

def get_predictor():
    """Get or create predictor instance"""
    global _predictor
    if _predictor is None:
        _predictor = RevenuePredictor()
    return _predictor
