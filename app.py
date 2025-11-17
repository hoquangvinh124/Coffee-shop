from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import pickle
import pandas as pd
import numpy as np
from typing import List, Optional
import os

# ============================================================================
# Store Revenue Predictor Class
# ============================================================================

class StoreRevenuePredictor:
    """Class to handle revenue prediction for stores"""

    def __init__(self, models_dir=None, metadata_file=None):
        """Initialize the predictor"""
        # Get absolute paths
        base_dir = os.path.dirname(os.path.abspath(__file__))

        if models_dir is None:
            models_dir = os.path.join(base_dir, 'ml-models', 'store_models')
        if metadata_file is None:
            metadata_file = os.path.join(base_dir, 'ml-models', 'store_models', 'stores_metadata.csv')

        self.models_dir = models_dir
        self.metadata = pd.read_csv(metadata_file)
        self.loaded_models = {}

    def get_store_info(self, store_nbr):
        """Get information about a specific store"""
        store_data = self.metadata[self.metadata['store_nbr'] == store_nbr]
        if len(store_data) == 0:
            raise ValueError(f"Store {store_nbr} not found in metadata")
        return store_data.iloc[0].to_dict()

    def list_stores(self, top_n=None, sort_by='historical_total_revenue'):
        """List available stores"""
        df = self.metadata.sort_values(sort_by, ascending=False)
        if top_n:
            df = df.head(top_n)
        return df[['store_nbr', 'city', 'type', 'historical_avg_daily', 'forecast_avg_daily', 'growth_percent']]

    def load_model(self, store_nbr):
        """Load a trained model for a specific store"""
        if store_nbr in self.loaded_models:
            return self.loaded_models[store_nbr]

        store_info = self.get_store_info(store_nbr)
        model_file = store_info['model_file']

        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")

        with open(model_file, 'rb') as f:
            model = pickle.load(f)

        self.loaded_models[store_nbr] = model
        return model

    def predict(self, store_nbr, periods=30, freq='D'):
        """Predict revenue for a specific store"""
        model = self.load_model(store_nbr)
        store_info = self.get_store_info(store_nbr)

        # Create future dataframe
        future = model.make_future_dataframe(periods=periods, freq=freq)
        forecast = model.predict(future)

        # Get only future predictions
        last_date = pd.to_datetime(store_info['date_to'])
        future_forecast = forecast[forecast['ds'] > last_date].copy()

        # Add store information
        future_forecast['store_nbr'] = store_nbr
        future_forecast['city'] = store_info['city']
        future_forecast['type'] = store_info['type']

        return future_forecast[['ds', 'store_nbr', 'city', 'type', 'yhat', 'yhat_lower', 'yhat_upper']]

    def predict_date_range(self, store_nbr, start_date, end_date):
        """Predict revenue for a specific date range"""
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        store_info = self.get_store_info(store_nbr)
        last_historical_date = pd.to_datetime(store_info['date_to'])

        # Calculate periods needed
        if start_date <= last_historical_date:
            start_date = last_historical_date + timedelta(days=1)

        periods = (end_date - last_historical_date).days

        if periods <= 0:
            raise ValueError("End date must be after the last historical date")

        # Predict
        forecast = self.predict(store_nbr, periods=periods)

        # Filter by date range
        mask = (forecast['ds'] >= start_date) & (forecast['ds'] <= end_date)
        return forecast[mask]

    def get_summary(self, store_nbr, periods=30):
        """Get prediction summary for a store"""
        forecast = self.predict(store_nbr, periods=periods)
        store_info = self.get_store_info(store_nbr)

        summary = {
            'store_nbr': store_nbr,
            'city': store_info['city'],
            'type': store_info['type'],
            'forecast_start': forecast['ds'].min(),
            'forecast_end': forecast['ds'].max(),
            'forecast_days': len(forecast),
            'avg_daily_forecast': forecast['yhat'].mean(),
            'total_forecast': forecast['yhat'].sum(),
            'min_forecast': forecast['yhat'].min(),
            'max_forecast': forecast['yhat'].max(),
            'historical_avg_daily': store_info['historical_avg_daily'],
            'growth_percent': ((forecast['yhat'].mean() - store_info['historical_avg_daily']) / store_info['historical_avg_daily'] * 100)
        }

        return summary

# ============================================================================
# FastAPI Application
# ============================================================================

# Load the trained Prophet model (overall system)
MODEL_PATH = "ml-models/revenue_prediction.pkl"
model = None
store_predictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model, store_predictor
    try:
        # Load overall model
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print("Overall model loaded successfully!")

        # Initialize store predictor
        store_predictor = StoreRevenuePredictor()
        print(f"Store predictor initialized with {len(store_predictor.metadata)} stores!")

    except Exception as e:
        print(f"Error loading models: {e}")
        raise
    yield
    # Shutdown (cleanup if needed)
    print("Shutting down...")

app = FastAPI(
    title="Coffee Shop Sales Forecasting API",
    description="Prophet time series forecasting API for coffee shop sales",
    version="1.0.0",
    lifespan=lifespan
)

# Request/Response models
class ForecastRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "days": 30,
                "start_date": "2025-01-01"
            }
        }
    )
    
    days: int = 30
    start_date: Optional[str] = None

class ForecastResponse(BaseModel):
    date: str
    forecast: float
    lower_bound: float
    upper_bound: float

class BatchForecastResponse(BaseModel):
    forecast_start: str
    forecast_end: str
    total_days: int
    forecasts: List[ForecastResponse]
    summary: dict

@app.get("/")
async def root():
    return {
        "message": "Coffee Shop Sales Forecasting API",
        "status": "running",
        "models": {
            "overall_model_loaded": model is not None,
            "store_predictor_loaded": store_predictor is not None,
            "total_stores": len(store_predictor.metadata) if store_predictor else 0
        },
        "endpoints": {
            "health": "/health",
            "overall_forecast": "/forecast",
            "overall_forecast_range": "/forecast/range",
            "stores_list": "/stores",
            "store_info": "/stores/{store_nbr}",
            "store_forecast": "/stores/{store_nbr}/forecast",
            "store_forecast_range": "/stores/{store_nbr}/forecast/range",
            "top_stores": "/stores/top/{n}"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
    }

@app.post("/forecast", response_model=BatchForecastResponse)
async def create_forecast(request: ForecastRequest):
    """
    Generate sales forecast for specified number of days

    - **days**: Number of days to forecast (default: 30, max: 365)
    - **start_date**: Optional start date in YYYY-MM-DD format (default: today)
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if request.days <= 0 or request.days > 365:
        raise HTTPException(status_code=400, detail="Days must be between 1 and 365")

    try:
        # Determine start date
        if request.start_date:
            start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        else:
            start_date = datetime.now()

        # Create future dataframe
        future_dates = pd.date_range(
            start=start_date,
            periods=request.days,
            freq='D'
        )
        future_df = pd.DataFrame({'ds': future_dates})

        # Generate predictions
        forecast = model.predict(future_df)

        # Prepare response
        forecasts = []
        for _, row in forecast.iterrows():
            forecasts.append(ForecastResponse(
                date=row['ds'].strftime("%Y-%m-%d"),
                forecast=round(row['yhat'], 2),
                lower_bound=round(row['yhat_lower'], 2),
                upper_bound=round(row['yhat_upper'], 2)
            ))

        # Calculate summary statistics
        summary = {
            "avg_daily_forecast": round(forecast['yhat'].mean(), 2),
            "total_forecast": round(forecast['yhat'].sum(), 2),
            "min_forecast": round(forecast['yhat'].min(), 2),
            "max_forecast": round(forecast['yhat'].max(), 2),
            "std_forecast": round(forecast['yhat'].std(), 2)
        }

        return BatchForecastResponse(
            forecast_start=forecasts[0].date,
            forecast_end=forecasts[-1].date,
            total_days=len(forecasts),
            forecasts=forecasts,
            summary=summary
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecasting error: {str(e)}")

@app.get("/forecast/range")
async def forecast_date_range(
    start_date: str,
    end_date: str
):
    """
    Generate sales forecast for a specific date range

    - **start_date**: Start date in YYYY-MM-DD format
    - **end_date**: End date in YYYY-MM-DD format
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        if start >= end:
            raise HTTPException(status_code=400, detail="Start date must be before end date")

        days_diff = (end - start).days + 1

        if days_diff > 365:
            raise HTTPException(status_code=400, detail="Date range cannot exceed 365 days")

        # Create future dataframe
        future_dates = pd.date_range(start=start, end=end, freq='D')
        future_df = pd.DataFrame({'ds': future_dates})

        # Generate predictions
        forecast = model.predict(future_df)

        # Prepare response
        forecasts = []
        for _, row in forecast.iterrows():
            forecasts.append(ForecastResponse(
                date=row['ds'].strftime("%Y-%m-%d"),
                forecast=round(row['yhat'], 2),
                lower_bound=round(row['yhat_lower'], 2),
                upper_bound=round(row['yhat_upper'], 2)
            ))

        # Calculate summary statistics
        summary = {
            "avg_daily_forecast": round(forecast['yhat'].mean(), 2),
            "total_forecast": round(forecast['yhat'].sum(), 2),
            "min_forecast": round(forecast['yhat'].min(), 2),
            "max_forecast": round(forecast['yhat'].max(), 2),
            "std_forecast": round(forecast['yhat'].std(), 2)
        }

        return BatchForecastResponse(
            forecast_start=forecasts[0].date,
            forecast_end=forecasts[-1].date,
            total_days=len(forecasts),
            forecasts=forecasts,
            summary=summary
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecasting error: {str(e)}")

# ============================================================================
# STORE-LEVEL ENDPOINTS
# ============================================================================

@app.get("/stores")
async def list_stores(top_n: Optional[int] = None):
    """Get list of all stores or top N stores by revenue"""
    if store_predictor is None:
        raise HTTPException(status_code=500, detail="Store predictor not loaded")

    try:
        stores = store_predictor.list_stores(top_n=top_n)
        return {
            "total_stores": len(stores),
            "stores": stores.to_dict('records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stores/top/{n}")
async def get_top_stores(n: int):
    """Get top N stores by revenue"""
    if store_predictor is None:
        raise HTTPException(status_code=500, detail="Store predictor not loaded")

    if n <= 0 or n > 54:
        raise HTTPException(status_code=400, detail="N must be between 1 and 54")

    try:
        stores = store_predictor.list_stores(top_n=n)
        return {
            "count": len(stores),
            "stores": stores.to_dict('records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stores/{store_nbr}")
async def get_store_info(store_nbr: int):
    """Get information about a specific store"""
    if store_predictor is None:
        raise HTTPException(status_code=500, detail="Store predictor not loaded")

    try:
        info = store_predictor.get_store_info(store_nbr)
        return {
            "store_nbr": store_nbr,
            "city": info['city'],
            "state": info['state'],
            "type": info['type'],
            "cluster": info['cluster'],
            "historical_avg_daily": round(info['historical_avg_daily'], 2),
            "forecast_avg_daily": round(info['forecast_avg_daily'], 2),
            "growth_percent": round(info['growth_percent'], 2),
            "data_from": str(info['date_from']),
            "data_to": str(info['date_to'])
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stores/{store_nbr}/forecast")
async def forecast_store(store_nbr: int, days: int = 30):
    """Generate forecast for a specific store"""
    if store_predictor is None:
        raise HTTPException(status_code=500, detail="Store predictor not loaded")

    if days <= 0 or days > 730:
        raise HTTPException(status_code=400, detail="Days must be between 1 and 730")

    try:
        forecast = store_predictor.predict(store_nbr=store_nbr, periods=days)
        summary = store_predictor.get_summary(store_nbr=store_nbr, periods=days)

        # Prepare forecasts
        forecasts = []
        for _, row in forecast.iterrows():
            forecasts.append({
                "date": row['ds'].strftime("%Y-%m-%d"),
                "forecast": round(row['yhat'], 2),
                "lower_bound": round(row['yhat_lower'], 2),
                "upper_bound": round(row['yhat_upper'], 2)
            })

        return {
            "store_nbr": store_nbr,
            "city": summary['city'],
            "type": summary['type'],
            "forecast_start": str(summary['forecast_start']),
            "forecast_end": str(summary['forecast_end']),
            "forecast_days": days,
            "historical_avg_daily": round(summary['historical_avg_daily'], 2),
            "forecast_avg_daily": round(summary['avg_daily_forecast'], 2),
            "total_forecast": round(summary['total_forecast'], 2),
            "growth_percent": round(summary['growth_percent'], 2),
            "forecasts": forecasts
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stores/{store_nbr}/forecast/range")
async def forecast_store_range(store_nbr: int, start_date: str, end_date: str):
    """Generate forecast for a specific store and date range"""
    if store_predictor is None:
        raise HTTPException(status_code=500, detail="Store predictor not loaded")

    try:
        forecast = store_predictor.predict_date_range(
            store_nbr=store_nbr,
            start_date=start_date,
            end_date=end_date
        )

        # Prepare forecasts
        forecasts = []
        for _, row in forecast.iterrows():
            forecasts.append({
                "date": row['ds'].strftime("%Y-%m-%d"),
                "forecast": round(row['yhat'], 2),
                "lower_bound": round(row['yhat_lower'], 2),
                "upper_bound": round(row['yhat_upper'], 2)
            })

        return {
            "store_nbr": store_nbr,
            "city": forecast['city'].iloc[0],
            "type": forecast['type'].iloc[0],
            "start_date": start_date,
            "end_date": end_date,
            "total_days": len(forecast),
            "total_forecast": round(forecast['yhat'].sum(), 2),
            "avg_daily_forecast": round(forecast['yhat'].mean(), 2),
            "min_daily": round(forecast['yhat'].min(), 2),
            "max_daily": round(forecast['yhat'].max(), 2),
            "forecasts": forecasts
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
