from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import pickle
import pandas as pd
import numpy as np
from typing import List, Optional

# Load the trained Prophet model
MODEL_PATH = "ml-models/revenue_prediction.pkl"
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
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
        "model_loaded": model is not None,
        "endpoints": {
            "health": "/health",
            "forecast": "/forecast",
            "forecast_range": "/forecast/range"
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
