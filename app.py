from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import numpy as np
import pickle
import joblib
from datetime import datetime, timedelta

# Initialize FastAPI app
app = FastAPI(
    title="Coffee Shop Revenue Prediction API",
    description="API for predicting daily revenue using ML models",
    version="1.0.0"
)

# Load models and scaler
try:
    with open('models/best_model_tuned.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/model_info_tuned.pkl', 'rb') as f:
        model_info = pickle.load(f)
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    model = None
    scaler = None
    model_info = None


# Pydantic models for request/response
class PredictionInput(BaseModel):
    number_of_customers_per_day: float = Field(..., description="Expected number of customers", gt=0)
    average_order_value: float = Field(..., description="Average order value in currency", gt=0)
    operating_hours_per_day: float = Field(..., description="Operating hours per day", gt=0, le=24)
    number_of_employees: int = Field(..., description="Number of employees", gt=0)
    marketing_spend_per_day: float = Field(..., description="Marketing spend per day", ge=0)
    location_foot_traffic: int = Field(..., description="Foot traffic at location", ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "number_of_customers_per_day": 150,
                "average_order_value": 7.5,
                "operating_hours_per_day": 12,
                "number_of_employees": 4,
                "marketing_spend_per_day": 100,
                "location_foot_traffic": 80
            }
        }


class PredictionOutput(BaseModel):
    predicted_revenue: float


class ModelInfo(BaseModel):
    model_type: str
    features: List[str]
    metrics: dict


# Helper function to extract features from date
def extract_features_from_date(date_str: str, day_of_week=None, day=None, month=None, week_of_year=None):
    """Extract time-based features from date string"""
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')

        features = {
            'DayOfWeek': day_of_week if day_of_week is not None else date_obj.weekday(),
            'Day': day if day is not None else date_obj.day,
            'Month': month if month is not None else date_obj.month,
            'WeekOfYear': week_of_year if week_of_year is not None else date_obj.isocalendar()[1]
        }

        return features, date_obj
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")


# Day of week names
DAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Coffee Shop Revenue Prediction API is running!",
        "model_loaded": model is not None
    }


@app.get("/model-info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the loaded model"""
    if model is None or model_info is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelInfo(
        model_type=model_info.get('model_name', 'Unknown'),
        features=model_info.get('features', []),
        metrics=model_info.get('metrics', {})
    )


@app.post("/predict", response_model=PredictionOutput)
async def predict_revenue(input_data: PredictionInput):
    """Predict revenue based on business metrics"""
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Build features dictionary in exact order as training
    features = {
        'Number_of_Customers_Per_Day': input_data.number_of_customers_per_day,
        'Average_Order_Value': input_data.average_order_value,
        'Operating_Hours_Per_Day': input_data.operating_hours_per_day,
        'Number_of_Employees': input_data.number_of_employees,
        'Marketing_Spend_Per_Day': input_data.marketing_spend_per_day,
        'Location_Foot_Traffic': input_data.location_foot_traffic
    }

    # Create DataFrame
    X = pd.DataFrame([features])

    # Scale and predict
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]

    return PredictionOutput(predicted_revenue=round(float(prediction), 2))


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model": {
            "loaded": model is not None,
            "type": model_info.get('model_name', 'Unknown') if model_info else None,
            "features_count": len(model_info.get('features', [])) if model_info else 0
        },
        "scaler_loaded": scaler is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
