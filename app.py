"""
FastAPI Application for Coffee Shop Revenue Prediction
Endpoint: POST /predict
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import pickle
import pandas as pd
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI(
    title="Coffee Shop Revenue Predictor API",
    description="Predict daily revenue based on operational metrics",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
try:
    # Try to load tuned model first
    try:
        with open('models/best_model_tuned.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/model_info_tuned.pkl', 'rb') as f:
            model_info = pickle.load(f)
        print("✓ Loaded tuned model")
    except FileNotFoundError:
        # Fall back to original model
        with open('models/best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        print("✓ Loaded original model")

    feature_names = model_info['feature_names']
    print(f"✓ Model: {model_info['model_name']}")
    print(f"  R²: {model_info['metrics'].get('R2', model_info['metrics'].get('R²', 'N/A'))}")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    model_info = None
    feature_names = []

# Pydantic models for request/response
class PredictionInput(BaseModel):
    """Input features for prediction"""
    Number_of_Customers_Per_Day: int = Field(..., ge=50, le=500, description="Number of customers (50-500)")
    Average_Order_Value: float = Field(..., ge=2.5, le=10.0, description="Average order value ($2.50-$10.00)")
    Operating_Hours_Per_Day: int = Field(..., ge=6, le=17, description="Operating hours (6-17)")
    Number_of_Employees: int = Field(..., ge=2, le=14, description="Number of employees (2-14)")
    Marketing_Spend_Per_Day: float = Field(..., ge=10.0, le=500.0, description="Marketing spend ($10-$500)")
    Location_Foot_Traffic: int = Field(..., ge=50, le=1000, description="Foot traffic (50-1000)")

    class Config:
        json_schema_extra = {
            "example": {
                "Number_of_Customers_Per_Day": 300,
                "Average_Order_Value": 7.5,
                "Operating_Hours_Per_Day": 12,
                "Number_of_Employees": 8,
                "Marketing_Spend_Per_Day": 250.0,
                "Location_Foot_Traffic": 600
            }
        }

class PredictionOutput(BaseModel):
    """Output prediction"""
    predicted_revenue: float = Field(..., description="Predicted daily revenue in USD")
    model_name: str = Field(..., description="Name of the model used")
    model_r2: float = Field(..., description="Model R² score")
    confidence: str = Field(..., description="Prediction confidence level")

class BatchPredictionInput(BaseModel):
    """Batch prediction input"""
    predictions: List[PredictionInput]

class HealthCheck(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_name: str
    model_metrics: dict

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Coffee Shop Revenue Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/docs": "API documentation",
            "/health": "Health check",
            "/predict": "Single prediction (POST)",
            "/predict/batch": "Batch prediction (POST)",
            "/model/info": "Model information"
        }
    }

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return HealthCheck(
        status="healthy",
        model_loaded=True,
        model_name=model_info['model_name'],
        model_metrics=model_info['metrics']
    )

@app.get("/model/info", response_model=dict)
async def get_model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_name": model_info['model_name'],
        "metrics": model_info['metrics'],
        "features": feature_names,
        "feature_count": len(feature_names)
    }

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """
    Predict daily revenue for a single day

    Returns:
    - predicted_revenue: Predicted revenue in USD
    - model_name: Name of the model used
    - model_r2: Model R² score
    - confidence: Confidence level based on input values
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert input to dataframe
        input_dict = input_data.model_dump()
        df = pd.DataFrame([input_dict])

        # Ensure correct column order
        df = df[feature_names]

        # Make prediction
        prediction = float(model.predict(df)[0])

        # Determine confidence level
        # High confidence if values are within typical ranges
        typical_customers = 200 <= input_data.Number_of_Customers_Per_Day <= 350
        typical_order = 5.0 <= input_data.Average_Order_Value <= 8.0

        if typical_customers and typical_order:
            confidence = "HIGH"
        elif typical_customers or typical_order:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        return PredictionOutput(
            predicted_revenue=round(prediction, 2),
            model_name=model_info['model_name'],
            model_r2=round(model_info['metrics'].get('R2', model_info['metrics'].get('R²', 0)), 4),
            confidence=confidence
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=List[PredictionOutput])
async def predict_batch(input_data: BatchPredictionInput):
    """
    Predict daily revenue for multiple days

    Accepts a list of prediction inputs and returns predictions for each
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert inputs to dataframe
        input_dicts = [item.model_dump() for item in input_data.predictions]
        df = pd.DataFrame(input_dicts)

        # Ensure correct column order
        df = df[feature_names]

        # Make predictions
        predictions = model.predict(df)

        # Create response
        results = []
        for i, (pred, input_item) in enumerate(zip(predictions, input_data.predictions)):
            # Determine confidence
            typical_customers = 200 <= input_item.Number_of_Customers_Per_Day <= 350
            typical_order = 5.0 <= input_item.Average_Order_Value <= 8.0

            if typical_customers and typical_order:
                confidence = "HIGH"
            elif typical_customers or typical_order:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"

            results.append(PredictionOutput(
                predicted_revenue=round(float(pred), 2),
                model_name=model_info['model_name'],
                model_r2=round(model_info['metrics'].get('R2', model_info['metrics'].get('R²', 0)), 4),
                confidence=confidence
            ))

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
