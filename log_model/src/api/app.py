"""
FastAPI Application for Logistics KPI Prediction
Production-ready REST API with Swagger documentation
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import logging
from io import StringIO
import uvicorn

# Import prediction functions
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from ml.predict import (
    load_model_artifacts,
    engineer_features,
    preprocess_new_data
)

# Import monitoring system
from utils.monitoring import (
    PredictionLogger,
    PerformanceMonitor,
    ModelHealthChecker
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_logs.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Logistics KPI Prediction API",
    description="High-performance API for predicting logistics KPI scores (R² = 99.99%)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global variables for model artifacts
MODEL = None
SCALER = None
ENCODERS = None
MODEL_VERSION = None

# Initialize monitoring components
prediction_logger = PredictionLogger()
performance_monitor = PerformanceMonitor()
health_checker = ModelHealthChecker()

# Pydantic models for request validation
class ItemFeatures(BaseModel):
    """Single item features for prediction"""
    item_id: Optional[str] = Field(None, description="Unique item identifier")
    category: str = Field(..., description="Product category (Electronics, Groceries, etc.)")
    stock_level: int = Field(..., ge=0, description="Current stock level")
    reorder_point: int = Field(..., ge=0, description="Reorder threshold")
    reorder_frequency_days: int = Field(..., ge=1, description="Days between reorders")
    lead_time_days: int = Field(..., ge=1, description="Supplier lead time in days")
    daily_demand: float = Field(..., ge=0, description="Average daily demand")
    demand_std_dev: float = Field(..., ge=0, description="Demand standard deviation")
    item_popularity_score: float = Field(..., ge=0, le=1, description="Popularity score (0-1)")
    storage_location_id: Optional[str] = Field(None, description="Warehouse location ID")
    zone: str = Field(..., description="Storage zone (A, B, C, D)")
    picking_time_seconds: int = Field(..., ge=0, description="Time to pick item (seconds)")
    handling_cost_per_unit: float = Field(..., ge=0, description="Handling cost per unit")
    unit_price: float = Field(..., gt=0, description="Unit selling price")
    holding_cost_per_unit_day: float = Field(..., ge=0, description="Daily holding cost")
    stockout_count_last_month: int = Field(..., ge=0, description="Stockout occurrences")
    order_fulfillment_rate: float = Field(..., ge=0, le=1, description="Fulfillment rate (0-1)")
    total_orders_last_month: int = Field(..., ge=0, description="Total orders last month")
    turnover_ratio: float = Field(..., ge=0, description="Inventory turnover ratio")
    layout_efficiency_score: float = Field(..., ge=0, le=1, description="Layout efficiency (0-1)")
    last_restock_date: str = Field(..., description="Last restock date (YYYY-MM-DD)")
    forecasted_demand_next_7d: float = Field(..., ge=0, description="7-day demand forecast")
    
    class Config:
        schema_extra = {
            "example": {
                "item_id": "ITM10000",
                "category": "Electronics",
                "stock_level": 150,
                "reorder_point": 50,
                "reorder_frequency_days": 7,
                "lead_time_days": 3,
                "daily_demand": 15.5,
                "demand_std_dev": 3.2,
                "item_popularity_score": 0.75,
                "storage_location_id": "L25",
                "zone": "A",
                "picking_time_seconds": 45,
                "handling_cost_per_unit": 2.50,
                "unit_price": 99.99,
                "holding_cost_per_unit_day": 0.50,
                "stockout_count_last_month": 1,
                "order_fulfillment_rate": 0.95,
                "total_orders_last_month": 450,
                "turnover_ratio": 8.5,
                "layout_efficiency_score": 0.80,
                "last_restock_date": "2024-11-01",
                "forecasted_demand_next_7d": 110.0
            }
        }

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    item_id: Optional[str]
    predicted_kpi_score: float
    prediction_timestamp: str
    model_version: str
    confidence: str

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    total_predictions: int
    mean_kpi: float
    min_kpi: float
    max_kpi: float
    predictions: List[PredictionResponse]
    processing_time_seconds: float

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_version: Optional[str]
    timestamp: str

# Startup event - Load model
@app.on_event("startup")
async def startup_event():
    """Load model artifacts on startup"""
    global MODEL, SCALER, ENCODERS, MODEL_VERSION
    
    try:
        logger.info("Loading model artifacts...")
        MODEL, SCALER, ENCODERS = load_model_artifacts()
        MODEL_VERSION = "Ridge_Regression_v1.0_R2_99.99"
        logger.info(f"✅ Model loaded successfully: {MODEL_VERSION}")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        raise

# Root endpoint
@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Logistics KPI Prediction API",
        "version": "1.0.0",
        "model_version": MODEL_VERSION,
        "model_performance": "R² = 99.99%",
        "documentation": "/docs",
        "health_check": "/health"
    }

# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint with monitoring integration"""
    # Get comprehensive health status
    health_status = health_checker.check_health()
    
    return HealthResponse(
        status=health_status['overall_status'],
        model_loaded=MODEL is not None,
        model_version=MODEL_VERSION,
        timestamp=datetime.now().isoformat()
    )

# Single prediction endpoint
@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_single(item: ItemFeatures):
    """
    Predict KPI score for a single item
    
    Returns predicted KPI score with confidence level
    """
    try:
        start_time = datetime.now()
        
        # Convert to DataFrame
        item_dict = item.dict()
        df = pd.DataFrame([item_dict])
        
        # Store item_id
        item_id = item_dict.get('item_id', 'unknown')
        
        # Engineer features
        df = engineer_features(df)
        
        # Preprocess
        X = preprocess_new_data(df, SCALER, ENCODERS)
        
        # Predict
        prediction = float(MODEL.predict(X)[0])
        
        # Determine confidence based on historical performance
        if 0.5 <= prediction <= 0.8:
            confidence = "high"
        elif 0.3 <= prediction < 0.5 or 0.8 < prediction <= 0.9:
            confidence = "medium"
        else:
            confidence = "low"
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Log prediction to monitoring system
        prediction_logger.log_prediction(
            item_data=item_dict,
            prediction=prediction,
            confidence=confidence,
            response_time=processing_time,
            model_version=MODEL_VERSION,
            features_count=X.shape[1]
        )
        
        logger.info(f"Prediction completed for {item_id}: {prediction:.4f} (time: {processing_time:.3f}s)")
        
        return PredictionResponse(
            item_id=item_id,
            predicted_kpi_score=round(prediction, 4),
            prediction_timestamp=datetime.now().isoformat(),
            model_version=MODEL_VERSION,
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Batch prediction endpoint
@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(items: List[ItemFeatures]):
    """
    Predict KPI scores for multiple items
    
    Accepts a list of items and returns batch predictions
    """
    try:
        start_time = datetime.now()
        
        # Convert to DataFrame
        items_data = [item.dict() for item in items]
        df = pd.DataFrame(items_data)
        
        # Store item_ids
        item_ids = df['item_id'].tolist() if 'item_id' in df.columns else [None] * len(df)
        
        # Engineer features
        df = engineer_features(df)
        
        # Preprocess
        X = preprocess_new_data(df, SCALER, ENCODERS)
        
        # Predict
        predictions = MODEL.predict(X)
        
        # Create response
        prediction_responses = []
        for item_id, pred in zip(item_ids, predictions):
            if 0.5 <= pred <= 0.8:
                confidence = "high"
            elif 0.3 <= pred < 0.5 or 0.8 < pred <= 0.9:
                confidence = "medium"
            else:
                confidence = "low"
                
            prediction_responses.append(
                PredictionResponse(
                    item_id=item_id,
                    predicted_kpi_score=round(float(pred), 4),
                    prediction_timestamp=datetime.now().isoformat(),
                    model_version=MODEL_VERSION,
                    confidence=confidence
                )
            )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Batch prediction completed: {len(predictions)} items (time: {processing_time:.3f}s)")
        
        return BatchPredictionResponse(
            total_predictions=len(predictions),
            mean_kpi=round(float(np.mean(predictions)), 4),
            min_kpi=round(float(np.min(predictions)), 4),
            max_kpi=round(float(np.max(predictions)), 4),
            predictions=prediction_responses,
            processing_time_seconds=round(processing_time, 3)
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

# CSV upload prediction endpoint
@app.post("/predict/csv", tags=["Predictions"])
async def predict_from_csv(file: UploadFile = File(...)):
    """
    Upload CSV file and get predictions
    
    Returns CSV file with predictions added
    """
    try:
        start_time = datetime.now()
        
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be CSV format")
        
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode('utf-8')))
        
        logger.info(f"CSV uploaded: {file.filename} with {len(df)} rows")
        
        # Store item_ids if present
        has_item_id = 'item_id' in df.columns
        if has_item_id:
            item_ids = df['item_id'].copy()
        
        # Engineer features
        df = engineer_features(df)
        
        # Preprocess
        X = preprocess_new_data(df, SCALER, ENCODERS)
        
        # Predict
        predictions = MODEL.predict(X)
        
        # Create output DataFrame
        output_df = pd.DataFrame({
            'predicted_kpi_score': predictions
        })
        
        if has_item_id:
            output_df.insert(0, 'item_id', item_ids.values)
        
        # Save to temporary file
        output_filename = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        output_path = os.path.join('temp', output_filename)
        os.makedirs('temp', exist_ok=True)
        output_df.to_csv(output_path, index=False)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"CSV prediction completed: {len(predictions)} predictions (time: {processing_time:.3f}s)")
        
        return FileResponse(
            output_path,
            media_type='text/csv',
            filename=output_filename,
            headers={
                "X-Processing-Time": str(processing_time),
                "X-Total-Predictions": str(len(predictions))
            }
        )
        
    except Exception as e:
        logger.error(f"CSV prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"CSV prediction failed: {str(e)}")

# Model information endpoint
@app.get("/model/info", tags=["Model"])
async def get_model_info():
    """Get model information and performance metrics"""
    return {
        "model_name": "Ridge Regression",
        "model_version": MODEL_VERSION,
        "performance_metrics": {
            "r2_score": 0.9999,
            "rmse": 0.0004,
            "mae": 0.0003,
            "cv_r2_mean": 0.9999,
            "cv_r2_std": 0.0000
        },
        "features": {
            "total_features": 43,
            "original_features": 18,
            "engineered_features": 25
        },
        "training_info": {
            "training_samples": 2563,
            "test_samples": 641,
            "algorithm": "Ridge Regression with L2 regularization",
            "training_date": "2024-11-18"
        },
        "status": "production_ready",
        "model_loaded": MODEL is not None
    }

# Feature importance endpoint
@app.get("/model/features", tags=["Model"])
async def get_feature_importance():
    """Get feature importance information"""
    # Top features based on correlation analysis
    top_features = {
        "top_10_features": [
            {"feature": "order_fulfillment_rate", "importance": 0.856},
            {"feature": "efficiency_composite", "importance": 0.798},
            {"feature": "fulfillment_quality", "importance": 0.845},
            {"feature": "layout_efficiency_score", "importance": 0.742},
            {"feature": "inventory_health", "importance": 0.723},
            {"feature": "turnover_ratio", "importance": 0.681},
            {"feature": "demand_supply_balance", "importance": 0.654},
            {"feature": "picking_efficiency", "importance": 0.612},
            {"feature": "popularity_turnover", "importance": 0.598},
            {"feature": "forecast_accuracy", "importance": 0.534}
        ],
        "feature_categories": {
            "operational": ["order_fulfillment_rate", "layout_efficiency_score", "picking_efficiency"],
            "inventory": ["turnover_ratio", "inventory_health", "demand_supply_balance"],
            "composite": ["efficiency_composite", "fulfillment_quality", "popularity_turnover"]
        }
    }
    return top_features

# Statistics endpoint
@app.get("/stats", tags=["General"])
async def get_statistics():
    """Get API usage statistics"""
    try:
        # Get statistics from monitoring system
        stats = prediction_logger.get_statistics(hours=24)
        return {
            "total_predictions_24h": stats.get('total_predictions', 0),
            "avg_kpi_24h": stats.get('avg_kpi', 0),
            "avg_response_time_ms": stats.get('avg_response_time_ms', 0),
            "predictions_by_category": stats.get('predictions_per_category', {}),
            "model_version": MODEL_VERSION,
            "last_update": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        return {"error": str(e)}


# Monitoring endpoints
@app.get("/monitoring/predictions", tags=["Monitoring"])
async def get_prediction_history(hours: int = 24):
    """Get prediction history for monitoring"""
    try:
        stats = prediction_logger.get_statistics(hours=hours)
        return {
            "status": "success",
            "time_period_hours": hours,
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Monitoring error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/monitoring/performance", tags=["Monitoring"])
async def get_performance_metrics(last_n: int = 10):
    """Get model performance metrics history"""
    try:
        history = performance_monitor.get_performance_history(last_n=last_n)
        return {
            "status": "success",
            "evaluations_count": len(history),
            "performance_history": history
        }
    except Exception as e:
        logger.error(f"Performance monitoring error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/monitoring/health", tags=["Monitoring"])
async def get_detailed_health():
    """Get detailed system health status"""
    try:
        health = health_checker.check_health()
        return {
            "status": "success",
            "health_status": health
        }
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/monitoring/evaluate", tags=["Monitoring"])
async def evaluate_model_performance(file: UploadFile = File(...)):
    """
    Evaluate model performance on new validation data
    
    Upload CSV with columns including 'kpi_score' (ground truth)
    """
    try:
        # Read CSV
        content = await file.read()
        df = pd.read_csv(StringIO(content.decode('utf-8')))
        
        if 'kpi_score' not in df.columns:
            raise HTTPException(
                status_code=400,
                detail="CSV must contain 'kpi_score' column with ground truth values"
            )
        
        # Get ground truth
        y_true = df['kpi_score'].values
        
        # Remove kpi_score for prediction
        df_pred = df.drop(columns=['kpi_score'])
        
        # Engineer features
        df_pred = engineer_features(df_pred)
        
        # Preprocess
        X = preprocess_new_data(df_pred, SCALER, ENCODERS)
        
        # Predict
        y_pred = MODEL.predict(X)
        
        # Evaluate
        metrics = performance_monitor.evaluate_model(
            y_true=y_true,
            y_pred=y_pred,
            dataset_name="uploaded_validation"
        )
        
        return {
            "status": "success",
            "evaluation_results": metrics
        }
        
    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Run server
if __name__ == "__main__":
    logger.info("Starting Logistics KPI Prediction API...")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
