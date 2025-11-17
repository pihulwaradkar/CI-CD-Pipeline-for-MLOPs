"""
FastAPI Application for House Price Prediction
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
from typing import List
import yaml
import os
import logging

from .schemas import PredictionRequest, PredictionResponse, HealthResponse, ModelInfo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
with open("config/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Initialize FastAPI app
app = FastAPI(
    title=config['api']['title'],
    version=config['api']['version'],
    description=config['api']['description']
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and preprocessor
model = None
preprocessor = None
feature_names = None


def load_artifacts():
    """Load model and preprocessor"""
    global model, preprocessor, feature_names
    
    model_type = config['model']['type']
    model_path = f"models/{model_type}_model.pkl"
    preprocessor_path = "models/preprocessor.pkl"
    
    try:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.warning(f"Model not found at {model_path}")
        
        if os.path.exists(preprocessor_path):
            preprocessor = joblib.load(preprocessor_path)
            logger.info(f"Preprocessor loaded from {preprocessor_path}")
        else:
            logger.warning(f"Preprocessor not found at {preprocessor_path}")
            
    except Exception as e:
        logger.error(f"Error loading artifacts: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting FastAPI application...")
    load_artifacts()


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint"""
    return {
        "status": "healthy",
        "message": "House Price Prediction API is running",
        "version": config['api']['version']
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    model_loaded = model is not None
    preprocessor_loaded = preprocessor is not None
    
    status = "healthy" if (model_loaded and preprocessor_loaded) else "unhealthy"
    
    return {
        "status": status,
        "message": f"Model loaded: {model_loaded}, Preprocessor loaded: {preprocessor_loaded}",
        "version": config['api']['version']
    }


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    model_type = config['model']['type']
    
    info = {
        "model_type": model_type,
        "model_class": type(model).__name__,
        "features": config['model']['features']['numerical'] + config['model']['features']['categorical'],
        "target": config['model']['target_column']
    }
    
    # Add model-specific info
    if hasattr(model, 'n_estimators'):
        info['n_estimators'] = model.n_estimators
    if hasattr(model, 'max_depth'):
        info['max_depth'] = model.max_depth
    
    return info


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make house price prediction
    
    Args:
        request: PredictionRequest with feature values
        
    Returns:
        PredictionResponse with predicted price
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model or preprocessor not loaded")
    
    try:
        # Convert request to dict (handles both Pydantic v1 and v2)
        try:
            input_dict = request.model_dump()  # Pydantic v2
        except AttributeError:
            input_dict = request.dict()  # Pydantic v1
        
        # Convert to DataFrame
        input_data = pd.DataFrame([input_dict])
        
        # Feature engineering (same as preprocessing)
        if 'total_rooms' in input_data.columns and 'households' in input_data.columns:
            input_data['rooms_per_household'] = input_data['total_rooms'] / input_data['households']
        
        if 'total_bedrooms' in input_data.columns and 'total_rooms' in input_data.columns:
            input_data['bedrooms_per_room'] = input_data['total_bedrooms'] / input_data['total_rooms']
        
        if 'population' in input_data.columns and 'households' in input_data.columns:
            input_data['population_per_household'] = input_data['population'] / input_data['households']
        
        # Handle infinite values
        input_data.replace([np.inf, -np.inf], 0, inplace=True)
        input_data.fillna(0, inplace=True)
        
        # Encode categorical variables
        if 'ocean_proximity' in input_data.columns:
            # One-hot encode (must match training)
            # Get unique categories from training
            ocean_categories = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
            for category in ocean_categories:
                if category != ocean_categories[0]:  # drop_first=True
                    col_name = f'ocean_proximity_{category}'
                    input_data[col_name] = 1 if input_data['ocean_proximity'].iloc[0] == category else 0
            
            input_data.drop('ocean_proximity', axis=1, inplace=True)
        
        # Scale features
        input_scaled = preprocessor.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Calculate confidence (for tree-based models)
        confidence = None
        if hasattr(model, 'estimators_'):
            try:
                predictions = np.array([tree.predict(input_scaled)[0] for tree in model.estimators_])
                std = np.std(predictions)
                mean = np.mean(predictions)
                if mean != 0:
                    confidence = float(1 - (std / abs(mean)))
                    confidence = max(0, min(1, confidence))  # Clamp between 0 and 1
            except:
                confidence = None
        
        return {
            "predicted_price": float(prediction),
            "confidence": confidence,
            "currency": "USD"
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(requests: List[PredictionRequest]):
    """
    Make batch predictions
    
    Args:
        requests: List of PredictionRequest objects
        
    Returns:
        List of PredictionResponse objects
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model or preprocessor not loaded")
    
    try:
        results = []
        for request in requests:
            result = await predict(request)
            results.append(result)
        
        return results
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.post("/reload")
async def reload_model():
    """Reload model and preprocessor"""
    try:
        load_artifacts()
        return {"message": "Model and preprocessor reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config['api']['host'],
        port=config['api']['port']
    )