"""
Pydantic Schemas for API Request/Response Models
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List


class PredictionRequest(BaseModel):
    """Schema for prediction request"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "longitude": -122.23,
            "latitude": 37.88,
            "housing_median_age": 41.0,
            "total_rooms": 880.0,
            "total_bedrooms": 129.0,
            "population": 322.0,
            "households": 126.0,
            "median_income": 8.3252,
            "ocean_proximity": "NEAR BAY"
        }
    })
    
    longitude: float = Field(..., description="Longitude coordinate", example=-122.23)
    latitude: float = Field(..., description="Latitude coordinate", example=37.88)
    housing_median_age: float = Field(..., description="Median age of houses", example=41.0)
    total_rooms: float = Field(..., description="Total number of rooms", example=880.0)
    total_bedrooms: float = Field(..., description="Total number of bedrooms", example=129.0)
    population: float = Field(..., description="Population in the area", example=322.0)
    households: float = Field(..., description="Number of households", example=126.0)
    median_income: float = Field(..., description="Median income (in $10,000s)", example=8.3252)
    ocean_proximity: str = Field(..., description="Proximity to ocean", example="NEAR BAY")


class PredictionResponse(BaseModel):
    """Schema for prediction response"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "predicted_price": 452600.0,
            "confidence": 0.85,
            "currency": "USD"
        }
    })
    
    predicted_price: float = Field(..., description="Predicted house price")
    confidence: Optional[float] = Field(None, description="Prediction confidence score (0-1)")
    currency: str = Field(default="USD", description="Currency of the prediction")


class HealthResponse(BaseModel):
    """Schema for health check response"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "status": "healthy",
            "message": "House Price Prediction API is running",
            "version": "1.0.0"
        }
    })
    
    status: str = Field(..., description="Service status")
    message: str = Field(..., description="Status message")
    version: str = Field(..., description="API version")


class ModelInfo(BaseModel):
    """Schema for model information"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "model_type": "random_forest",
            "model_class": "RandomForestRegressor",
            "features": ["longitude", "latitude", "housing_median_age"],
            "target": "median_house_value"
        }
    })
    
    model_type: str = Field(..., description="Type of the model")
    model_class: str = Field(..., description="Model class name")
    features: List[str] = Field(..., description="List of features used")
    target: str = Field(..., description="Target variable name")