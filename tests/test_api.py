"""
Unit tests for FastAPI application
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.main import app

client = TestClient(app)


class TestAPIEndpoints:
    """Test API endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "message" in data
    
    def test_model_info_endpoint(self):
        """Test model info endpoint"""
        response = client.get("/model/info")
        # May return 503 if model not loaded
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "model_type" in data
            assert "features" in data
    
    def test_predict_endpoint(self):
        """Test prediction endpoint"""
        test_data = {
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
        
        response = client.post("/predict", json=test_data)
        # May return 503 if model not loaded
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "predicted_price" in data
            assert isinstance(data["predicted_price"], (int, float))
    
    def test_predict_invalid_data(self):
        """Test prediction with invalid data"""
        invalid_data = {
            "longitude": "invalid",  # Should be float
            "latitude": 37.88
        }
        
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    def test_batch_predict_endpoint(self):
        """Test batch prediction endpoint"""
        test_data = [
            {
                "longitude": -122.23,
                "latitude": 37.88,
                "housing_median_age": 41.0,
                "total_rooms": 880.0,
                "total_bedrooms": 129.0,
                "population": 322.0,
                "households": 126.0,
                "median_income": 8.3252,
                "ocean_proximity": "NEAR BAY"
            },
            {
                "longitude": -122.25,
                "latitude": 37.85,
                "housing_median_age": 35.0,
                "total_rooms": 1200.0,
                "total_bedrooms": 250.0,
                "population": 500.0,
                "households": 200.0,
                "median_income": 7.5,
                "ocean_proximity": "NEAR OCEAN"
            }
        ]
        
        response = client.post("/predict/batch", json=test_data)
        # May return 503 if model not loaded
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)
            assert len(data) == 2


class TestAPIValidation:
    """Test API input validation"""
    
    def test_missing_required_field(self):
        """Test prediction with missing required field"""
        incomplete_data = {
            "longitude": -122.23,
            "latitude": 37.88
            # Missing other required fields
        }
        
        response = client.post("/predict", json=incomplete_data)
        assert response.status_code == 422
    
    def test_wrong_data_type(self):
        """Test prediction with wrong data type"""
        wrong_type_data = {
            "longitude": -122.23,
            "latitude": 37.88,
            "housing_median_age": "not_a_number",  # Should be float
            "total_rooms": 880.0,
            "total_bedrooms": 129.0,
            "population": 322.0,
            "households": 126.0,
            "median_income": 8.3252,
            "ocean_proximity": "NEAR BAY"
        }
        
        response = client.post("/predict", json=wrong_type_data)
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])