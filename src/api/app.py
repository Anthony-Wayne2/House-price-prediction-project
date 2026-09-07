from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import sys
import os
import time
from prometheus_client import Counter, Histogram, generate_latest, REGISTRY
from starlette.responses import Response

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.predict import HousePricePredictor

# Initialize FastAPI app
app = FastAPI(
    title="Bengaluru House Price Prediction API",
    description="API for predicting house prices in Bengaluru",
    version="1.0.0"
)

# Define metrics
PREDICTION_COUNT = Counter('house_price_predictions_total', 'Total number of predictions made')
PREDICTION_LATENCY = Histogram('house_price_prediction_seconds', 'Prediction latency in seconds')
PREDICTION_ERRORS = Counter('house_price_prediction_errors_total', 'Total prediction errors')

# Initialize predictor (lazy loading)
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        # Look for model files in the project root
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'model.pickle')
        params_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'params.pickle')
        predictor = HousePricePredictor(model_path, params_path)
    return predictor

# Request/Response models
class PredictionRequest(BaseModel):
    total_sq_feet: float = Field(..., gt=0, description="Total square feet area")
    bathrooms: int = Field(..., gt=0, le=20, description="Number of bathrooms")
    bedrooms: int = Field(..., gt=0, le=20, description="Number of bedrooms")
    location: str = Field(..., description="Location name in Bengaluru")
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_sq_feet": 1200,
                "bathrooms": 2,
                "bedrooms": 3,
                "location": "Whitefield"
            }
        }

class PredictionResponse(BaseModel):
    price: float
    price_formatted: str
    location: str
    features: dict

class LocationsResponse(BaseModel):
    total_locations: int
    locations: list

# Endpoints
@app.get("/")
async def root():
    return {
        "message": "Bengaluru House Price Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Predict house price",
            "/locations": "GET - List available locations",
            "/health": "GET - Health check",
            "/metrics": "GET - Prometheus metrics"
        }
    }

@app.get("/health")
async def health_check():
    try:
        predictor = get_predictor()
        return {"status": "healthy", "model_loaded": predictor.model is not None}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.get("/locations", response_model=LocationsResponse)
async def get_locations():
    """Get all available locations."""
    try:
        predictor = get_predictor()
        locations = predictor.get_available_locations()
        return LocationsResponse(
            total_locations=len(locations),
            locations=locations
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictionResponse)
async def predict_price(request: PredictionRequest):
    """Predict house price based on input features."""
    start_time = time.time()
    try:
        predictor = get_predictor()
        
        # Make prediction
        price = predictor.predict(
            total_sq_feet=request.total_sq_feet,
            bathrooms=request.bathrooms,
            bedrooms=request.bedrooms,
            location=request.location
        )
        
        # Record metrics
        PREDICTION_COUNT.inc()
        
        # Format response
        return PredictionResponse(
            price=price,
            price_formatted=f"₹{price:,.2f}",
            location=request.location,
            features={
                "total_sq_feet": request.total_sq_feet,
                "bathrooms": request.bathrooms,
                "bedrooms": request.bedrooms
            }
        )
    except ValueError as e:
        PREDICTION_ERRORS.inc()
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        PREDICTION_ERRORS.inc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        duration = time.time() - start_time
        PREDICTION_LATENCY.observe(duration)

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(REGISTRY), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

# Add MLflow import (optional)
try:
    import mlflow
    import mlflow.pyfunc
    MLFLOW_ENABLED = True
except ImportError:
    MLFLOW_ENABLED = False
    print("MLflow not available, using local model")

def get_predictor_with_mlflow():
    """Try to load model from MLflow first, fallback to local."""
    global predictor
    
    if predictor is not None:
        return predictor
    
    if MLFLOW_ENABLED:
        try:
            mlflow.set_tracking_uri("http://mlflow-tracking-service:5000")
            # Try to load production model
            mlflow_model = mlflow.pyfunc.load_model(
                "models:/HousePricePredictor/Production"
            )
            # Wrap MLflow model to match our interface
            class MLflowPredictor:
                def __init__(self, model):
                    self.model = model
                    self.params = {'columns': [], 'prefix': 3}
                    self.model_loaded = True
                
                def predict(self, total_sq_feet, bathrooms, bedrooms, location):
                    import pandas as pd
                    df = pd.DataFrame([{
                        'total_sqft': total_sq_feet,
                        'bath': bathrooms,
                        'bhk': bedrooms,
                        'location': location
                    }])
                    return self.model.predict(df)[0]
                
                def get_available_locations(self):
                    # Use the locations from the original model
                    import pickle
                    with open('params.pickle', 'rb') as f:
                        params = pickle.load(f)
                    return params['columns']
            
            predictor = MLflowPredictor(mlflow_model)
            print("Using MLflow model")
            return predictor
        except Exception as e:
            print(f"MLflow model load failed: {e}")
    
    # Fallback to local model
    return get_predictor()
