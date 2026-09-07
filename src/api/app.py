from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.predict import HousePricePredictor

# Initialize FastAPI app
app = FastAPI(
    title="Bengaluru House Price Prediction API",
    description="API for predicting house prices in Bengaluru",
    version="1.0.0"
)

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
            "/health": "GET - Health check"
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
    try:
        predictor = get_predictor()
        
        # Make prediction
        price = predictor.predict(
            total_sq_feet=request.total_sq_feet,
            bathrooms=request.bathrooms,
            bedrooms=request.bedrooms,
            location=request.location
        )
        
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
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
