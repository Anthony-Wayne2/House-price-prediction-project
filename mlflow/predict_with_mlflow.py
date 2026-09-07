#!/usr/bin/env python3
"""
Prediction script using MLflow model.
"""
import os
import sys
import mlflow
import pandas as pd
import numpy as np

# MLflow configuration
mlflow.set_tracking_uri("http://mlflow-tracking-service:5000")

def load_production_model():
    """Load the production model from MLflow registry."""
    try:
        # Get the latest production model
        model = mlflow.pyfunc.load_model(
            "models:/HousePricePredictor/Production"
        )
        print("Loaded production model from MLflow registry")
        return model
    except Exception as e:
        print(f"Error loading production model: {e}")
        # Fallback to local model
        print("Falling back to local model...")
        import pickle
        with open('model.pickle', 'rb') as f:
            model = pickle.load(f)
        return model

def predict_with_mlflow(model, features):
    """Make prediction using MLflow model."""
    # Convert features to DataFrame
    df = pd.DataFrame([features])
    
    # Make prediction
    prediction = model.predict(df)
    return prediction[0]

if __name__ == "__main__":
    # Example usage
    features = {
        'total_sqft': 1200,
        'bath': 2,
        'bhk': 3,
        'location': 'Whitefield'
    }
    
    model = load_production_model()
    price = predict_with_mlflow(model, features)
    print(f"Predicted price: ₹{price:,.2f}")
