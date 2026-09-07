#!/usr/bin/env python3
"""
Simple MLflow predictor without Unicode issues.
"""
import mlflow
import pandas as pd
import numpy as np

def predict_price(total_sqft, bath, bhk, location):
    """Make a prediction using the MLflow model."""
    # Set tracking URI
    mlflow.set_tracking_uri('sqlite:///mlflow/mlflow.db')
    
    # Load model from production
    model = mlflow.pyfunc.load_model('models:/HousePricePredictor/Production')
    
    # Create DataFrame with proper types
    data = pd.DataFrame({
        'total_sqft': pd.Series([float(total_sqft)], dtype='float64'),
        'bath': pd.Series([float(bath)], dtype='float64'),
        'bhk': pd.Series([int(bhk)], dtype='int64'),
        'location': pd.Series([str(location)], dtype='string')
    })
    
    # Make prediction
    prediction = model.predict(data)
    return prediction[0]

if __name__ == "__main__":
    # Test predictions
    test_cases = [
        (1200, 2, 3, 'Whitefield'),
        (1500, 3, 4, 'Koramangala'),
        (800, 1, 2, 'HSR Layout'),
        (2000, 4, 5, 'Indira Nagar')
    ]
    
    print("=" * 60)
    print("MLflow Model Predictions")
    print("=" * 60)
    
    for total_sqft, bath, bhk, location in test_cases:
        try:
            price = predict_price(total_sqft, bath, bhk, location)
            print(f"Location: {location}")
            print(f"  Sqft: {total_sqft}, Baths: {bath}, BHK: {bhk}")
            print(f"  Predicted Price: Rs {price:,.2f}")
            print("-" * 40)
        except Exception as e:
            print(f"Error for {location}: {e}")
    
    print("=" * 60)
