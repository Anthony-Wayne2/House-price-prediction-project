#!/usr/bin/env python3
"""
MLflow training script for house price prediction model.
"""
import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# MLflow configuration
mlflow.set_tracking_uri("http://mlflow-tracking-service:5000")

def load_data():
    """Load and preprocess the dataset."""
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Bengaluru_House_Data.csv')
    df = pd.read_csv(data_path)
    return df

def preprocess_data(df):
    """Preprocess the data for training."""
    # Drop rows with missing values
    df = df.dropna()
    
    # Convert total_sqft to float (handling ranges)
    def convert_sqft(sqft):
        try:
            return float(sqft)
        except:
            return float(sqft.split('-')[0])  # Handle ranges like '1000-1200'
    
    df['total_sqft'] = df['total_sqft'].apply(convert_sqft)
    
    # Extract BHK from size column
    df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]))
    
    # Select features
    features = ['total_sqft', 'bath', 'bhk', 'location']
    X = df[features]
    y = df['price'] * 100000  # Convert to actual price
    
    return X, y

def build_pipeline():
    """Build the preprocessing pipeline."""
    numeric_features = ['total_sqft', 'bath', 'bhk']
    categorical_features = ['location']
    
    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Full pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    return pipeline

def train_model():
    """Train the model with MLflow tracking."""
    print("Loading data...")
    df = load_data()
    
    print("Preprocessing data...")
    X, y = preprocess_data(df)
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Start MLflow run
    with mlflow.start_run() as run:
        print(f"Started MLflow run: {run.info.run_id}")
        
        # Log parameters
        mlflow.log_params({
            "test_size": 0.2,
            "random_state": 42,
            "model_type": "LinearRegression",
            "features": list(X.columns),
        })
        
        # Build and train pipeline
        pipeline = build_pipeline()
        print("Training model...")
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metrics({
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2_score": r2,
        })
        
        print(f"R² Score: {r2:.4f}")
        print(f"RMSE: {rmse:.2f}")
        
        # Infer signature
        signature = infer_signature(X_train, y_pred)
        
        # Log model
        mlflow.sklearn.log_model(
            pipeline,
            "house_price_model",
            signature=signature,
            registered_model_name="HousePricePredictor"
        )
        
        # Save model locally for deployment
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model.pickle')
        with open(model_path, 'wb') as f:
            pickle.dump(pipeline.named_steps['regressor'], f)
        
        # Save params
        params_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'params.pickle')
        locations = X['location'].unique().tolist()
        with open(params_path, 'wb') as f:
            pickle.dump({'columns': locations, 'prefix': 3}, f)
        
        print(f"Model saved to: {model_path}")
        print(f"Params saved to: {params_path}")
        
        return run.info.run_id

if __name__ == "__main__":
    run_id = train_model()
    print(f"Training complete! Run ID: {run_id}")
