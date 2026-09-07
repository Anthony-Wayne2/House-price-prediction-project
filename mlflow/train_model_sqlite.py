#!/usr/bin/env python3
"""
MLflow training script using SQLite (no PostgreSQL dependency).
"""
import os
import sys
import pickle
import re
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# Use local SQLite backend
mlflow.set_tracking_uri("sqlite:///mlflow/mlflow.db")

# Create mlflow directory if it doesn't exist
os.makedirs("mlflow", exist_ok=True)

def load_data():
    """Load and preprocess the dataset."""
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Bengaluru_House_Data.csv')
    df = pd.read_csv(data_path)
    return df

def convert_sqft(sqft):
    """Convert sqft to float handling various formats."""
    if isinstance(sqft, (int, float)):
        return float(sqft)
    
    sqft = str(sqft).strip()
    
    # Handle ranges like '1000-1200'
    if '-' in sqft:
        parts = sqft.split('-')
        try:
            return (float(parts[0]) + float(parts[1])) / 2
        except:
            return float(parts[0])
    
    # Handle 'Sq. Meter' or 'Sq. Yards'
    sqft = sqft.lower().replace('sq. meter', '').replace('sq. yards', '').strip()
    sqft = re.sub(r'[^0-9.]', '', sqft)
    
    try:
        return float(sqft)
    except:
        return 0.0

def preprocess_data(df):
    """Preprocess the data for training."""
    df = df.dropna()
    
    # Convert total_sqft
    df['total_sqft'] = df['total_sqft'].apply(convert_sqft)
    
    # Remove rows with invalid sqft
    df = df[df['total_sqft'] > 0]
    
    # Extract BHK from size column
    def extract_bhk(size):
        try:
            return int(re.search(r'\d+', str(size)).group())
        except:
            return 1
    
    df['bhk'] = df['size'].apply(extract_bhk)
    
    # Select features
    features = ['total_sqft', 'bath', 'bhk', 'location']
    X = df[features]
    y = df['price'] * 100000  # Convert to actual price
    
    # Drop rows with invalid target
    df_clean = pd.concat([X, y.to_frame('price')], axis=1)
    df_clean = df_clean[df_clean['price'] > 0]
    
    X = df_clean[features]
    y = df_clean['price']
    
    return X, y

def build_pipeline():
    """Build the preprocessing pipeline."""
    numeric_features = ['total_sqft', 'bath', 'bhk']
    categorical_features = ['location']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
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
    print(f"Data shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    with mlflow.start_run() as run:
        print(f"Started MLflow run: {run.info.run_id}")
        
        mlflow.log_params({
            "test_size": 0.2,
            "random_state": 42,
            "model_type": "LinearRegression",
            "features": list(X.columns),
        })
        
        pipeline = build_pipeline()
        print("Training model...")
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        mlflow.log_metrics({
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2_score": r2,
        })
        
        print(f"R² Score: {r2:.4f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        
        signature = infer_signature(X_train, y_pred)
        
        mlflow.sklearn.log_model(
            pipeline,
            "house_price_model",
            signature=signature,
            registered_model_name="HousePricePredictor"
        )
        
        # Save model locally
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model.pickle')
        with open(model_path, 'wb') as f:
            pickle.dump(pipeline.named_steps['regressor'], f)
        
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
