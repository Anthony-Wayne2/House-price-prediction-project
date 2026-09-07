"""
MLflow adapter for FastAPI - handles model loading and prediction.
"""
import pandas as pd
import numpy as np
import mlflow

class MLflowModelAdapter:
    def __init__(self, tracking_uri='sqlite:///mlflow/mlflow.db', model_name='HousePricePredictor'):
        self.tracking_uri = tracking_uri
        self.model_name = model_name
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the model from MLflow registry."""
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            self.model = mlflow.pyfunc.load_model(f'models:/{self.model_name}/Production')
            print(f"Loaded model: {self.model_name} from MLflow")
        except Exception as e:
            print(f"Failed to load MLflow model: {e}")
            raise
    
    def predict(self, total_sqft, bath, bhk, location):
        """Make a prediction."""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Create DataFrame with proper types
        data = pd.DataFrame({
            'total_sqft': pd.Series([float(total_sqft)], dtype='float64'),
            'bath': pd.Series([float(bath)], dtype='float64'),
            'bhk': pd.Series([int(bhk)], dtype='int64'),
            'location': pd.Series([str(location)], dtype='string')
        })
        
        prediction = self.model.predict(data)
        return prediction[0]

# Singleton instance
_adapter = None

def get_mlflow_adapter():
    """Get or create the MLflow adapter singleton."""
    global _adapter
    if _adapter is None:
        _adapter = MLflowModelAdapter()
    return _adapter
