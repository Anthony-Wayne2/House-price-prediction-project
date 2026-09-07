import pickle
import numpy as np
import os

class HousePricePredictor:
    def __init__(self, model_path='model.pickle', params_path='params.pickle'):
        """Initialize the predictor with trained model and parameters."""
        self.model = None
        self.params = None
        self.load_model(model_path, params_path)
    
    def load_model(self, model_path, params_path):
        """Load the trained model and parameters."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(params_path):
            raise FileNotFoundError(f"Params file not found: {params_path}")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(params_path, 'rb') as f:
            self.params = pickle.load(f)
    
    def predict(self, total_sq_feet, bathrooms, bedrooms, location):
        """
        Predict house price based on input features.
        
        Args:
            total_sq_feet (float): Total square feet area
            bathrooms (int): Number of bathrooms
            bedrooms (int): Number of bedrooms
            location (str): Location name
        
        Returns:
            float: Predicted price in INR
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Validate location
        if location not in self.params['columns']:
            raise ValueError(f"Location '{location}' not found. Available locations: {self.params['columns'][:5]}...")
        
        # Create feature vector
        prefix = self.params['prefix']
        features = [0] * (len(self.params['columns']) + prefix)
        
        # Set numeric features
        features[0] = float(total_sq_feet)
        features[1] = float(bathrooms)
        features[2] = float(bedrooms)
        
        # Set location one-hot encoding
        location_index = self.params['columns'].index(location)
        features[prefix + location_index] = 1
        
        # Predict
        price = self.model.predict([features])[0] * 100000
        return max(0, price)  # Ensure non-negative price
    
    def get_available_locations(self):
        """Get list of all available locations."""
        return self.params['columns'] if self.params else []
