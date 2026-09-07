"""
Pytest configuration for CI/CD pipeline.
"""
import pytest
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture(scope="session")
def test_data():
    """Provide test data for tests."""
    return {
        "valid_prediction": {
            "total_sq_feet": 1200,
            "bathrooms": 2,
            "bedrooms": 3,
            "location": "Whitefield"
        },
        "invalid_location": {
            "total_sq_feet": 1200,
            "bathrooms": 2,
            "bedrooms": 3,
            "location": "InvalidLocation"
        }
    }

# Skip tests that require model files if they don't exist
def pytest_configure(config):
    """Configure pytest."""
    model_exists = os.path.exists("model.pickle") and os.path.exists("params.pickle")
    if not model_exists:
        config.addinivalue_line(
            "markers",
            "skip_if_no_model: skip tests that require the model files"
        )
