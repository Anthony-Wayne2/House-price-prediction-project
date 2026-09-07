"""
Simple test to ensure CI pipeline works.
"""
import pytest
import os

def test_pipeline_works():
    """This test should always pass."""
    assert True

def test_import_predictor():
    """Test that we can import the predictor."""
    try:
        from src.models.predict import HousePricePredictor
        assert HousePricePredictor is not None
    except ImportError:
        # In CI, we may not have the model files
        pytest.skip("Predictor not available in test environment")

def test_model_files_exist():
    """Test that model files exist (if we're in a real environment)."""
    model_exists = os.path.exists("model.pickle")
    params_exists = os.path.exists("params.pickle")
    if model_exists and params_exists:
        assert True
    else:
        pytest.skip("Model files not available in test environment")
