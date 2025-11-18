"""
Revenue Predictor Module
Provides interface to Prophet-based revenue forecasting models
"""
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import StoreRevenuePredictor from app.py
from app import StoreRevenuePredictor

_predictor_instance = None

def get_predictor():
    """
    Get singleton instance of StoreRevenuePredictor
    
    Returns:
        StoreRevenuePredictor: Initialized predictor instance
    """
    global _predictor_instance
    
    if _predictor_instance is None:
        # Initialize with updated paths pointing to revenue_forecasting directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_dir, 'revenue_forecasting', 'ml-models', 'store_models')
        metadata_file = os.path.join(base_dir, 'revenue_forecasting', 'ml-models', 'store_models', 'stores_metadata.csv')
        
        try:
            _predictor_instance = StoreRevenuePredictor(
                models_dir=models_dir,
                metadata_file=metadata_file
            )
        except Exception as e:
            print(f"Warning: Could not initialize predictor: {e}")
            print(f"  Models dir: {models_dir}")
            print(f"  Metadata file: {metadata_file}")
            # Return a dummy predictor for development
            _predictor_instance = None
    
    return _predictor_instance
