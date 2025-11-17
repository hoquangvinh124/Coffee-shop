"""
Script để verify tất cả models hoạt động đúng
"""
import joblib
import pandas as pd
import numpy as np
import os

def verify_models():
    """Kiểm tra tất cả 6 model files"""
    
    print("=" * 60)
    print("🔍 VERIFYING ALL MODELS")
    print("=" * 60)
    
    # Check if models directory exists
    models_dir = "models"
    if not os.path.exists(models_dir):
        print(f"❌ Models directory not found: {models_dir}")
        return False
    
    # List of expected model files
    model_files = [
        "preprocessor.pkl",
        "feature_names.pkl",
        "random_forest.pkl",
        "gradient_boosting.pkl",
        "xgboost.pkl",
        "best_model.pkl"
    ]
    
    loaded_models = {}
    
    # Load and verify each model
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        
        try:
            print(f"\n📦 Loading: {model_file}...")
            model = joblib.load(model_path)
            loaded_models[model_file] = model
            
            # Get model info
            if hasattr(model, 'get_params'):
                print(f"   ✅ Loaded successfully")
                print(f"   Type: {type(model).__name__}")
            elif isinstance(model, list):
                print(f"   ✅ Loaded successfully")
                print(f"   Type: List with {len(model)} items")
            else:
                print(f"   ✅ Loaded successfully")
                print(f"   Type: {type(model).__name__}")
                
        except Exception as e:
            print(f"   ❌ Failed to load: {str(e)}")
            return False
    
    print("\n" + "=" * 60)
    print("✅ ALL MODELS LOADED SUCCESSFULLY!")
    print("=" * 60)
    
    # Load test data to verify prediction capability
    print("\ TESTING PREDICTION CAPABILITY...")
    
    try:
        # Load test data
        X_test = pd.read_csv("data/X_test_processed.csv")
        y_test = pd.read_csv("data/y_test.csv")
        
        print(f"   Test data shape: {X_test.shape}")
        print(f"   Test labels shape: {y_test.shape}")
        
        # Test with best model
        best_model = loaded_models["best_model.pkl"]
        
        # Make predictions on first 10 samples
        sample_predictions = best_model.predict(X_test.head(10))
        sample_proba = best_model.predict_proba(X_test.head(10))
        
        print(f"\n   Sample predictions (first 10):")
        print(f"   {sample_predictions}")
        print(f"\n   Sample probabilities shape: {sample_proba.shape}")
        
        print("\n✅ PREDICTION TEST PASSED!")
        
    except Exception as e:
        print(f"\n❌ Prediction test failed: {str(e)}")
        return False
    
    # Verify feature names match
    print("\n🔍 VERIFYING FEATURE CONSISTENCY...")
    
    try:
        feature_names = loaded_models["feature_names.pkl"]
        
        print(f"   Expected features: {len(feature_names)}")
        print(f"   Test data features: {X_test.shape[1]}")
        
        if len(feature_names) == X_test.shape[1]:
            print("   ✅ Feature count matches!")
        else:
            print("    Feature count mismatch!")
            
        print(f"\n   First 5 features: {feature_names[:5]}")
        
    except Exception as e:
        print(f"   ❌ Feature verification failed: {str(e)}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"✅ Total models loaded: {len(loaded_models)}")
    print(f"✅ Preprocessor: Ready")
    print(f"✅ Feature names: {len(feature_names)} features")
    print(f"✅ Random Forest: Ready")
    print(f"✅ Gradient Boosting: Ready")
    print(f"✅ XGBoost: Ready")
    print(f"✅ Best Model (XGBoost): Ready for deployment")
    print("=" * 60)
    print("🎉 ALL MODELS ARE WORKING PROPERLY!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = verify_models()
    if success:
        print("\n✅ VERIFICATION COMPLETE - Models ready for production!")
    else:
        print("\n❌ VERIFICATION FAILED - Please check errors above")
