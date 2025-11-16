"""
Coffee Shop Revenue Prediction Script
Use the trained model to make predictions on new data
"""

import pandas as pd
import pickle
import numpy as np

print("=" * 80)
print("COFFEE SHOP REVENUE PREDICTOR")
print("=" * 80)

# Load model and scaler
print("\n[1] Loading model...")
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/model_info.pkl', 'rb') as f:
    model_info = pickle.load(f)

print(f"✓ Loaded: {model_info['model_name']}")
print(f"  R²: {model_info['metrics']['R2']:.4f}")
print(f"  MAPE: {model_info['metrics']['MAPE']:.2f}%")

# Get feature names
feature_names = model_info['feature_names']
print(f"\n[2] Required features: {len(feature_names)}")
for i, feat in enumerate(feature_names, 1):
    print(f"  {i}. {feat}")

# Example prediction
print("\n" + "=" * 80)
print("EXAMPLE PREDICTION")
print("=" * 80)

# Create example input
example_data = pd.DataFrame({
    'Number_of_Customers_Per_Day': [300],
    'Average_Order_Value': [7.5],
    'Operating_Hours_Per_Day': [12],
    'Number_of_Employees': [8],
    'Marketing_Spend_Per_Day': [250.0],
    'Location_Foot_Traffic': [600]
})

print("\nInput:")
for col in example_data.columns:
    print(f"  {col}: {example_data[col].values[0]}")

# Make prediction
prediction = model.predict(example_data)[0]

print(f"\n💰 Predicted Daily Revenue: ${prediction:.2f}")

# Show feature importance
print("\n" + "=" * 80)
print("FEATURE IMPORTANCE")
print("=" * 80)

if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
    feature_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    print("\nMost Important Features:")
    for i, row in feature_imp.iterrows():
        bar_length = int(row['Importance'] * 50)
        bar = '█' * bar_length
        print(f"  {row['Feature']:<35} {bar} {row['Importance']:.4f}")

# Instructions for batch prediction
print("\n" + "=" * 80)
print("HOW TO USE FOR BATCH PREDICTIONS")
print("=" * 80)

print("""
1. Create a CSV file with the required columns:
   - Number_of_Customers_Per_Day
   - Average_Order_Value
   - Operating_Hours_Per_Day
   - Number_of_Employees
   - Marketing_Spend_Per_Day
   - Location_Foot_Traffic

2. Run the following code:

import pandas as pd
import pickle

# Load model
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load your data
df = pd.read_csv('your_data.csv')

# Make predictions
predictions = model.predict(df)

# Add predictions to dataframe
df['Predicted_Revenue'] = predictions

# Save results
df.to_csv('predictions.csv', index=False)

3. Done! Check predictions.csv for results.
""")

print("=" * 80)
