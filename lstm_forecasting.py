"""
LSTM Time Series Forecasting for Coffee Shop Revenue
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("LSTM TIME SERIES FORECASTING - COFFEE SHOP REVENUE")
print("=" * 80)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load data
print("\n[1] Loading data...")
daily_revenue = pd.read_csv('data/processed/daily_revenue.csv')
daily_revenue['date'] = pd.to_datetime(daily_revenue['date'])
daily_revenue = daily_revenue.sort_values('date').reset_index(drop=True)

print(f"✓ Data loaded: {len(daily_revenue)} days")
print(f"  Date range: {daily_revenue['date'].min()} to {daily_revenue['date'].max()}")
print(f"  Mean revenue: ${daily_revenue['revenue'].mean():.2f}")

# Prepare data
revenue = daily_revenue['revenue'].values.reshape(-1, 1)
dates = daily_revenue['date'].values

# Normalize data (LSTM works better with normalized data)
print("\n[2] Normalizing data...")
scaler = MinMaxScaler(feature_range=(0, 1))
revenue_scaled = scaler.fit_transform(revenue)
print(f"✓ Scaled to range [0, 1]")

# Create sequences
def create_sequences(data, lookback=30):
    """
    Create sequences for LSTM
    lookback: number of previous days to use for prediction
    """
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

LOOKBACK = 30  # Use 30 days to predict next day

print(f"\n[3] Creating sequences (lookback={LOOKBACK} days)...")
X, y = create_sequences(revenue_scaled, LOOKBACK)
print(f"✓ Created {len(X)} sequences")
print(f"  X shape: {X.shape} (samples, lookback)")
print(f"  y shape: {y.shape}")

# Reshape for LSTM [samples, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))
print(f"✓ Reshaped X to: {X.shape} for LSTM")

# Split data (temporal split for time series)
train_size = int(0.8 * len(X))
val_size = int(0.1 * len(X))

X_train = X[:train_size]
y_train = y[:train_size]

X_val = X[train_size:train_size+val_size]
y_val = y[train_size:train_size+val_size]

X_test = X[train_size+val_size:]
y_test = y[train_size+val_size:]

print(f"\n[4] Data split:")
print(f"  Train: {len(X_train)} samples")
print(f"  Val:   {len(X_val)} samples")
print(f"  Test:  {len(X_test)} samples")

# Build LSTM model
print("\n[5] Building LSTM model...")
model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(LOOKBACK, 1)),
    Dropout(0.2),
    LSTM(50, activation='relu', return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

print(model.summary())

# Train model
print("\n[6] Training LSTM model...")
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

print("\n✅ Training complete!")

# Evaluate
print("\n[7] Evaluating model...")

# Predictions
y_train_pred = model.predict(X_train, verbose=0)
y_val_pred = model.predict(X_val, verbose=0)
y_test_pred = model.predict(X_test, verbose=0)

# Inverse transform to get actual revenue values
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
y_train_pred_actual = scaler.inverse_transform(y_train_pred)

y_val_actual = scaler.inverse_transform(y_val.reshape(-1, 1))
y_val_pred_actual = scaler.inverse_transform(y_val_pred)

y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
y_test_pred_actual = scaler.inverse_transform(y_test_pred)

# Calculate metrics
def calculate_metrics(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)

    print(f"\n{name}:")
    print(f"  MAE:  ${mae:.2f}")
    print(f"  RMSE: ${rmse:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R²:   {r2:.4f}")

    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}

train_metrics = calculate_metrics(y_train_actual, y_train_pred_actual, "Train Set")
val_metrics = calculate_metrics(y_val_actual, y_val_pred_actual, "Validation Set")
test_metrics = calculate_metrics(y_test_actual, y_test_pred_actual, "Test Set")

# Save model
print("\n[8] Saving model...")
model.save('models/lstm_model.keras')
print("✓ Model saved to models/lstm_model.keras")

# Save scaler
import pickle
with open('models/lstm_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Scaler saved to models/lstm_scaler.pkl")

# Visualization
print("\n[9] Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Training history
axes[0, 0].plot(history.history['loss'], label='Train Loss')
axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
axes[0, 0].set_title('Model Loss During Training')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss (MSE)')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Predictions vs Actual (Test Set)
axes[0, 1].plot(y_test_actual, label='Actual', alpha=0.7)
axes[0, 1].plot(y_test_pred_actual, label='Predicted', alpha=0.7)
axes[0, 1].set_title(f'Test Set: Actual vs Predicted (MAPE: {test_metrics["MAPE"]:.2f}%)')
axes[0, 1].set_xlabel('Sample')
axes[0, 1].set_ylabel('Revenue ($)')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Scatter plot
axes[1, 0].scatter(y_test_actual, y_test_pred_actual, alpha=0.5)
axes[1, 0].plot([y_test_actual.min(), y_test_actual.max()],
                [y_test_actual.min(), y_test_actual.max()], 'r--', lw=2)
axes[1, 0].set_title(f'Predicted vs Actual (R²: {test_metrics["R2"]:.4f})')
axes[1, 0].set_xlabel('Actual Revenue ($)')
axes[1, 0].set_ylabel('Predicted Revenue ($)')
axes[1, 0].grid(True)

# Error distribution
errors = y_test_actual - y_test_pred_actual
axes[1, 1].hist(errors, bins=30, edgecolor='black', alpha=0.7)
axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[1, 1].set_title('Prediction Error Distribution')
axes[1, 1].set_xlabel('Error ($)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('results/lstm_performance.png', dpi=150, bbox_inches='tight')
print("✓ Saved visualization to results/lstm_performance.png")

# Summary
print("\n" + "=" * 80)
print("LSTM MODEL SUMMARY")
print("=" * 80)
print(f"\n📊 Test Set Performance:")
print(f"   MAPE: {test_metrics['MAPE']:.2f}%")
print(f"   RMSE: ${test_metrics['RMSE']:.2f}")
print(f"   R²:   {test_metrics['R2']:.4f}")
print(f"\n✅ Model ready for future predictions!")
print("=" * 80)
