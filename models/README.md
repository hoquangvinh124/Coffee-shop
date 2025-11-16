# Models Directory

This directory contains trained models. These files are gitignored to keep the repository size small.

## Files

- `lightgbm_model.pkl` - Trained LightGBM model for ML regression approach
- `lstm_model.keras` - Trained LSTM model for time series forecasting
- `lstm_scaler.pkl` - MinMaxScaler used for LSTM data normalization

## How to Generate Models

### ML Regression Model (LightGBM)

```bash
python predict_future.py 2023-07-15
```

This will automatically train and save the model on first run.

### LSTM Model

```bash
python lstm_forecasting.py
```

This will:
1. Load data from `data/processed/daily_revenue.csv`
2. Train LSTM model (2-layer with dropout)
3. Save model to `models/lstm_model.keras`
4. Save scaler to `models/lstm_scaler.pkl`
5. Generate performance visualization

**Training time**: ~2 minutes on CPU

## Model Sizes

- LightGBM: ~120 KB
- LSTM model: ~415 KB
- LSTM scaler: ~0.5 KB

**Total**: ~535 KB

## Usage

Models are automatically loaded by prediction scripts:
- `predict_future.py` - Uses LightGBM model
- `predict_lstm.py` - Uses LSTM model

If models don't exist, the scripts will train them automatically.
