"""
Quick test script for Prophet forecasting
Tests the main logic from the notebook
"""
import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import datetime
import sys

# Set UTF-8 encoding for console output
if sys.platform == 'win32':
    import os
    os.system('chcp 65001 >nul 2>&1')

print("="*70)
print("TESTING PROPHET FORECASTING")
print("="*70)

# 1. Load data
print("\n1. Loading data...")
df = pd.read_csv('data/daily_sales_cafe.csv')
df['ds'] = pd.to_datetime(df['ds'])
print(f"   OK Loaded {len(df)} days of data")
print(f"   Date range: {df['ds'].min()} to {df['ds'].max()}")

# 2. Prepare training data
print("\n2. Preparing training data...")
train_df = df[['ds', 'y']].copy()
print(f"   OK Training data shape: {train_df.shape}")

# 3. Load holidays (optional)
print("\n3. Loading holidays...")
try:
    holidays = pd.read_csv('data/holidays_prepared.csv')
    holidays['ds'] = pd.to_datetime(holidays['ds'])
    holidays_prophet = holidays[['ds', 'holiday']].copy()
    holidays_prophet['lower_window'] = -2
    holidays_prophet['upper_window'] = 2
    print(f"   OK Loaded {len(holidays_prophet)} custom holidays")
    has_holidays = True
except FileNotFoundError:
    print("   WARNING Holidays file not found. Using Ecuador holidays only.")
    holidays_prophet = None
    has_holidays = False

# 4. Initialize Prophet
print("\n4. Initializing Prophet model...")
config = {
    'growth': 'linear',
    'changepoint_prior_scale': 0.05,
    'seasonality_mode': 'multiplicative',
    'yearly_seasonality': 20,
    'weekly_seasonality': 10,
    'daily_seasonality': False,
    'interval_width': 0.95
}

if has_holidays and holidays_prophet is not None:
    model = Prophet(holidays=holidays_prophet, **config)
else:
    model = Prophet(**config)

model.add_country_holidays(country_name='EC')
print("   OK Model initialized with configuration")

# 5. Train model
print("\n5. Training Prophet model...")
start_time = datetime.now()
model.fit(train_df)
training_time = (datetime.now() - start_time).total_seconds()
print(f"   OK Training completed in {training_time:.2f} seconds")

# 6. Generate forecast
print("\n6. Generating 8-year forecast...")
periods = 2920  # 8 years
future = model.make_future_dataframe(periods=periods, freq='D')
forecast = model.predict(future)
print(f"   OK Forecast generated: {len(forecast)} days")

# 7. Split forecast
train_end = train_df['ds'].max()
in_sample = forecast[forecast['ds'] <= train_end].copy()
out_sample = forecast[forecast['ds'] > train_end].copy()
print(f"   • In-sample: {len(in_sample)} days")
print(f"   • Out-of-sample: {len(out_sample)} days ({len(out_sample)/365:.1f} years)")

# 8. Evaluate
print("\n7. Evaluating model...")
eval_df = train_df.merge(
    in_sample[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
    on='ds',
    how='inner'
)

mae = np.mean(np.abs(eval_df['y'] - eval_df['yhat']))
mape = np.mean(np.abs((eval_df['y'] - eval_df['yhat']) / eval_df['y'])) * 100
rmse = np.sqrt(np.mean((eval_df['y'] - eval_df['yhat']) ** 2))
in_interval = ((eval_df['y'] >= eval_df['yhat_lower']) &
               (eval_df['y'] <= eval_df['yhat_upper']))
coverage = in_interval.mean() * 100

print("\n" + "="*70)
print("MODEL EVALUATION METRICS")
print("="*70)
print(f"MAE:  ${mae:,.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"RMSE: ${rmse:,.2f}")
print(f"Coverage (95% CI): {coverage:.2f}%")

# 9. Forecast summary
print("\n" + "="*70)
print("FORECAST SUMMARY")
print("="*70)
out_sample['year'] = out_sample['ds'].dt.year
yearly_forecast = out_sample.groupby('year').agg({
    'yhat': ['mean', 'sum']
}).reset_index()
yearly_forecast.columns = ['Year', 'Avg_Daily', 'Total']
yearly_forecast['Total_M'] = yearly_forecast['Total'] / 1e6

print("\nYearly Forecast:")
print(yearly_forecast.to_string(index=False))

first_year_avg = yearly_forecast['Avg_Daily'].iloc[0]
last_year_avg = yearly_forecast['Avg_Daily'].iloc[-1]
num_years = len(yearly_forecast) - 1
cagr = (last_year_avg / first_year_avg) ** (1 / num_years) - 1 if num_years > 0 else 0

print(f"\nProjected CAGR: {cagr:.2%}")
print(f"Total 8-Year Forecast: ${yearly_forecast['Total_M'].sum():.2f}M")

print("\n" + "="*70)
print("SUCCESS - TEST COMPLETED!")
print("="*70)
print("\nNext step: Open 'notebooks/prophet_forecasting.ipynb' in Jupyter to run the full analysis with visualizations.")
