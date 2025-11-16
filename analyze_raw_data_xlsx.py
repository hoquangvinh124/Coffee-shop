"""Comprehensive analysis of new Raw Data.xlsx"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ANALYZING NEW DATASET: Raw Data.xlsx")
print("="*80)

# Load the new XLSX file
file_path = '/home/user/Coffee-shop/Raw Data.xlsx'
print(f"\nLoading: {file_path}")

# Try to load it (might have multiple sheets)
try:
    xl = pd.ExcelFile(file_path)
    print(f"✓ Excel file loaded")
    print(f"  Sheets available: {xl.sheet_names}")

    # Load first sheet
    df = pd.read_excel(file_path, sheet_name=0)
    print(f"✓ Loaded sheet: {xl.sheet_names[0]}")
except Exception as e:
    print(f"✗ Error loading: {e}")
    exit(1)

print("\n" + "="*80)
print("1. DATASET OVERVIEW")
print("="*80)
print(f"Shape: {df.shape}")
print(f"\nColumns ({len(df.columns)}):")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col} ({df[col].dtype})")

print(f"\nFirst 10 rows:")
print(df.head(10))

print(f"\nData types:")
print(df.dtypes)

print(f"\nMissing values:")
print(df.isnull().sum())

print(f"\nBasic statistics:")
print(df.describe())

print("\n" + "="*80)
print("2. IDENTIFYING DATE AND REVENUE COLUMNS")
print("="*80)

# Try to identify date and revenue columns
date_col = None
revenue_col = None

# Look for date columns
for col in df.columns:
    if 'date' in col.lower() or 'time' in col.lower() or df[col].dtype == 'datetime64[ns]':
        print(f"  Potential date column: '{col}'")
        try:
            df[col] = pd.to_datetime(df[col])
            date_col = col
            print(f"    ✓ Successfully converted to datetime")
            break
        except:
            print(f"    ✗ Could not convert to datetime")

# Look for revenue/sales/amount columns
for col in df.columns:
    if any(keyword in col.lower() for keyword in ['revenue', 'sales', 'amount', 'price', 'total', 'money']):
        if pd.api.types.is_numeric_dtype(df[col]):
            print(f"  Potential revenue column: '{col}'")
            revenue_col = col
            break

if date_col is None:
    print("\n⚠️  Could not automatically identify date column")
    print("Available columns:", df.columns.tolist())
    # Try the first column
    print(f"\nTrying first column as date: '{df.columns[0]}'")
    try:
        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
        date_col = df.columns[0]
        print(f"✓ Success!")
    except:
        print(f"✗ Failed. Manual inspection needed.")

if revenue_col is None:
    print("\n⚠️  Could not automatically identify revenue column")
    print("Numeric columns:", [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])])
    # Try to find numeric columns
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        revenue_col = numeric_cols[0]
        print(f"\nTrying first numeric column as revenue: '{revenue_col}'")

print(f"\n✓ Using:")
print(f"  Date column: '{date_col}'")
print(f"  Revenue column: '{revenue_col}'")

if date_col is None or revenue_col is None:
    print("\n✗ Cannot proceed without date and revenue columns")
    print("\nPlease check the data structure:")
    print(df.head(20))
    exit(1)

print("\n" + "="*80)
print("3. DATA CLEANING & AGGREGATION")
print("="*80)

# Check if data is already aggregated or transaction-level
print(f"\nTotal rows: {len(df)}")
print(f"Unique dates: {df[date_col].nunique()}")

if len(df) == df[date_col].nunique():
    print("✓ Data appears to be ALREADY AGGREGATED (one row per date)")
    daily = df[[date_col, revenue_col]].copy()
    daily.columns = ['date', 'revenue']
else:
    print("✓ Data is TRANSACTION-LEVEL, aggregating to daily...")
    daily = df.groupby(date_col)[revenue_col].sum().reset_index()
    daily.columns = ['date', 'revenue']

daily = daily.sort_values('date').reset_index(drop=True)

print(f"\n✓ Daily revenue created:")
print(f"  Days: {len(daily)}")
print(f"  Date range: {daily['date'].min()} to {daily['date'].max()}")
print(f"  Total days: {(daily['date'].max() - daily['date'].min()).days + 1}")

print("\n" + "="*80)
print("4. REVENUE STATISTICS")
print("="*80)

print(f"Total revenue: ${daily['revenue'].sum():,.2f}")
print(f"Mean daily: ${daily['revenue'].mean():,.2f}")
print(f"Median daily: ${daily['revenue'].median():,.2f}")
print(f"Std daily: ${daily['revenue'].std():,.2f}")
print(f"Min daily: ${daily['revenue'].min():,.2f}")
print(f"Max daily: ${daily['revenue'].max():,.2f}")

# Check for data quality issues
print(f"\nData quality:")
print(f"  Zero revenue days: {(daily['revenue'] == 0).sum()}")
print(f"  Negative revenue days: {(daily['revenue'] < 0).sum()}")
print(f"  Missing values: {daily['revenue'].isnull().sum()}")

print("\n" + "="*80)
print("5. TREND ANALYSIS")
print("="*80)

# Calculate trend
daily['day_num'] = range(len(daily))
correlation = daily[['day_num', 'revenue']].corr().iloc[0, 1]
growth = (daily['revenue'].iloc[-1] - daily['revenue'].iloc[0]) / daily['revenue'].iloc[0] * 100

print(f"Correlation with time: {correlation:.4f}")
print(f"Overall growth: {growth:.2f}%")
print(f"First week mean: ${daily['revenue'].iloc[:7].mean():.2f}")
print(f"Last week mean: ${daily['revenue'].iloc[-7:].mean():.2f}")

if abs(correlation) > 0.3:
    print(f"\n⚠️  STRONG TREND DETECTED (|r|={abs(correlation):.2f})")
    print(f"    This may cause negative R² issues")
else:
    print(f"\n✓ WEAK/NO TREND (|r|={abs(correlation):.2f})")
    print(f"    Good sign for positive R²")

print("\n" + "="*80)
print("6. STATIONARITY TEST")
print("="*80)

adf_result = adfuller(daily['revenue'].values, autolag='AIC')
print(f"ADF Statistic: {adf_result[0]:.4f}")
print(f"p-value: {adf_result[1]:.4f}")
print(f"Critical values: {adf_result[4]}")

if adf_result[1] < 0.05:
    print(f"\n✓ STATIONARY (p < 0.05)")
    print(f"  Good for forecasting and R²!")
else:
    print(f"\n⚠️  NON-STATIONARY (p >= 0.05)")
    print(f"  May need differencing for ARIMA")

print("\n" + "="*80)
print("7. QUICK MODEL TEST - CHECKING R²")
print("="*80)

# Split data
train_size = int(0.8 * len(daily))
val_size = int(0.1 * len(daily))

train = daily['revenue'].iloc[:train_size]
val = daily['revenue'].iloc[train_size:train_size+val_size]
test = daily['revenue'].iloc[train_size+val_size:]

print(f"Split:")
print(f"  Train: {len(train)} days (mean=${train.mean():.2f})")
print(f"  Val:   {len(val)} days (mean=${val.mean():.2f})")
print(f"  Test:  {len(test)} days (mean=${test.mean():.2f})")
print(f"\nTrain-Test gap: {((test.mean() - train.mean()) / train.mean() * 100):.1f}%")

# Test baseline models
print(f"\n{'Model':<15} {'MAPE':<10} {'RMSE':<12} {'MAE':<12} {'R²':<10}")
print("-" * 70)

models = {}

# Naive
naive_pred = np.array([train.iloc[-1]] * len(test))
models['Naive'] = naive_pred

# Mean
mean_pred = np.array([train.mean()] * len(test))
models['Mean'] = mean_pred

# MA_3
ma3_pred = np.array([train.tail(3).mean()] * len(test))
models['MA_3'] = ma3_pred

# MA_7
ma7_pred = np.array([train.tail(7).mean()] * len(test))
models['MA_7'] = ma7_pred

best_r2 = -999
best_model = None

for name, pred in models.items():
    mape = mean_absolute_percentage_error(test.values, pred) * 100
    rmse = np.sqrt(mean_squared_error(test.values, pred))
    mae = mean_absolute_error(test.values, pred)
    r2 = r2_score(test.values, pred)

    print(f"{name:<15} {mape:>6.2f}%   ${rmse:>8.2f}   ${mae:>8.2f}   {r2:>8.4f}")

    if r2 > best_r2:
        best_r2 = r2
        best_model = name

print(f"\nBest model: {best_model} with R² = {best_r2:.4f}")

print("\n" + "="*80)
print("8. COMPARISON WITH OLD DATASETS")
print("="*80)

print("\nDATASET 1 - Coffee Shop Sales.xlsx (CURRENT):")
print("  • Transactions: 149,116")
print("  • Days: 181")
print("  • Mean daily: $3,860")
print("  • Growth: +124.4%")
print("  • R²: -0.03")
print("  • MAPE: 6.68% ✓ (EXCELLENT)")
print("  • Status: 90% complete")

print("\nDATASET 2 - Coffe_sales.csv:")
print("  • Transactions: 3,547")
print("  • Days: 381")
print("  • Mean daily: $295")
print("  • Growth: -48%")
print("  • R²: -0.88 ✗ (TERRIBLE)")
print("  • MAPE: 42.82% ✗ (FAIL)")

print(f"\nDATASET 3 - Raw Data.xlsx (NEW):")
print(f"  • Rows: {len(df)}")
print(f"  • Days: {len(daily)}")
print(f"  • Mean daily: ${daily['revenue'].mean():.2f}")
print(f"  • Growth: {growth:.2f}%")
print(f"  • R²: {best_r2:.4f}")
if best_r2 > 0:
    print(f"  • R² Status: ✓ POSITIVE!")
elif best_r2 > -0.1:
    print(f"  • R² Status: ~ MARGINAL (close to 0)")
else:
    print(f"  • R² Status: ✗ NEGATIVE")

# Calculate MAPE for best model
best_pred = models[best_model]
best_mape = mean_absolute_percentage_error(test.values, best_pred) * 100
print(f"  • MAPE: {best_mape:.2f}%", end="")
if best_mape < 15:
    print(" ✓ (MEETS TARGET)")
elif best_mape < 25:
    print(" ~ (ACCEPTABLE)")
else:
    print(" ✗ (POOR)")

print("\n" + "="*80)
print("9. RECOMMENDATION")
print("="*80)

# Scoring system
scores = {
    'Dataset 1 (Coffee Shop Sales)': 0,
    'Dataset 2 (Coffe_sales)': 0,
    'Dataset 3 (Raw Data)': 0
}

# Score based on R²
if best_r2 > 0.5:
    scores['Dataset 3 (Raw Data)'] += 3
    print("✓ Dataset 3 has excellent R² (>0.5)")
elif best_r2 > 0:
    scores['Dataset 3 (Raw Data)'] += 2
    print("✓ Dataset 3 has positive R²")
elif best_r2 > -0.1:
    scores['Dataset 3 (Raw Data)'] += 1
    scores['Dataset 1 (Coffee Shop Sales)'] += 1
    print("~ Dataset 3 R² similar to Dataset 1")
else:
    scores['Dataset 1 (Coffee Shop Sales)'] += 2
    print("✗ Dataset 3 R² worse than Dataset 1")

# Score based on MAPE
if best_mape < 10:
    scores['Dataset 3 (Raw Data)'] += 3
    print("✓ Dataset 3 has excellent MAPE (<10%)")
elif best_mape < 15:
    scores['Dataset 3 (Raw Data)'] += 2
    print("✓ Dataset 3 meets MAPE target (<15%)")
else:
    scores['Dataset 1 (Coffee Shop Sales)'] += 2
    print("✗ Dataset 1 has better MAPE (6.68%)")

# Consider work done
scores['Dataset 1 (Coffee Shop Sales)'] += 3  # 90% work already done
print("✓ Dataset 1 has 90% work completed")

print(f"\n{'Dataset':<35} {'Score':<10} {'Verdict'}")
print("-" * 70)
for dataset, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
    verdict = "⭐ BEST CHOICE" if score == max(scores.values()) else ""
    print(f"{dataset:<35} {score}/8       {verdict}")

print("\n" + "="*80)
print("FINAL RECOMMENDATION")
print("="*80)

if scores['Dataset 3 (Raw Data)'] > scores['Dataset 1 (Coffee Shop Sales)'] + 2:
    print("\n🎯 SWITCH TO DATASET 3 (Raw Data.xlsx)")
    print(f"   Reasons:")
    print(f"   • R²: {best_r2:.4f} (much better)")
    print(f"   • MAPE: {best_mape:.2f}%")
    print(f"   • Worth restarting the project")
elif scores['Dataset 3 (Raw Data)'] > scores['Dataset 1 (Coffee Shop Sales)']:
    print("\n⚠️  MARGINAL IMPROVEMENT")
    print(f"   Dataset 3 is slightly better, but:")
    print(f"   • Need to redo 90% of work (10-12 hours)")
    print(f"   • Improvement not significant enough")
    print(f"\n   👉 RECOMMEND: Stick with Dataset 1")
else:
    print("\n👉 STICK WITH DATASET 1 (Coffee Shop Sales.xlsx)")
    print(f"   Reasons:")
    print(f"   • MAPE 6.68% is excellent (Dataset 3: {best_mape:.2f}%)")
    print(f"   • 90% work already completed")
    print(f"   • R² can be explained academically")
    print(f"   • Dataset 3 not significantly better")

# Save for reference
daily.to_csv('/home/user/Coffee-shop/data/processed/raw_data_daily_revenue.csv', index=False)
print(f"\n✓ Saved daily revenue to: data/processed/raw_data_daily_revenue.csv")

print("\n" + "="*80)
