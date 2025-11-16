"""Test R² and MAPE for Raw Data.xlsx"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

print("="*80)
print("TESTING RAW DATA.XLSX - R² AND MAPE")
print("="*80)

# Load processed daily revenue
daily = pd.read_csv('/home/user/Coffee-shop/data/processed/raw_data_daily_revenue.csv')
daily['date'] = pd.to_datetime(daily['date'])
daily = daily.sort_values('date').reset_index(drop=True)

print(f"\nDataset: {len(daily)} days")
print(f"Date range: {daily['date'].min().date()} to {daily['date'].max().date()}")
print(f"Mean daily revenue: ${daily['revenue'].mean():.2f}")
print(f"Std daily revenue: ${daily['revenue'].std():.2f}")
print(f"Min: ${daily['revenue'].min():.2f}, Max: ${daily['revenue'].max():.2f}")

# Check for trend
daily['day_num'] = range(len(daily))
correlation = daily[['day_num', 'revenue']].corr().iloc[0, 1]
growth = (daily['revenue'].iloc[-1] - daily['revenue'].iloc[0]) / daily['revenue'].iloc[0] * 100

print(f"\nTrend analysis:")
print(f"  Correlation with time: {correlation:.4f}")
print(f"  Overall growth: {growth:.2f}%")

# Stationarity test
try:
    adf_result = adfuller(daily['revenue'].values, autolag='AIC')
    print(f"\nStationarity test (ADF):")
    print(f"  ADF Statistic: {adf_result[0]:.4f}")
    print(f"  p-value: {adf_result[1]:.4f}")
    if adf_result[1] < 0.05:
        print(f"  ✓ STATIONARY (p < 0.05)")
    else:
        print(f"  ⚠️  NON-STATIONARY (p >= 0.05)")
except Exception as e:
    print(f"\n⚠️  Stationarity test failed: {e}")

# Split data
train_size = int(0.8 * len(daily))
val_size = int(0.1 * len(daily))

train = daily['revenue'].iloc[:train_size]
val = daily['revenue'].iloc[train_size:train_size+val_size]
test = daily['revenue'].iloc[train_size+val_size:]

print(f"\nData split:")
print(f"  Train: {len(train)} days (${train.mean():.2f} ± ${train.std():.2f})")
print(f"  Val:   {len(val)} days (${val.mean():.2f} ± ${val.std():.2f})")
print(f"  Test:  {len(test)} days (${test.mean():.2f} ± ${test.std():.2f})")
print(f"\nTrain-Test gap: {((test.mean() - train.mean()) / train.mean() * 100):.1f}%")

print("\n" + "="*80)
print("BASELINE MODEL RESULTS")
print("="*80)

models = {}

# Naive
naive_pred = np.array([train.iloc[-1]] * len(test))
models['Naive'] = naive_pred

# Mean
mean_pred = np.array([train.mean()] * len(test))
models['Mean'] = mean_pred

# MA_3
if len(train) >= 3:
    ma3_pred = np.array([train.tail(3).mean()] * len(test))
    models['MA_3'] = ma3_pred

# MA_7
if len(train) >= 7:
    ma7_pred = np.array([train.tail(7).mean()] * len(test))
    models['MA_7'] = ma7_pred

# Evaluate
print(f"\n{'Model':<15} {'MAPE':<10} {'RMSE':<12} {'MAE':<12} {'R²':<10}")
print("-" * 75)

best_r2 = -999
best_model = None
best_mape = 999

for name, pred in models.items():
    mape = mean_absolute_percentage_error(test.values, pred) * 100
    rmse = np.sqrt(mean_squared_error(test.values, pred))
    mae = mean_absolute_error(test.values, pred)
    r2 = r2_score(test.values, pred)

    print(f"{name:<15} {mape:>6.2f}%   ${rmse:>8.2f}   ${mae:>8.2f}   {r2:>8.4f}")

    if r2 > best_r2:
        best_r2 = r2
        best_model = name

    if mape < best_mape:
        best_mape = mape

print(f"\nBest R²: {best_r2:.4f} ({best_model})")
print(f"Best MAPE: {best_mape:.2f}%")

print("\n" + "="*80)
print("COMPARISON WITH OTHER DATASETS")
print("="*80)

datasets_comparison = {
    'Dataset 1: Coffee Shop Sales.xlsx': {
        'Days': 181,
        'Transactions': 149116,
        'Mean daily': 3860,
        'R²': -0.03,
        'MAPE': 6.68,
        'Status': '90% complete'
    },
    'Dataset 2: Coffe_sales.csv': {
        'Days': 381,
        'Transactions': 3547,
        'Mean daily': 295,
        'R²': -0.88,
        'MAPE': 42.82,
        'Status': 'Rejected'
    },
    'Dataset 3: Raw Data.xlsx (NEW)': {
        'Days': len(daily),
        'Transactions': 1000,
        'Mean daily': int(daily['revenue'].mean()),
        'R²': round(best_r2, 4),
        'MAPE': round(best_mape, 2),
        'Status': 'Testing'
    }
}

print(f"\n{'Dataset':<40} {'Days':<8} {'Trans':<10} {'$/day':<10} {'R²':<10} {'MAPE':<10}")
print("-" * 100)
for dataset, metrics in datasets_comparison.items():
    print(f"{dataset:<40} {metrics['Days']:<8} {metrics['Transactions']:<10} "
          f"${metrics['Mean daily']:<9} {metrics['R²']:<10} {metrics['MAPE']:.2f}%")

print("\n" + "="*80)
print("VERDICT")
print("="*80)

# Score each dataset
scores = {}

# Dataset 1 scoring
score1 = 0
score1 += 3 if -0.03 > best_r2 else 0  # R² comparison
score1 += 3 if 6.68 < best_mape else 0  # MAPE comparison
score1 += 3  # 90% work done bonus
score1 += 2 if 3860 > daily['revenue'].mean() else 0  # Data volume
scores['Dataset 1'] = score1

# Dataset 3 scoring
score3 = 0
score3 += 3 if best_r2 > -0.03 else 0  # Better R²
score3 += 3 if best_mape < 6.68 else 0  # Better MAPE
score3 += 2 if best_r2 > 0 else 0  # Positive R² bonus
score3 += 2 if best_mape < 10 else 0  # Excellent MAPE bonus
scores['Dataset 3'] = score3

print(f"\nScoring:")
print(f"  Dataset 1 (Coffee Shop Sales): {scores['Dataset 1']}/11 points")
print(f"  Dataset 3 (Raw Data.xlsx):     {scores['Dataset 3']}/11 points")

print(f"\n{'='*80}")
print("FINAL RECOMMENDATION")
print("="*80)

if best_r2 > 0.5 and best_mape < 10:
    print("\n🎯 STRONGLY RECOMMEND: Dataset 3 (Raw Data.xlsx)")
    print(f"   Reasons:")
    print(f"   ✅ R² = {best_r2:.4f} (EXCELLENT, positive!)")
    print(f"   ✅ MAPE = {best_mape:.2f}% (beats target)")
    print(f"   ✅ Much better than current dataset")
    print(f"\n   Worth restarting the project (10-12 hours)")
elif best_r2 > 0 and best_mape < 15:
    print("\n⚖️  MARGINAL IMPROVEMENT")
    print(f"   Dataset 3 metrics:")
    print(f"   • R² = {best_r2:.4f} (positive, but not great)")
    print(f"   • MAPE = {best_mape:.2f}% (meets target)")
    print(f"\n   Dataset 1 metrics:")
    print(f"   • R² = -0.03 (slightly negative)")
    print(f"   • MAPE = 6.68% (EXCELLENT)")
    print(f"\n   Decision: Dataset 1 has better MAPE, worth keeping")
    print(f"   👉 RECOMMEND: Stick with Dataset 1")
else:
    print("\n👉 STICK WITH DATASET 1 (Coffee Shop Sales.xlsx)")
    print(f"   Comparison:")
    print(f"   • Dataset 1 MAPE: 6.68% ✅ (excellent)")
    print(f"   • Dataset 3 MAPE: {best_mape:.2f}%", end="")
    if best_mape > 15:
        print(" ✗ (worse)")
    else:
        print(" ~ (similar)")
    print(f"   • Dataset 1: 90% complete")
    print(f"   • Dataset 3: Would need 10-12 hours to rebuild")
    print(f"\n   Not worth switching!")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Full time series
ax1 = axes[0, 0]
ax1.plot(range(len(train)), train.values, 'b-', label='Train', alpha=0.7)
ax1.plot(range(len(train), len(train)+len(val)), val.values, 'orange', label='Val', alpha=0.7)
ax1.plot(range(len(train)+len(val), len(daily)), test.values, 'g-', label='Test (Actual)', alpha=0.7, linewidth=2)
if 'MA_7' in models:
    ax1.axhline(y=models['MA_7'][0], color='r', linestyle='--', label=f'MA_7 Pred (${models["MA_7"][0]:.0f})', linewidth=2)
ax1.set_xlabel('Day')
ax1.set_ylabel('Revenue ($)')
ax1.set_title('Raw Data.xlsx: Time Series with Predictions')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Test set predictions
ax2 = axes[0, 1]
test_indices = range(len(test))
ax2.plot(test_indices, test.values, 'g-o', label='Actual', linewidth=2, markersize=4)
if 'MA_7' in models:
    ax2.axhline(y=models['MA_7'][0], color='r', linestyle='--', label=f'MA_7', linewidth=2)
    ax2.fill_between(test_indices, models['MA_7'][0], test.values, alpha=0.3, color='red')
ax2.set_xlabel('Test Day')
ax2.set_ylabel('Revenue ($)')
ax2.set_title(f'Test Set: R² = {best_r2:.4f}, MAPE = {best_mape:.2f}%')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Distribution
ax3 = axes[1, 0]
ax3.hist(train.values, bins=30, alpha=0.5, label='Train', color='blue')
ax3.hist(test.values, bins=30, alpha=0.5, label='Test', color='green')
ax3.axvline(x=train.mean(), color='blue', linestyle='--', linewidth=2)
ax3.axvline(x=test.mean(), color='green', linestyle='--', linewidth=2)
ax3.set_xlabel('Revenue ($)')
ax3.set_ylabel('Frequency')
ax3.set_title('Train vs Test Distribution')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Dataset comparison
ax4 = axes[1, 1]
datasets = ['Dataset 1\n(Coffee Shop)', 'Dataset 2\n(Coffe_sales)', 'Dataset 3\n(Raw Data)']
r2_values = [-0.03, -0.88, best_r2]
mape_values = [6.68, 42.82, best_mape]

x = np.arange(len(datasets))
width = 0.35

bars1 = ax4.bar(x - width/2, r2_values, width, label='R²', alpha=0.7)
ax4_twin = ax4.twinx()
bars2 = ax4_twin.bar(x + width/2, mape_values, width, label='MAPE %', alpha=0.7, color='orange')

ax4.set_xlabel('Dataset')
ax4.set_ylabel('R² Score')
ax4_twin.set_ylabel('MAPE (%)')
ax4.set_title('Dataset Comparison')
ax4.set_xticks(x)
ax4.set_xticklabels(datasets)
ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax4.axhline(y=0.85, color='green', linestyle='--', linewidth=1, alpha=0.5, label='R² target')
ax4_twin.axhline(y=15, color='red', linestyle='--', linewidth=1, alpha=0.5, label='MAPE target')
ax4.legend(loc='upper left')
ax4_twin.legend(loc='upper right')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/user/Coffee-shop/results/raw_data_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved visualization: results/raw_data_analysis.png")

print("\n" + "="*80)
