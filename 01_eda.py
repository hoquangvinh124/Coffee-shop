"""
Coffee Shop Revenue Prediction - EDA & Model Training
Dataset: coffee_shop_revenue1.csv (2000 rows, 7 columns)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 80)
print("COFFEE SHOP REVENUE PREDICTION - EDA & MODEL TRAINING")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1] Loading data...")
df = pd.read_csv('coffee_shop_revenue1.csv')

print(f"✓ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nColumns:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")

# ============================================================================
# 2. BASIC INFORMATION
# ============================================================================
print("\n" + "=" * 80)
print("[2] BASIC INFORMATION")
print("=" * 80)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nBasic Statistics:")
print(df.describe())

# ============================================================================
# 3. TARGET VARIABLE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("[3] TARGET VARIABLE ANALYSIS: Daily_Revenue")
print("=" * 80)

print(f"\nMean Revenue: ${df['Daily_Revenue'].mean():.2f}")
print(f"Median Revenue: ${df['Daily_Revenue'].median():.2f}")
print(f"Std Revenue: ${df['Daily_Revenue'].std():.2f}")
print(f"Min Revenue: ${df['Daily_Revenue'].min():.2f}")
print(f"Max Revenue: ${df['Daily_Revenue'].max():.2f}")
print(f"Range: ${df['Daily_Revenue'].max() - df['Daily_Revenue'].min():.2f}")

# ============================================================================
# 4. FEATURE DISTRIBUTIONS
# ============================================================================
print("\n" + "=" * 80)
print("[4] FEATURE DISTRIBUTIONS")
print("=" * 80)

features = [col for col in df.columns if col != 'Daily_Revenue']

print("\nFeature Statistics:")
print("-" * 80)
for feature in features:
    print(f"\n{feature}:")
    print(f"  Mean: {df[feature].mean():.2f}")
    print(f"  Median: {df[feature].median():.2f}")
    print(f"  Std: {df[feature].std():.2f}")
    print(f"  Range: [{df[feature].min():.2f}, {df[feature].max():.2f}]")

# ============================================================================
# 5. CORRELATION ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("[5] CORRELATION ANALYSIS")
print("=" * 80)

correlations = df.corr()['Daily_Revenue'].sort_values(ascending=False)
print("\nCorrelation with Daily_Revenue:")
print(correlations)

# ============================================================================
# 6. VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("[6] Creating visualizations...")
print("=" * 80)

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 12))

# 6.1 Target Distribution
ax1 = plt.subplot(3, 3, 1)
df['Daily_Revenue'].hist(bins=50, edgecolor='black', alpha=0.7)
ax1.axvline(df['Daily_Revenue'].mean(), color='r', linestyle='--', label=f'Mean: ${df["Daily_Revenue"].mean():.0f}')
ax1.axvline(df['Daily_Revenue'].median(), color='g', linestyle='--', label=f'Median: ${df["Daily_Revenue"].median():.0f}')
ax1.set_title('Daily Revenue Distribution', fontsize=12, fontweight='bold')
ax1.set_xlabel('Daily Revenue ($)')
ax1.set_ylabel('Frequency')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 6.2 Correlation Heatmap
ax2 = plt.subplot(3, 3, 2)
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, ax=ax2, cbar_kws={'shrink': 0.8})
ax2.set_title('Feature Correlation Matrix', fontsize=12, fontweight='bold')

# 6.3 Box Plot - Revenue
ax3 = plt.subplot(3, 3, 3)
bp = ax3.boxplot(df['Daily_Revenue'], vert=True, patch_artist=True)
bp['boxes'][0].set_facecolor('skyblue')
ax3.set_ylabel('Daily Revenue ($)')
ax3.set_title('Revenue Box Plot (Outlier Detection)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 6.4-6.9 Scatter plots for each feature
for idx, feature in enumerate(features, start=4):
    ax = plt.subplot(3, 3, idx)
    ax.scatter(df[feature], df['Daily_Revenue'], alpha=0.5, s=10)

    # Add trend line
    z = np.polyfit(df[feature], df['Daily_Revenue'], 1)
    p = np.poly1d(z)
    ax.plot(df[feature], p(df[feature]), "r--", alpha=0.8, linewidth=2)

    # Get correlation
    corr = df[feature].corr(df['Daily_Revenue'])
    ax.set_xlabel(feature.replace('_', ' '))
    ax.set_ylabel('Daily Revenue ($)')
    ax.set_title(f'{feature.replace("_", " ")}\n(Corr: {corr:.3f})', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/eda_comprehensive.png', dpi=150, bbox_inches='tight')
print("✓ Saved: results/eda_comprehensive.png")

# ============================================================================
# 7. KEY INSIGHTS
# ============================================================================
print("\n" + "=" * 80)
print("[7] KEY INSIGHTS")
print("=" * 80)

# Find top correlated features
top_features = correlations[1:].head(3)  # Exclude self-correlation
print("\nTop 3 Features Most Correlated with Revenue:")
for i, (feature, corr) in enumerate(top_features.items(), 1):
    print(f"  {i}. {feature}: {corr:.4f}")

# Check for outliers
Q1 = df['Daily_Revenue'].quantile(0.25)
Q3 = df['Daily_Revenue'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['Daily_Revenue'] < Q1 - 1.5*IQR) | (df['Daily_Revenue'] > Q3 + 1.5*IQR)]
print(f"\nOutliers detected: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")

# Check for multicollinearity
print("\nFeature Intercorrelations (potential multicollinearity):")
feature_corr = df[features].corr()
high_corr_pairs = []
for i in range(len(feature_corr.columns)):
    for j in range(i+1, len(feature_corr.columns)):
        if abs(feature_corr.iloc[i, j]) > 0.7:
            high_corr_pairs.append((feature_corr.columns[i], feature_corr.columns[j], feature_corr.iloc[i, j]))

if high_corr_pairs:
    for feat1, feat2, corr in high_corr_pairs:
        print(f"  {feat1} <-> {feat2}: {corr:.4f}")
else:
    print("  None (no pairs with |corr| > 0.7)")

print("\n" + "=" * 80)
print("EDA COMPLETE!")
print("=" * 80)
print("\nNext steps:")
print("  1. Data preprocessing (scaling, encoding if needed)")
print("  2. Train-test split")
print("  3. Model training (Linear Regression, Random Forest, XGBoost, LightGBM)")
print("  4. Model evaluation and comparison")
print("=" * 80)
