"""
Exploratory Data Analysis for Promotional Response Prediction
=============================================================
Phân tích dữ liệu để hiểu rõ patterns, distributions, và insights
cho bài toán dự báo phản ứng khách hàng với khuyến mãi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "data.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

print("="*80)
print("EXPLORATORY DATA ANALYSIS - PROMOTIONAL RESPONSE PREDICTION")
print("="*80)

# Load data
print("\n1. LOADING DATA...")
df = pd.read_csv(DATA_PATH)
print(f"✓ Loaded {len(df):,} records with {df.shape[1]} columns")

# Basic info
print("\n2. DATASET OVERVIEW")
print("-" * 80)
print(df.info())
print("\nFirst 5 rows:")
print(df.head())

# Statistical summary
print("\n3. STATISTICAL SUMMARY")
print("-" * 80)
print(df.describe())

# Missing values
print("\n4. DATA QUALITY CHECK")
print("-" * 80)
missing = df.isnull().sum()
print("Missing values:")
print(missing[missing > 0] if missing.sum() > 0 else "✓ No missing values")

# Duplicates
duplicates = df.duplicated().sum()
print(f"\nDuplicate rows: {duplicates}")

# Target variable analysis
print("\n5. TARGET VARIABLE ANALYSIS (conversion)")
print("-" * 80)
conversion_counts = df['conversion'].value_counts()
conversion_pct = df['conversion'].value_counts(normalize=True) * 100

print(f"Class 0 (No Conversion): {conversion_counts[0]:,} ({conversion_pct[0]:.2f}%)")
print(f"Class 1 (Conversion):    {conversion_counts[1]:,} ({conversion_pct[1]:.2f}%)")
print(f"\n⚠️  Class Imbalance Ratio: {conversion_pct[0]/conversion_pct[1]:.2f}:1")
print("   → Need SMOTE or class weights to handle imbalance")

# Numerical features analysis
print("\n6. NUMERICAL FEATURES ANALYSIS")
print("-" * 80)

numerical_cols = ['recency', 'history']

for col in numerical_cols:
    print(f"\n{col.upper()}:")
    print(f"  Range: [{df[col].min():.2f}, {df[col].max():.2f}]")
    print(f"  Mean: {df[col].mean():.2f}")
    print(f"  Median: {df[col].median():.2f}")
    print(f"  Std: {df[col].std():.2f}")
    print(f"  Skewness: {df[col].skew():.2f}")
    
    # Conversion rate by quantiles
    df[f'{col}_quartile'] = pd.qcut(df[col], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
    conv_by_quartile = df.groupby(f'{col}_quartile')['conversion'].agg(['mean', 'count'])
    print(f"\n  Conversion rate by quartile:")
    print(conv_by_quartile)

# Categorical features analysis
print("\n7. CATEGORICAL FEATURES ANALYSIS")
print("-" * 80)

categorical_cols = ['used_discount', 'used_bogo', 'zip_code', 'is_referral', 'channel', 'offer']

for col in categorical_cols:
    print(f"\n{col.upper()}:")
    value_counts = df[col].value_counts()
    print(f"  Unique values: {df[col].nunique()}")
    print(f"  Distribution:")
    for val, count in value_counts.items():
        pct = (count / len(df)) * 100
        conv_rate = df[df[col] == val]['conversion'].mean() * 100
        print(f"    {val}: {count:,} ({pct:.2f}%) | Conversion: {conv_rate:.2f}%")

# Correlation analysis
print("\n8. CORRELATION ANALYSIS")
print("-" * 80)

# Create numerical encoding for correlation
df_encoded = df[['recency', 'history', 'used_discount', 'used_bogo', 'is_referral', 'conversion']].copy()
for col in categorical_cols:
    if col in df.columns and df[col].dtype == 'object':
        df_encoded[col] = pd.Categorical(df[col]).codes

correlation = df_encoded.corr()
print("\nCorrelation with TARGET (conversion):")
target_corr = correlation['conversion'].sort_values(ascending=False)
print(target_corr)

# Key insights
print("\n9. KEY INSIGHTS FROM EDA")
print("="*80)

# Insight 1: Referral impact
referral_conv = df.groupby('is_referral')['conversion'].mean()
print(f"✓ INSIGHT 1: Referral Impact")
print(f"  - Referred customers: {referral_conv[1]*100:.2f}% conversion")
print(f"  - Non-referred: {referral_conv[0]*100:.2f}% conversion")
print(f"  - Uplift: {(referral_conv[1] - referral_conv[0])*100:.2f} percentage points")

# Insight 2: Offer effectiveness
offer_conv = df.groupby('offer')['conversion'].mean().sort_values(ascending=False)
print(f"\n✓ INSIGHT 2: Offer Effectiveness")
for offer, rate in offer_conv.items():
    print(f"  - {offer}: {rate*100:.2f}% conversion")

# Insight 3: Channel performance
channel_conv = df.groupby('channel')['conversion'].mean().sort_values(ascending=False)
print(f"\n✓ INSIGHT 3: Channel Performance")
for channel, rate in channel_conv.items():
    count = len(df[df['channel'] == channel])
    print(f"  - {channel}: {rate*100:.2f}% conversion ({count:,} customers)")

# Insight 4: Previous behavior impact
behavior_analysis = df.groupby(['used_discount', 'used_bogo'])['conversion'].agg(['mean', 'count'])
print(f"\n✓ INSIGHT 4: Previous Promotional Behavior")
print(behavior_analysis)

# Insight 5: Recency sweet spot
recency_conv = df.groupby('recency')['conversion'].mean().sort_values(ascending=False).head(5)
print(f"\n✓ INSIGHT 5: Best Recency Values (Top 5)")
for recency, rate in recency_conv.items():
    print(f"  - {recency} days: {rate*100:.2f}% conversion")

# Insight 6: High-value customers
high_value_threshold = df['history'].quantile(0.75)
high_value_conv = df[df['history'] >= high_value_threshold]['conversion'].mean()
low_value_conv = df[df['history'] < high_value_threshold]['conversion'].mean()
print(f"\n✓ INSIGHT 6: Customer Value Impact")
print(f"  - High-value (>= ${high_value_threshold:.2f}): {high_value_conv*100:.2f}% conversion")
print(f"  - Regular customers: {low_value_conv*100:.2f}% conversion")

# Save insights to file
print("\n10. SAVING RESULTS...")
print("-" * 80)

insights_report = f"""
PROMOTIONAL RESPONSE PREDICTION - EDA INSIGHTS
===============================================

Dataset: {len(df):,} customer records
Target: Conversion (binary classification)
Class Imbalance: {conversion_pct[0]:.2f}% No / {conversion_pct[1]:.2f}% Yes

KEY FINDINGS:

1. REFERRAL PROGRAM IMPACT ⭐
   - Referred: {referral_conv[1]*100:.2f}% conversion
   - Non-referred: {referral_conv[0]*100:.2f}% conversion
   - Impact: +{(referral_conv[1] - referral_conv[0])*100:.2f} percentage points
   → ACTION: Prioritize referred customers in targeting

2. OFFER EFFECTIVENESS
{chr(10).join([f"   - {offer}: {rate*100:.2f}%" for offer, rate in offer_conv.items()])}
   → ACTION: Focus budget on most effective offer types

3. CHANNEL PERFORMANCE
{chr(10).join([f"   - {ch}: {rate*100:.2f}%" for ch, rate in channel_conv.items()])}
   → ACTION: Optimize channel mix based on conversion

4. RECENCY MATTERS
   - Best conversion: {recency_conv.index[0]} days ({recency_conv.iloc[0]*100:.2f}%)
   - Pattern: Recent activity predicts higher conversion
   → ACTION: Time campaigns for optimal recency window

5. CUSTOMER VALUE
   - High-value customers: {high_value_conv*100:.2f}% conversion
   - Regular customers: {low_value_conv*100:.2f}% conversion
   → ACTION: Different strategies for different segments

FEATURE ENGINEERING RECOMMENDATIONS:

✓ Create RFM (Recency-Frequency-Monetary) scores
✓ Add interaction features (e.g., offer × channel)
✓ Calculate customer lifetime value indicators
✓ Create promotional affinity scores
✓ Segment customers by behavior patterns

MODEL RECOMMENDATIONS:

✓ Use SMOTE to handle {conversion_pct[0]/conversion_pct[1]:.1f}:1 class imbalance
✓ Try LightGBM, CatBoost (handle categorical well)
✓ Ensemble methods for robust predictions
✓ Optimize for ROC-AUC and business metrics (ROI)
"""

report_path = RESULTS_DIR / "eda_insights.txt"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(insights_report)

print(f"✓ Insights saved to: {report_path}")

# Create visualizations
print("\n11. CREATING VISUALIZATIONS...")
print("-" * 80)

# Figure 1: Target distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pie chart
axes[0].pie(conversion_counts, labels=['No Conversion', 'Conversion'], 
            autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff'])
axes[0].set_title('Target Distribution (Conversion)', fontsize=14, fontweight='bold')

# Bar chart with conversion rate
conversion_data = pd.DataFrame({
    'Category': ['No Conversion', 'Conversion'],
    'Count': conversion_counts.values,
    'Percentage': conversion_pct.values
})
bars = axes[1].bar(conversion_data['Category'], conversion_data['Count'], color=['#ff9999', '#66b3ff'])
axes[1].set_title('Conversion Counts', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Number of Customers')
axes[1].set_xlabel('')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(RESULTS_DIR / "01_target_distribution.png", dpi=300, bbox_inches='tight')
print("✓ Saved: 01_target_distribution.png")
plt.close()

# Figure 2: Conversion rate by categorical features
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, col in enumerate(categorical_cols):
    conv_by_cat = df.groupby(col)['conversion'].mean().sort_values(ascending=False) * 100
    
    bars = axes[idx].bar(range(len(conv_by_cat)), conv_by_cat.values, 
                         color=plt.cm.viridis(np.linspace(0.3, 0.9, len(conv_by_cat))))
    axes[idx].set_xticks(range(len(conv_by_cat)))
    axes[idx].set_xticklabels(conv_by_cat.index, rotation=45, ha='right')
    axes[idx].set_title(f'Conversion Rate by {col.upper()}', fontweight='bold')
    axes[idx].set_ylabel('Conversion Rate (%)')
    axes[idx].axhline(y=conversion_pct[1], color='red', linestyle='--', alpha=0.5, label='Overall avg')
    axes[idx].legend()
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                      f'{height:.1f}%',
                      ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "02_conversion_by_categories.png", dpi=300, bbox_inches='tight')
print("✓ Saved: 02_conversion_by_categories.png")
plt.close()

# Figure 3: Numerical features distribution and conversion
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Recency distribution
axes[0, 0].hist(df['recency'], bins=12, color='skyblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Recency Distribution', fontweight='bold')
axes[0, 0].set_xlabel('Days since last purchase')
axes[0, 0].set_ylabel('Frequency')

# History distribution
axes[0, 1].hist(df['history'], bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Purchase History Distribution', fontweight='bold')
axes[0, 1].set_xlabel('Total spending ($)')
axes[0, 1].set_ylabel('Frequency')

# Conversion by recency
recency_conv_plot = df.groupby('recency')['conversion'].mean() * 100
axes[1, 0].plot(recency_conv_plot.index, recency_conv_plot.values, marker='o', 
                linewidth=2, markersize=8, color='navy')
axes[1, 0].axhline(y=conversion_pct[1], color='red', linestyle='--', alpha=0.5)
axes[1, 0].set_title('Conversion Rate by Recency', fontweight='bold')
axes[1, 0].set_xlabel('Recency (days)')
axes[1, 0].set_ylabel('Conversion Rate (%)')
axes[1, 0].grid(True, alpha=0.3)

# Conversion by history quintiles
history_bins = pd.qcut(df['history'], q=5, duplicates='drop')
history_conv = df.groupby(history_bins)['conversion'].mean() * 100
axes[1, 1].bar(range(len(history_conv)), history_conv.values, color='coral', alpha=0.7)
axes[1, 1].set_xticks(range(len(history_conv)))
axes[1, 1].set_xticklabels([f'Q{i+1}' for i in range(len(history_conv))])
axes[1, 1].axhline(y=conversion_pct[1], color='red', linestyle='--', alpha=0.5)
axes[1, 1].set_title('Conversion Rate by Spending Quintile', fontweight='bold')
axes[1, 1].set_xlabel('Spending Quintile')
axes[1, 1].set_ylabel('Conversion Rate (%)')

plt.tight_layout()
plt.savefig(RESULTS_DIR / "03_numerical_features_analysis.png", dpi=300, bbox_inches='tight')
print("✓ Saved: 03_numerical_features_analysis.png")
plt.close()

# Figure 4: Correlation heatmap
plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(correlation, dtype=bool))
sns.heatmap(correlation, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "04_correlation_matrix.png", dpi=300, bbox_inches='tight')
print("✓ Saved: 04_correlation_matrix.png")
plt.close()

print("\n" + "="*80)
print("✅ EDA COMPLETE!")
print("="*80)
print(f"\nResults saved to: {RESULTS_DIR}")
print("\nNext steps:")
print("  1. Run 02_feature_engineering.py for advanced feature creation")
print("  2. Run 03_model_training.py for model development")
print("  3. Run 04_model_evaluation.py for performance analysis")
