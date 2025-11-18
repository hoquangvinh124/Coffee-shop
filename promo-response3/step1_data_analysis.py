"""
STEP 1: DATA ANALYSIS - Phân tích Dữ liệu Ban đầu
================================
Mục tiêu: Hiểu rõ dữ liệu, phát hiện imbalance, và xác định chiến lược
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

print("=" * 80)
print("BƯỚC 1: PHÂN TÍCH DỮ LIỆU - THE DATA FOUNDATION")
print("=" * 80)

# Load data
print("\n📊 Loading data.csv...")
df = pd.read_csv('data/data.csv')

print(f"\n✅ Data loaded successfully!")
print(f"   Shape: {df.shape}")
print(f"   Columns: {df.columns.tolist()}")

# ==================== CRITICAL: CLASS IMBALANCE ANALYSIS ====================
print("\n" + "=" * 80)
print("⚠️  PHÁT HIỆN THEN CHỐT: CLASS IMBALANCE ANALYSIS")
print("=" * 80)

conversion_counts = df['conversion'].value_counts()
conversion_ratio = df['conversion'].value_counts(normalize=True) * 100

print("\n📈 Conversion Distribution:")
print(f"   Class 0 (No Conversion): {conversion_counts[0]:,} ({conversion_ratio[0]:.2f}%)")
print(f"   Class 1 (Conversion):    {conversion_counts[1]:,} ({conversion_ratio[1]:.2f}%)")
print(f"\n   ⚠️  IMBALANCE RATIO: {conversion_ratio[0] / conversion_ratio[1]:.2f} : 1")
print(f"   ⚠️  Đây là dữ liệu MẤT CÂN BẰNG NGHIÊM TRỌNG!")

# Visual: Class distribution
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Bar plot
conversion_counts.plot(kind='bar', ax=axes[0], color=['#e74c3c', '#2ecc71'])
axes[0].set_title('Class Distribution - Absolute Count', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Conversion (0=No, 1=Yes)', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].tick_params(rotation=0)

# Pie chart
axes[1].pie(conversion_counts, labels=['No Conversion (0)', 'Conversion (1)'], 
            autopct='%1.1f%%', colors=['#e74c3c', '#2ecc71'], startangle=90)
axes[1].set_title('Class Distribution - Percentage', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('01_class_imbalance_analysis.png', dpi=300, bbox_inches='tight')
print("\n✅ Visualization saved: 01_class_imbalance_analysis.png")

# ==================== BASIC STATISTICS ====================
print("\n" + "=" * 80)
print("📊 BASIC STATISTICS")
print("=" * 80)

print("\n1. Missing Values Check:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("   ✅ No missing values detected!")
else:
    print(missing[missing > 0])

print("\n2. Data Types:")
print(df.dtypes)

print("\n3. Numerical Features Statistics:")
print(df.describe())

print("\n4. Categorical Features - Unique Values:")
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"   {col}: {df[col].nunique()} unique values - {df[col].unique()[:5]}")

# ==================== FEATURE ANALYSIS ====================
print("\n" + "=" * 80)
print("🔍 FEATURE ANALYSIS BY CONVERSION")
print("=" * 80)

# Numerical features by conversion
numerical_cols = ['recency', 'history']
print("\n1. Numerical Features Statistics by Conversion:")
for col in numerical_cols:
    print(f"\n   {col.upper()}:")
    print(df.groupby('conversion')[col].describe()[['mean', 'std', 'min', 'max']])

# Categorical features by conversion
print("\n2. Categorical Features Distribution by Conversion:")
categorical_analysis = ['offer', 'channel', 'zip_code']
for col in categorical_analysis:
    print(f"\n   {col.upper()}:")
    crosstab = pd.crosstab(df[col], df['conversion'], normalize='index') * 100
    print(crosstab.round(2))

# ==================== CORRELATION ANALYSIS ====================
print("\n" + "=" * 80)
print("🔗 CORRELATION ANALYSIS")
print("=" * 80)

# Prepare data for correlation
df_corr = df.copy()
# Encode binary features
df_corr['used_discount'] = df_corr['used_discount'].astype(int)
df_corr['used_bogo'] = df_corr['used_bogo'].astype(int)
df_corr['is_referral'] = df_corr['is_referral'].astype(int)

# Select numeric columns for correlation
numeric_features = ['recency', 'history', 'used_discount', 'used_bogo', 'is_referral', 'conversion']
correlation_matrix = df_corr[numeric_features].corr()

print("\n📊 Correlation with Conversion:")
conversion_corr = correlation_matrix['conversion'].sort_values(ascending=False)
print(conversion_corr)

# Visual: Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, fmt='.3f')
plt.title('Correlation Matrix - Original Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('02_correlation_matrix.png', dpi=300, bbox_inches='tight')
print("\n✅ Visualization saved: 02_correlation_matrix.png")

# ==================== KEY INSIGHTS ====================
print("\n" + "=" * 80)
print("💡 KEY INSIGHTS & STRATEGIC IMPLICATIONS")
print("=" * 80)

print("""
1. ⚠️  CRITICAL CHALLENGE: Severe Class Imbalance (85.3% : 14.7%)
   → Strategy: SMOTE + ENN resampling + Class Weights required

2. 📊 Feature Correlation with Conversion:
   → Identify which features have strongest signal
   → Need feature engineering to amplify these signals

3. 🎯 Path to F1 > 90%:
   ✓ Enhanced features with F&B context (seat_usage, time_of_day, etc.)
   ✓ Interaction features (spending_velocity, context_combo, etc.)
   ✓ Advanced ensemble (LightGBM + XGBoost + CatBoost + Stacking)
   ✓ Threshold tuning for optimal F1-score

4. 📈 Next Steps:
   → Step 2: Create enhanced_data.csv (15 columns)
   → Step 3: Advanced feature engineering
   → Step 4: Build "Big 3" models with Optuna tuning
   → Step 5: Stacking ensemble + Meta-model
   → Step 6: Threshold optimization

""")

print("=" * 80)
print("✅ STEP 1 COMPLETED: Data Analysis Foundation Established")
print("=" * 80)
print("\n📊 Generated Files:")
print("   - 01_class_imbalance_analysis.png")
print("   - 02_correlation_matrix.png")
print("\n🚀 Ready to proceed to Step 2: Enhanced Feature Creation")
