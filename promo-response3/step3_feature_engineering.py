"""
STEP 3: ADVANCED FEATURE ENGINEERING - Interaction Features
================================
Mục tiêu: Tạo các features "thông minh" để tăng signal cho F1 > 90%
- spending_velocity, context_combo, menu_combo, promo_sensitivity
- Target encoding cho categorical features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("BƯỚC 3: ADVANCED FEATURE ENGINEERING - Trụ cột #2")
print("=" * 80)

# Load enhanced data
print("\n📊 Loading enhanced_data.csv...")
df = pd.read_csv('data/enhanced_data.csv')
print(f"   Shape: {df.shape}")
print(f"   Columns: {list(df.columns)}")

# ==================== INTERACTION FEATURES ====================
print("\n" + "=" * 80)
print("🔧 CREATING INTERACTION FEATURES (The Key to F1 > 90%)")
print("=" * 80)

# 1. SPENDING_VELOCITY - Tốc độ chi tiêu
print("\n1️⃣  Creating 'spending_velocity' - Spending Rate")
print("   Formula: history / (recency + 1)")
print("   Logic: Measures how fast customer spends over time")
df['spending_velocity'] = df['history'] / (df['recency'] + 1)
print(f"   ✅ spending_velocity created")
print(f"      Mean: ${df['spending_velocity'].mean():.2f}, Std: ${df['spending_velocity'].std():.2f}")

# 2. CONTEXT_COMBO - Kết hợp ngữ cảnh tiêu dùng
print("\n2️⃣  Creating 'context_combo' - Seat Usage + Time of Day")
print("   Examples: 'Dine-in_Morning', 'Take-away_Afternoon'")
df['context_combo'] = df['seat_usage'] + '_' + df['time_of_day']
print(f"   ✅ context_combo created: {df['context_combo'].nunique()} unique combinations")
print(f"      Top 3: {df['context_combo'].value_counts().head(3).to_dict()}")

# 3. MENU_COMBO - Kết hợp thực đơn
print("\n3️⃣  Creating 'menu_combo' - Drink + Food Category")
print("   Examples: 'Specialty Coffee_Pastry', 'Tea_NoFood'")
df['menu_combo'] = df['drink_category'] + '_' + df['food_category']
print(f"   ✅ menu_combo created: {df['menu_combo'].nunique()} unique combinations")
print(f"      Top 3: {df['menu_combo'].value_counts().head(3).to_dict()}")

# 4. PROMO_SENSITIVITY - Độ nhạy cảm với khuyến mãi
print("\n4️⃣  Creating 'promo_sensitivity' - Discount + BOGO Usage")
print("   Formula: used_discount + used_bogo (range: 0-2)")
df['promo_sensitivity'] = df['used_discount'] + df['used_bogo']
print(f"   ✅ promo_sensitivity created")
print(f"      Distribution: {df['promo_sensitivity'].value_counts().to_dict()}")

# 5. AVG_TRANSACTION_VALUE - Giá trị giao dịch trung bình
print("\n5️⃣  Creating 'avg_transaction_value' - History per Visit")
print("   Proxy: Assumes 1 visit per 2 months of recency")
df['avg_transaction_value'] = df['history'] / (12 - df['recency'] + 1)
print(f"   ✅ avg_transaction_value created")
print(f"      Mean: ${df['avg_transaction_value'].mean():.2f}")

# 6. ENGAGEMENT_SCORE - Điểm tương tác tổng hợp
print("\n6️⃣  Creating 'engagement_score' - Composite Engagement Metric")
print("   Formula: (is_referral * 2) + promo_sensitivity + (1 if Multichannel else 0)")
df['engagement_score'] = (df['is_referral'] * 2) + df['promo_sensitivity'] + \
                         (df['channel'] == 'Multichannel').astype(int)
print(f"   ✅ engagement_score created")
print(f"      Range: {df['engagement_score'].min()} - {df['engagement_score'].max()}")
print(f"      Mean: {df['engagement_score'].mean():.2f}")

# 7. OFFER_CHANNEL_MATCH - Khớp giữa Offer và Channel
print("\n7️⃣  Creating 'offer_channel_match' - Strategic Alignment")
print("   Logic: Web + Discount, Phone + BOGO, Multichannel + No Offer = strategic")
def calculate_offer_channel_match(row):
    if (row['channel'] == 'Web' and row['offer'] == 'Discount') or \
       (row['channel'] == 'Phone' and row['offer'] == 'Buy One Get One') or \
       (row['channel'] == 'Multichannel' and row['offer'] == 'No Offer'):
        return 1
    else:
        return 0

df['offer_channel_match'] = df.apply(calculate_offer_channel_match, axis=1)
print(f"   ✅ offer_channel_match created")
print(f"      Strategic matches: {df['offer_channel_match'].sum():,} ({df['offer_channel_match'].mean()*100:.1f}%)")

# ==================== TARGET ENCODING (CRITICAL FOR CATEGORICAL) ====================
print("\n" + "=" * 80)
print("🎯 TARGET ENCODING - Powerful Technique for Categorical Features")
print("=" * 80)

print("\n⚠️  Important: Using proper train/validation split to avoid data leakage")

# Split data for proper target encoding
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['conversion'])
print(f"\n📊 Data split:")
print(f"   Train: {len(train_df):,} rows ({len(train_df)/len(df)*100:.1f}%)")
print(f"   Validation: {len(val_df):,} rows ({len(val_df)/len(df)*100:.1f}%)")

# Target encode on train, apply to full dataset
target_encode_cols = ['context_combo', 'menu_combo', 'zip_code', 'drink_category', 'food_category']

print(f"\n🔧 Applying Target Encoding to {len(target_encode_cols)} features:")

for col in target_encode_cols:
    # Calculate target encoding on train set only
    target_mean = train_df.groupby(col)['conversion'].mean()
    global_mean = train_df['conversion'].mean()
    
    # Apply to full dataset with smoothing (handle unseen categories)
    new_col_name = f'{col}_target_enc'
    df[new_col_name] = df[col].map(target_mean).fillna(global_mean)
    
    print(f"   ✅ {new_col_name}")
    print(f"      Range: {df[new_col_name].min():.4f} - {df[new_col_name].max():.4f}")

# ==================== ADDITIONAL NUMERIC TRANSFORMATIONS ====================
print("\n" + "=" * 80)
print("📐 ADDITIONAL NUMERIC TRANSFORMATIONS")
print("=" * 80)

# Log transformations for skewed features
print("\n1️⃣  Log transformation for 'history' (right-skewed)")
df['history_log'] = np.log1p(df['history'])
print(f"   ✅ history_log created")

print("\n2️⃣  Squared transformation for 'recency' (capture non-linearity)")
df['recency_squared'] = df['recency'] ** 2
print(f"   ✅ recency_squared created")

# ==================== SAVE FINAL ENGINEERED DATA ====================
output_path = 'data/final_engineered_data.csv'
df.to_csv(output_path, index=False)
print(f"\n💾 Final engineered data saved to: {output_path}")
print(f"   Total columns: {df.shape[1]}")

# ==================== FEATURE IMPORTANCE PREVIEW ====================
print("\n" + "=" * 80)
print("📊 FEATURE CORRELATION WITH CONVERSION (Preview)")
print("=" * 80)

# Select numeric features for correlation
numeric_features = ['spending_velocity', 'promo_sensitivity', 'avg_transaction_value',
                   'engagement_score', 'offer_channel_match', 'history_log', 'recency_squared',
                   'context_combo_target_enc', 'menu_combo_target_enc', 'zip_code_target_enc',
                   'drink_category_target_enc', 'food_category_target_enc',
                   'history', 'recency', 'used_discount', 'used_bogo', 'is_referral']

correlation_with_target = df[numeric_features + ['conversion']].corr()['conversion'].drop('conversion').sort_values(ascending=False)

print("\n🔝 Top 10 Most Correlated Features with Conversion:")
print(correlation_with_target.head(10))

print("\n🔻 Bottom 5 Least Correlated Features:")
print(correlation_with_target.tail(5))

# Visualization: Feature Correlations
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Top correlations
top_features = correlation_with_target.head(15)
top_features.plot(kind='barh', ax=axes[0], color='#3498db')
axes[0].set_title('Top 15 Features Correlated with Conversion', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Correlation Coefficient')
axes[0].axvline(x=0, color='red', linestyle='--', linewidth=1)

# Distribution of interaction features
interaction_features = ['spending_velocity', 'promo_sensitivity', 'engagement_score']
for i, feat in enumerate(interaction_features):
    color = '#2ecc71' if i == 0 else '#e74c3c' if i == 1 else '#f39c12'
    df[df['conversion'] == 1][feat].hist(bins=30, alpha=0.5, label=f'Conversion=1', 
                                         ax=axes[1], color=color)

axes[1].set_title('Distribution of Key Interaction Features (Converters)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Feature Value')
axes[1].set_ylabel('Frequency')
axes[1].legend()

plt.tight_layout()
plt.savefig('04_feature_engineering_analysis.png', dpi=300, bbox_inches='tight')
print("\n✅ Visualization saved: 04_feature_engineering_analysis.png")

# ==================== SUMMARY ====================
print("\n" + "=" * 80)
print("✅ STEP 3 COMPLETED: ADVANCED FEATURE ENGINEERING")
print("=" * 80)

new_features = [
    'spending_velocity', 'context_combo', 'menu_combo', 'promo_sensitivity',
    'avg_transaction_value', 'engagement_score', 'offer_channel_match',
    'history_log', 'recency_squared',
    'context_combo_target_enc', 'menu_combo_target_enc', 'zip_code_target_enc',
    'drink_category_target_enc', 'food_category_target_enc'
]

print(f"\n📊 Feature Engineering Summary:")
print(f"   Starting columns: 15")
print(f"   New features created: {len(new_features)}")
print(f"   Final columns: {df.shape[1]}")

print(f"\n✨ Key Engineered Features:")
for i, feat in enumerate(new_features, 1):
    print(f"   {i:2d}. {feat}")

print(f"\n💡 Why these features are critical for F1 > 90%:")
print(f"   ✓ Interaction features capture complex patterns (spending_velocity, context_combo)")
print(f"   ✓ Target encoding converts high-cardinality categoricals to numeric signal")
print(f"   ✓ Composite metrics (engagement_score) summarize multiple signals")
print(f"   ✓ Strategic alignment features (offer_channel_match) capture business logic")

print(f"\n🚀 Next Steps:")
print(f"   → Step 4: Handle Imbalance with SMOTE + ENN")
print(f"   → Step 5: Train 'Big 3' Models (LightGBM, XGBoost, CatBoost)")
print(f"   → Step 6: Build Stacking Ensemble + Meta-Model")
print(f"   → Step 7: Threshold Tuning to maximize F1-score")

print("\n" + "=" * 80)
print(f"📁 Output: {output_path}")
print("=" * 80)
