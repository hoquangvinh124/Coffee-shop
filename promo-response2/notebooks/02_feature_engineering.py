"""
Feature Engineering for Promotional Response Prediction
=======================================================
Tạo các features nâng cao để cải thiện hiệu suất model
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "data.csv"
OUTPUT_DIR = BASE_DIR / "data"
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*80)
print("FEATURE ENGINEERING - PROMOTIONAL RESPONSE PREDICTION")
print("="*80)

# Load data
print("\n1. LOADING DATA...")
df = pd.read_csv(DATA_PATH)
print(f"✓ Loaded {len(df):,} records")
print(f"  Original features: {df.shape[1]}")

# Create engineered features
print("\n2. CREATING ENGINEERED FEATURES...")
print("-" * 80)

# ========== RFM ANALYSIS ==========
print("\n✓ RFM Features")

# Recency Score (1-5, lower recency = higher score)
df['recency_score'] = pd.cut(df['recency'], bins=5, labels=[5, 4, 3, 2, 1])
df['recency_score'] = df['recency_score'].astype(int)

# Monetary Score (1-5, higher spending = higher score)
df['monetary_score'] = pd.qcut(df['history'], q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
df['monetary_score'] = df['monetary_score'].astype(int)

# Combined RFM score
df['rfm_score'] = df['recency_score'] + df['monetary_score']

# RFM segments
def rfm_segment(row):
    if row['rfm_score'] >= 9:
        return 'Champions'
    elif row['rfm_score'] >= 7:
        return 'Loyal'
    elif row['rfm_score'] >= 5:
        return 'Potential'
    elif row['rfm_score'] >= 3:
        return 'At Risk'
    else:
        return 'Lost'

df['customer_segment'] = df.apply(rfm_segment, axis=1)

print(f"  - recency_score (1-5)")
print(f"  - monetary_score (1-5)")
print(f"  - rfm_score (2-10)")
print(f"  - customer_segment (5 categories)")

# ========== BEHAVIORAL FEATURES ==========
print("\n✓ Behavioral Features")

# Promo engagement score
df['promo_engagement'] = (df['used_discount'].astype(int) + df['used_bogo'].astype(int)) / 2

# Promo variety (used different types)
df['promo_variety'] = (df['used_discount'] != df['used_bogo']).astype(int)

# High recency flag (inactive customer)
df['is_inactive'] = (df['recency'] >= 7).astype(int)

# Recent customer flag
df['is_recent'] = (df['recency'] <= 3).astype(int)

# High value customer
history_75 = df['history'].quantile(0.75)
df['is_high_value'] = (df['history'] >= history_75).astype(int)

# Low value customer
history_25 = df['history'].quantile(0.25)
df['is_low_value'] = (df['history'] <= history_25).astype(int)

print(f"  - promo_engagement (0-1)")
print(f"  - promo_variety (0/1)")
print(f"  - is_inactive (0/1)")
print(f"  - is_recent (0/1)")
print(f"  - is_high_value (0/1)")
print(f"  - is_low_value (0/1)")

# ========== INTERACTION FEATURES ==========
print("\n✓ Interaction Features")

# Offer × Channel interactions
df['offer_channel'] = df['offer'] + '_' + df['channel']

# Referral × Recency
df['referral_recent'] = df['is_referral'] * df['is_recent']

# High value × Offer type
df['highvalue_discount'] = df['is_high_value'] * df['used_discount']
df['highvalue_bogo'] = df['is_high_value'] * df['used_bogo']

# Location × Channel
df['location_channel'] = df['zip_code'] + '_' + df['channel']

# Promo engagement × Current offer
df['engagement_discount_offer'] = df['promo_engagement'] * (df['offer'] == 'Discount').astype(int)
df['engagement_bogo_offer'] = df['promo_engagement'] * (df['offer'] == 'Buy One Get One').astype(int)

print(f"  - offer_channel (9 combinations)")
print(f"  - referral_recent (0/1)")
print(f"  - highvalue_discount (0/1)")
print(f"  - highvalue_bogo (0/1)")
print(f"  - location_channel (9 combinations)")
print(f"  - engagement_discount_offer (0-1)")
print(f"  - engagement_bogo_offer (0-1)")

# ========== SPENDING FEATURES ==========
print("\n✓ Spending Features")

# Spending per day (proxy for frequency)
df['spending_per_day'] = df['history'] / (df['recency'] + 1)  # +1 to avoid division by zero

# Spending category
df['spending_category'] = pd.qcut(df['history'], q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')

# Log transform for skewed history
df['history_log'] = np.log1p(df['history'])

print(f"  - spending_per_day (continuous)")
print(f"  - spending_category (Low/Medium/High)")
print(f"  - history_log (log-transformed)")

# ========== CHANNEL PREFERENCE ==========
print("\n✓ Channel Features")

# Channel diversity score
channel_map = {'Phone': 1, 'Web': 2, 'Multichannel': 3}
df['channel_score'] = df['channel'].map(channel_map)

# Preferred digital channel
df['is_digital'] = (df['channel'].isin(['Web', 'Multichannel'])).astype(int)

print(f"  - channel_score (1-3)")
print(f"  - is_digital (0/1)")

# ========== OFFER MATCHING ==========
print("\n✓ Offer Matching Features")

# Offer match with previous behavior
df['discount_match'] = ((df['offer'] == 'Discount') & (df['used_discount'] == 1)).astype(int)
df['bogo_match'] = ((df['offer'] == 'Buy One Get One') & (df['used_bogo'] == 1)).astype(int)
df['offer_mismatch'] = ((df['discount_match'] == 0) & (df['bogo_match'] == 0) & (df['offer'] != 'No Offer')).astype(int)

print(f"  - discount_match (0/1)")
print(f"  - bogo_match (0/1)")
print(f"  - offer_mismatch (0/1)")

# ========== LOCATION FEATURES ==========
print("\n✓ Location Features")

# Urban preference
df['is_urban'] = (df['zip_code'] == 'Urban').astype(int)
df['is_rural'] = (df['zip_code'] == 'Rural').astype(int)

print(f"  - is_urban (0/1)")
print(f"  - is_rural (0/1)")

# Summary
print("\n3. FEATURE ENGINEERING SUMMARY")
print("="*80)
print(f"Original features: {9}")
print(f"Engineered features: {df.shape[1] - 9}")
print(f"Total features: {df.shape[1]}")

# Show new columns
new_cols = [col for col in df.columns if col not in ['recency', 'history', 'used_discount', 
                                                       'used_bogo', 'zip_code', 'is_referral', 
                                                       'channel', 'offer', 'conversion']]
print(f"\nNew features created ({len(new_cols)}):")
for i, col in enumerate(new_cols, 1):
    print(f"  {i:2d}. {col}")

# Display feature statistics by conversion
print("\n4. FEATURE IMPACT ON CONVERSION")
print("-" * 80)

# Top features by conversion difference
feature_importance = {}

# RFM segments
segment_conv = df.groupby('customer_segment')['conversion'].mean().sort_values(ascending=False)
print("\n✓ Customer Segments:")
for segment, rate in segment_conv.items():
    count = len(df[df['customer_segment'] == segment])
    print(f"  {segment:15s}: {rate*100:5.2f}% conversion ({count:,} customers)")

# Behavioral flags
behavioral_features = ['is_inactive', 'is_recent', 'is_high_value', 'is_low_value', 
                       'is_digital', 'promo_variety']
print("\n✓ Behavioral Flags:")
for feat in behavioral_features:
    conv_0 = df[df[feat] == 0]['conversion'].mean() * 100
    conv_1 = df[df[feat] == 1]['conversion'].mean() * 100
    diff = conv_1 - conv_0
    print(f"  {feat:20s}: {conv_0:5.2f}% (0) vs {conv_1:5.2f}% (1) | Δ {diff:+5.2f}%")

# Offer matching
print("\n✓ Offer Matching:")
for feat in ['discount_match', 'bogo_match', 'offer_mismatch']:
    conv_0 = df[df[feat] == 0]['conversion'].mean() * 100
    conv_1 = df[df[feat] == 1]['conversion'].mean() * 100
    diff = conv_1 - conv_0
    count = df[feat].sum()
    print(f"  {feat:20s}: {conv_1:5.2f}% conversion ({count:,} customers) | Δ {diff:+5.2f}%")

# Save engineered dataset
print("\n5. SAVING ENGINEERED DATASET...")
print("-" * 80)

output_path = OUTPUT_DIR / "data_engineered.csv"
df.to_csv(output_path, index=False)
print(f"✓ Saved to: {output_path}")
print(f"  Shape: {df.shape}")

# Save feature list
feature_list_path = OUTPUT_DIR / "feature_list.txt"
with open(feature_list_path, 'w', encoding='utf-8') as f:
    f.write("FEATURE LIST - PROMOTIONAL RESPONSE PREDICTION\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("ORIGINAL FEATURES (9):\n")
    f.write("-" * 80 + "\n")
    original_features = ['recency', 'history', 'used_discount', 'used_bogo', 
                        'zip_code', 'is_referral', 'channel', 'offer', 'conversion']
    for i, feat in enumerate(original_features, 1):
        f.write(f"{i:2d}. {feat}\n")
    
    f.write("\n\nENGINEERED FEATURES ({}):\n".format(len(new_cols)))
    f.write("-" * 80 + "\n")
    for i, feat in enumerate(new_cols, 1):
        f.write(f"{i:2d}. {feat}\n")
    
    f.write("\n\nFEATURE CATEGORIES:\n")
    f.write("-" * 80 + "\n")
    f.write("1. RFM Features: recency_score, monetary_score, rfm_score, customer_segment\n")
    f.write("2. Behavioral: promo_engagement, promo_variety, is_inactive, is_recent\n")
    f.write("3. Customer Value: is_high_value, is_low_value\n")
    f.write("4. Interactions: offer_channel, referral_recent, highvalue_*, location_channel\n")
    f.write("5. Spending: spending_per_day, spending_category, history_log\n")
    f.write("6. Channel: channel_score, is_digital\n")
    f.write("7. Offer Matching: discount_match, bogo_match, offer_mismatch\n")
    f.write("8. Location: is_urban, is_rural\n")

print(f"✓ Feature list saved to: {feature_list_path}")

print("\n" + "="*80)
print("✅ FEATURE ENGINEERING COMPLETE!")
print("="*80)
print(f"\nDataset ready for modeling:")
print(f"  - Records: {len(df):,}")
print(f"  - Features: {df.shape[1]}")
print(f"  - Target: conversion")
print(f"\nNext step: Run 03_preprocessing_pipeline.py")
