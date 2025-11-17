"""
STEP 1: Preprocessing Pipeline
Script này thực hiện tiền xử lý dữ liệu cho mô hình ML
Input: data/enhanced_data.csv
Output: X_train_processed.csv, X_test_processed.csv, y_train.csv, y_test.csv
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("STEP 1: DATA PREPROCESSING PIPELINE")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n📊 Loading enhanced data...")
df = pd.read_csv('../data/enhanced_data.csv')
print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")

# ============================================================================
# 2. DATA OVERVIEW
# ============================================================================
print("\n📋 Data Info:")
print(f"Shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"\nTarget distribution:")
print(df['conversion'].value_counts())
print(f"Conversion rate: {df['conversion'].mean():.2%}")

# ============================================================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================================================
print("\n🔧 Separating features and target...")

# Drop drink_item and food_item (too many categories, keep only category columns)
X = df.drop(['conversion', 'drink_item', 'food_item'], axis=1)
y = df['conversion']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# ============================================================================
# 4. DEFINE FEATURE TYPES
# ============================================================================
# Numeric features
numeric_features = ['recency', 'history']

# Categorical features
categorical_features = [
    'used_discount', 'used_bogo', 'zip_code', 'is_referral',
    'channel', 'offer', 'seat_usage', 'time_of_day',
    'drink_category', 'food_category'
]

print(f"\nNumeric features ({len(numeric_features)}): {numeric_features}")
print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

# ============================================================================
# 5. TRAIN/TEST SPLIT (STRATIFIED)
# ============================================================================
print("\n Splitting data (80/20, stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print(f"Train set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
print(f"Train conversion rate: {y_train.mean():.2%}")
print(f"Test conversion rate: {y_test.mean():.2%}")

# ============================================================================
# 6. CREATE PREPROCESSING PIPELINE
# ============================================================================
print("\n🔨 Creating preprocessing pipeline...")

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), 
         categorical_features)
    ],
    remainder='drop'
)

# ============================================================================
# 7. FIT AND TRANSFORM
# ============================================================================
print("\n⚙️ Fitting and transforming data...")

# Fit on training data and transform both
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"✅ Processed train shape: {X_train_processed.shape}")
print(f"✅ Processed test shape: {X_test_processed.shape}")

# Get feature names after preprocessing
feature_names = (
    numeric_features + 
    list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
)
print(f"Total features after encoding: {len(feature_names)}")

# ============================================================================
# 8. HANDLE CLASS IMBALANCE (SMOTE)
# ============================================================================
print("\n⚖️ Handling class imbalance with SMOTE...")
print(f"Before SMOTE - Class distribution:")
print(f"  Class 0: {(y_train == 0).sum()}")
print(f"  Class 1: {(y_train == 1).sum()}")

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_processed, y_train)

print(f"After SMOTE - Class distribution:")
print(f"  Class 0: {(y_train_balanced == 0).sum()}")
print(f"  Class 1: {(y_train_balanced == 1).sum()}")
print(f"✅ Balanced train set: {len(X_train_balanced)} samples")

# ============================================================================
# 9. SAVE PROCESSED DATA
# ============================================================================
print("\n💾 Saving processed data...")

# Save as DataFrames with proper column names
pd.DataFrame(X_train_balanced, columns=feature_names).to_csv(
    '../data/X_train_processed.csv', index=False
)
pd.DataFrame(X_test_processed, columns=feature_names).to_csv(
    '../data/X_test_processed.csv', index=False
)
pd.Series(y_train_balanced, name='conversion').to_csv(
    '../data/y_train.csv', index=False
)
pd.Series(y_test, name='conversion').to_csv(
    '../data/y_test.csv', index=False
)

print("✅ X_train_processed.csv")
print("✅ X_test_processed.csv")
print("✅ y_train.csv")
print("✅ y_test.csv")

# Save preprocessor for future use
with open('../models/preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)
print("✅ preprocessor.pkl")

# Save feature names
with open('../models/feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)
print("✅ feature_names.pkl")

# ============================================================================
# 10. SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("📊 PREPROCESSING SUMMARY")
print("=" * 80)
print(f"Original data: {df.shape}")
print(f"Features used: {X.shape[1]}")
print(f"Features after encoding: {len(feature_names)}")
print(f"Train set (balanced): {X_train_balanced.shape}")
print(f"Test set: {X_test_processed.shape}")
print(f"Class balance achieved: 50-50")
print("\n✅ STEP 1 COMPLETED SUCCESSFULLY!")
print("=" * 80)
