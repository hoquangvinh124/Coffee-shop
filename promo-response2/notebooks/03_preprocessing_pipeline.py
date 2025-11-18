"""
Data Preprocessing Pipeline for Model Training
===============================================
Chuẩn bị dữ liệu cho training: encoding, scaling, train/test split, SMOTE
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "data_engineered.csv"
OUTPUT_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

print("="*80)
print("DATA PREPROCESSING PIPELINE")
print("="*80)

# Load engineered data
print("\n1. LOADING ENGINEERED DATA...")
df = pd.read_csv(DATA_PATH)
print(f"✓ Loaded {len(df):,} records with {df.shape[1]} features")

# Separate features and target
print("\n2. SEPARATING FEATURES AND TARGET...")
X = df.drop('conversion', axis=1)
y = df['conversion']

print(f"✓ Features shape: {X.shape}")
print(f"✓ Target shape: {y.shape}")
print(f"✓ Class distribution:")
print(f"   - Class 0: {(y==0).sum():,} ({(y==0).mean()*100:.2f}%)")
print(f"   - Class 1: {(y==1).sum():,} ({(y==1).mean()*100:.2f}%)")

# Feature selection
print("\n3. FEATURE SELECTION...")
print("-" * 80)

# Drop high-cardinality interaction features for initial model
# (These can be tested later if needed)
features_to_drop = ['offer_channel', 'location_channel', 'customer_segment', 'spending_category']

print(f"Dropping high-cardinality features: {features_to_drop}")
X = X.drop(columns=features_to_drop)

print(f"✓ Features after selection: {X.shape[1]}")
print(f"\nSelected features:")
for i, col in enumerate(X.columns, 1):
    print(f"  {i:2d}. {col}")

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

print(f"\n✓ Numerical features ({len(numerical_cols)}): {numerical_cols}")
print(f"✓ Categorical features ({len(categorical_cols)}): {categorical_cols}")

# Encode categorical features
print("\n4. ENCODING CATEGORICAL FEATURES...")
print("-" * 80)

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le
    print(f"✓ Encoded: {col} ({len(le.classes_)} classes)")

# Save encoders
encoder_path = MODELS_DIR / "label_encoders.pkl"
joblib.dump(label_encoders, encoder_path)
print(f"\n✓ Label encoders saved to: {encoder_path}")

# Train/Test split
print("\n5. TRAIN/TEST SPLIT...")
print("-" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print(f"✓ Train set: {X_train.shape[0]:,} records ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"✓ Test set:  {X_test.shape[0]:,} records ({X_test.shape[0]/len(X)*100:.1f}%)")

print(f"\nClass distribution in train set:")
print(f"  - Class 0: {(y_train==0).sum():,} ({(y_train==0).mean()*100:.2f}%)")
print(f"  - Class 1: {(y_train==1).sum():,} ({(y_train==1).mean()*100:.2f}%)")

print(f"\nClass distribution in test set:")
print(f"  - Class 0: {(y_test==0).sum():,} ({(y_test==0).mean()*100:.2f}%)")
print(f"  - Class 1: {(y_test==1).sum():,} ({(y_test==1).mean()*100:.2f}%)")

# Feature scaling
print("\n6. FEATURE SCALING...")
print("-" * 80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for easier handling
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

print(f"✓ Scaled train set: {X_train_scaled.shape}")
print(f"✓ Scaled test set: {X_test_scaled.shape}")

# Save scaler
scaler_path = MODELS_DIR / "scaler.pkl"
joblib.dump(scaler, scaler_path)
print(f"✓ Scaler saved to: {scaler_path}")

# Apply SMOTE to handle class imbalance
print("\n7. APPLYING SMOTE FOR CLASS BALANCING...")
print("-" * 80)

print(f"Before SMOTE:")
print(f"  - Total samples: {len(X_train_scaled):,}")
print(f"  - Class 0: {(y_train==0).sum():,}")
print(f"  - Class 1: {(y_train==1).sum():,}")
print(f"  - Imbalance ratio: {(y_train==0).sum()/(y_train==1).sum():.2f}:1")

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

# Convert back to DataFrame
X_train_balanced = pd.DataFrame(X_train_balanced, columns=X_train_scaled.columns)
y_train_balanced = pd.Series(y_train_balanced)

print(f"\nAfter SMOTE:")
print(f"  - Total samples: {len(X_train_balanced):,}")
print(f"  - Class 0: {(y_train_balanced==0).sum():,}")
print(f"  - Class 1: {(y_train_balanced==1).sum():,}")
print(f"  - Imbalance ratio: 1:1 (balanced)")
print(f"  - New samples created: {len(X_train_balanced) - len(X_train_scaled):,}")

# Save preprocessed data
print("\n8. SAVING PREPROCESSED DATA...")
print("-" * 80)

# Save training data (balanced)
X_train_balanced.to_csv(OUTPUT_DIR / "X_train_balanced.csv", index=False)
y_train_balanced.to_csv(OUTPUT_DIR / "y_train_balanced.csv", index=False, header=True)
print(f"✓ Saved: X_train_balanced.csv ({X_train_balanced.shape})")
print(f"✓ Saved: y_train_balanced.csv ({y_train_balanced.shape})")

# Save original training data (unbalanced) for comparison
X_train_scaled.to_csv(OUTPUT_DIR / "X_train_original.csv", index=False)
y_train.to_csv(OUTPUT_DIR / "y_train_original.csv", index=False, header=True)
print(f"✓ Saved: X_train_original.csv ({X_train_scaled.shape})")
print(f"✓ Saved: y_train_original.csv ({y_train.shape})")

# Save test data
X_test_scaled.to_csv(OUTPUT_DIR / "X_test.csv", index=False)
y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False, header=True)
print(f"✓ Saved: X_test.csv ({X_test_scaled.shape})")
print(f"✓ Saved: y_test.csv ({y_test.shape})")

# Save feature names
feature_names = X_train_balanced.columns.tolist()
feature_names_path = MODELS_DIR / "feature_names.pkl"
joblib.dump(feature_names, feature_names_path)
print(f"✓ Saved: feature_names.pkl ({len(feature_names)} features)")

# Create preprocessing summary
print("\n9. PREPROCESSING SUMMARY")
print("="*80)

summary = f"""
PREPROCESSING PIPELINE SUMMARY
==============================

INPUT DATA:
  - Total records: {len(df):,}
  - Original features: {df.shape[1]}
  - Target: conversion (binary)
  - Class imbalance: {(y==0).sum()/(y==1).sum():.2f}:1

FEATURE SELECTION:
  - Features dropped: {len(features_to_drop)} (high cardinality)
  - Features selected: {len(X.columns)}
  - Numerical features: {len(numerical_cols)}
  - Categorical features: {len(categorical_cols)}

ENCODING:
  - Label encoding applied to categorical features
  - Encoders saved for production use

TRAIN/TEST SPLIT:
  - Train: {len(X_train):,} records (80%)
  - Test: {len(X_test):,} records (20%)
  - Stratified split (maintains class distribution)

SCALING:
  - StandardScaler applied to all features
  - Fit on train, transform on test
  - Scaler saved for production use

CLASS BALANCING (SMOTE):
  - Applied to training set only
  - Original train size: {len(X_train):,}
  - Balanced train size: {len(X_train_balanced):,}
  - New synthetic samples: {len(X_train_balanced) - len(X_train):,}
  - Final ratio: 1:1 (perfect balance)

OUTPUT FILES:
  ✓ X_train_balanced.csv - Training features (SMOTE applied)
  ✓ y_train_balanced.csv - Training labels (balanced)
  ✓ X_train_original.csv - Training features (original)
  ✓ y_train_original.csv - Training labels (original)
  ✓ X_test.csv - Test features
  ✓ y_test.csv - Test labels
  ✓ label_encoders.pkl - Categorical encoders
  ✓ scaler.pkl - Feature scaler
  ✓ feature_names.pkl - Feature list

READY FOR MODELING:
  - Features: {len(feature_names)}
  - Train samples: {len(X_train_balanced):,} (balanced)
  - Test samples: {len(X_test):,}
  - All preprocessing artifacts saved

NEXT STEPS:
  1. Run 04_model_training.py to train ML models
  2. Compare balanced vs unbalanced training
  3. Evaluate on holdout test set
  4. Select best model for deployment
"""

summary_path = OUTPUT_DIR / "preprocessing_summary.txt"
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write(summary)

print(summary)
print(f"\n✓ Summary saved to: {summary_path}")

print("\n" + "="*80)
print("✅ PREPROCESSING COMPLETE!")
print("="*80)
print(f"\nData ready for model training:")
print(f"  - Training samples: {len(X_train_balanced):,} (balanced)")
print(f"  - Test samples: {len(X_test):,}")
print(f"  - Features: {len(feature_names)}")
print(f"\nNext step: Run 04_model_training.py")
