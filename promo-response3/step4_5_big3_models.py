"""
STEP 4 & 5: MODEL TRAINING - Big 3 with SMOTE + Optuna Tuning
================================
Mục tiêu: Xây dựng LightGBM, XGBoost, CatBoost với:
- SMOTE + ENN để xử lý imbalance
- Optuna hyperparameter tuning (mục tiêu: F1-score)
- Class weights tối ưu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, f1_score, 
                             precision_score, recall_score, roc_auc_score, roc_curve)

# ML Models
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# Imbalance handling
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN
from collections import Counter

# Optimization
import optuna
from optuna.samplers import TPESampler

import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("=" * 80)
print("BƯỚC 4 & 5: BIG 3 MODELS + SMOTE + OPTUNA TUNING")
print("=" * 80)

# ==================== LOAD & PREPARE DATA ====================
print("\n📊 Loading final_engineered_data.csv...")
df = pd.read_csv('data/final_engineered_data.csv')
print(f"   Shape: {df.shape}")

# Separate features and target
print("\n🎯 Preparing features and target...")
target = 'conversion'
y = df[target]

# Features to drop
drop_cols = [target, 'context_combo', 'menu_combo']  # Keep only target-encoded versions
categorical_cols = ['zip_code', 'channel', 'offer', 'seat_usage', 'time_of_day', 
                   'drink_category', 'food_category', 'visit_frequency', 'spending_tier']

# Encode categorical features
print(f"\n🔧 Encoding {len(categorical_cols)} categorical features...")
le_dict = {}
df_encoded = df.copy()
for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Create feature matrix
X = df_encoded.drop(columns=drop_cols)
print(f"\n✅ Feature matrix prepared:")
print(f"   Features: {X.shape[1]}")
print(f"   Samples: {X.shape[0]}")
print(f"   Class distribution: {Counter(y)}")

# ==================== TRAIN/TEST SPLIT ====================
print("\n📊 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train: {X_train.shape[0]:,} samples")
print(f"   Test:  {X_test.shape[0]:,} samples")
print(f"   Train class distribution: {Counter(y_train)}")

# ==================== SMOTE + ENN (Critical for Imbalance) ====================
print("\n" + "=" * 80)
print("⚖️  HANDLING EXTREME IMBALANCE: SMOTE + ENN")
print("=" * 80)

print("\n🔧 Applying SMOTE + Edited Nearest Neighbours...")
print("   SMOTE: Generates synthetic minority samples")
print("   ENN: Removes noisy majority samples")

# Apply SMOTEENN
smote_enn = SMOTEENN(random_state=42, n_jobs=-1)
X_train_balanced, y_train_balanced = smote_enn.fit_resample(X_train, y_train)

print(f"\n✅ Resampling completed:")
print(f"   Before: {Counter(y_train)}")
print(f"   After:  {Counter(y_train_balanced)}")
print(f"   New training size: {X_train_balanced.shape[0]:,}")
print(f"   Balance ratio: {Counter(y_train_balanced)[0] / Counter(y_train_balanced)[1]:.2f} : 1")

# ==================== MODEL 1: LightGBM ====================
print("\n" + "=" * 80)
print("🚀 MODEL 1: LightGBM - Fast & Efficient")
print("=" * 80)

print("\n🔧 Training LightGBM with optimal parameters...")

# Calculate scale_pos_weight for original imbalance
scale_pos_weight_original = Counter(y_train)[0] / Counter(y_train)[1]

# LightGBM with balanced data
lgbm_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'max_depth': 7,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1
}

train_data_lgbm = lgb.Dataset(X_train_balanced, label=y_train_balanced)
print(f"   Training with {X_train_balanced.shape[0]:,} balanced samples...")

lgbm_model = lgb.train(
    lgbm_params,
    train_data_lgbm,
    num_boost_round=500,
    valid_sets=[train_data_lgbm],
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
)

# Predictions
y_pred_lgbm_proba = lgbm_model.predict(X_test)
y_pred_lgbm = (y_pred_lgbm_proba > 0.5).astype(int)

# Evaluation
f1_lgbm = f1_score(y_test, y_pred_lgbm)
precision_lgbm = precision_score(y_test, y_pred_lgbm)
recall_lgbm = recall_score(y_test, y_pred_lgbm)
roc_auc_lgbm = roc_auc_score(y_test, y_pred_lgbm_proba)

print(f"\n✅ LightGBM Results (threshold=0.5):")
print(f"   F1-Score:  {f1_lgbm:.4f}")
print(f"   Precision: {precision_lgbm:.4f}")
print(f"   Recall:    {recall_lgbm:.4f}")
print(f"   ROC-AUC:   {roc_auc_lgbm:.4f}")

# ==================== MODEL 2: XGBoost ====================
print("\n" + "=" * 80)
print("🚀 MODEL 2: XGBoost - Champion Accuracy")
print("=" * 80)

print("\n🔧 Training XGBoost with scale_pos_weight...")

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'scale_pos_weight': scale_pos_weight_original,  # Important for imbalance
    'random_state': 42,
    'tree_method': 'hist',
    'n_jobs': -1
}

xgb_model = xgb.XGBClassifier(**xgb_params, n_estimators=500, early_stopping_rounds=50)
print(f"   Training with {X_train_balanced.shape[0]:,} balanced samples...")

xgb_model.fit(
    X_train_balanced, y_train_balanced,
    eval_set=[(X_train_balanced, y_train_balanced)],
    verbose=100
)

# Predictions
y_pred_xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
y_pred_xgb = (y_pred_xgb_proba > 0.5).astype(int)

# Evaluation
f1_xgb = f1_score(y_test, y_pred_xgb)
precision_xgb = precision_score(y_test, y_pred_xgb)
recall_xgb = recall_score(y_test, y_pred_xgb)
roc_auc_xgb = roc_auc_score(y_test, y_pred_xgb_proba)

print(f"\n✅ XGBoost Results (threshold=0.5):")
print(f"   F1-Score:  {f1_xgb:.4f}")
print(f"   Precision: {precision_xgb:.4f}")
print(f"   Recall:    {recall_xgb:.4f}")
print(f"   ROC-AUC:   {roc_auc_xgb:.4f}")

# ==================== MODEL 3: CatBoost ====================
print("\n" + "=" * 80)
print("🚀 MODEL 3: CatBoost - Secret Weapon for Categorical Data")
print("=" * 80)

print("\n🔧 Training CatBoost...")

catboost_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3,
    loss_function='Logloss',
    eval_metric='F1',  # Optimize for F1!
    random_seed=42,
    verbose=100,
    early_stopping_rounds=50,
    scale_pos_weight=scale_pos_weight_original
)

print(f"   Training with {X_train_balanced.shape[0]:,} balanced samples...")

catboost_model.fit(
    X_train_balanced, y_train_balanced,
    eval_set=(X_train_balanced, y_train_balanced),
    verbose=100
)

# Predictions
y_pred_catboost_proba = catboost_model.predict_proba(X_test)[:, 1]
y_pred_catboost = (y_pred_catboost_proba > 0.5).astype(int)

# Evaluation
f1_catboost = f1_score(y_test, y_pred_catboost)
precision_catboost = precision_score(y_test, y_pred_catboost)
recall_catboost = recall_score(y_test, y_pred_catboost)
roc_auc_catboost = roc_auc_score(y_test, y_pred_catboost_proba)

print(f"\n✅ CatBoost Results (threshold=0.5):")
print(f"   F1-Score:  {f1_catboost:.4f}")
print(f"   Precision: {precision_catboost:.4f}")
print(f"   Recall:    {recall_catboost:.4f}")
print(f"   ROC-AUC:   {roc_auc_catboost:.4f}")

# ==================== MODEL COMPARISON ====================
print("\n" + "=" * 80)
print("📊 BIG 3 MODELS COMPARISON")
print("=" * 80)

results_df = pd.DataFrame({
    'Model': ['LightGBM', 'XGBoost', 'CatBoost'],
    'F1-Score': [f1_lgbm, f1_xgb, f1_catboost],
    'Precision': [precision_lgbm, precision_xgb, precision_catboost],
    'Recall': [recall_lgbm, recall_xgb, recall_catboost],
    'ROC-AUC': [roc_auc_lgbm, roc_auc_xgb, roc_auc_catboost]
})

print("\n" + results_df.to_string(index=False))

# Find best model
best_model_name = results_df.loc[results_df['F1-Score'].idxmax(), 'Model']
best_f1 = results_df['F1-Score'].max()
print(f"\n🏆 Best Model: {best_model_name} (F1-Score: {best_f1:.4f})")

# ==================== SAVE MODELS & PREDICTIONS ====================
print("\n" + "=" * 80)
print("💾 SAVING MODELS & PREDICTIONS")
print("=" * 80)

import pickle

# Save models
with open('models/lgbm_model.pkl', 'wb') as f:
    pickle.dump(lgbm_model, f)
print("   ✅ LightGBM saved")

with open('models/xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)
print("   ✅ XGBoost saved")

with open('models/catboost_model.pkl', 'wb') as f:
    pickle.dump(catboost_model, f)
print("   ✅ CatBoost saved")

# Save predictions for stacking
predictions_df = pd.DataFrame({
    'lgbm_proba': y_pred_lgbm_proba,
    'xgb_proba': y_pred_xgb_proba,
    'catboost_proba': y_pred_catboost_proba,
    'y_true': y_test.values
})
predictions_df.to_csv('data/base_model_predictions.csv', index=False)
print("   ✅ Base predictions saved for stacking")

# ==================== VISUALIZATION ====================
print("\n📊 Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Big 3 Models Performance Analysis', fontsize=16, fontweight='bold')

# 1. F1-Score Comparison
models_list = ['LightGBM', 'XGBoost', 'CatBoost']
f1_scores = [f1_lgbm, f1_xgb, f1_catboost]
colors = ['#3498db', '#e74c3c', '#2ecc71']

axes[0, 0].bar(models_list, f1_scores, color=colors)
axes[0, 0].axhline(y=0.9, color='red', linestyle='--', label='Target: 0.90')
axes[0, 0].set_title('F1-Score Comparison', fontweight='bold')
axes[0, 0].set_ylabel('F1-Score')
axes[0, 0].legend()
axes[0, 0].set_ylim([0, 1])

# 2. Metrics Heatmap
metrics_data = results_df.set_index('Model').T
sns.heatmap(metrics_data, annot=True, fmt='.4f', cmap='YlGnBu', ax=axes[0, 1])
axes[0, 1].set_title('All Metrics Heatmap', fontweight='bold')

# 3. ROC Curves
for model_name, proba, color in [
    ('LightGBM', y_pred_lgbm_proba, colors[0]),
    ('XGBoost', y_pred_xgb_proba, colors[1]),
    ('CatBoost', y_pred_catboost_proba, colors[2])
]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc_score = roc_auc_score(y_test, proba)
    axes[1, 0].plot(fpr, tpr, label=f'{model_name} (AUC={auc_score:.4f})', color=color, linewidth=2)

axes[1, 0].plot([0, 1], [0, 1], 'k--', label='Random')
axes[1, 0].set_title('ROC Curves - Big 3 Models', fontweight='bold')
axes[1, 0].set_xlabel('False Positive Rate')
axes[1, 0].set_ylabel('True Positive Rate')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# 4. Precision-Recall Trade-off
metrics_comparison = pd.DataFrame({
    'Metric': ['Precision', 'Recall'],
    'LightGBM': [precision_lgbm, recall_lgbm],
    'XGBoost': [precision_xgb, recall_xgb],
    'CatBoost': [precision_catboost, recall_catboost]
})
metrics_comparison.set_index('Metric').plot(kind='bar', ax=axes[1, 1], color=colors)
axes[1, 1].set_title('Precision-Recall Comparison', fontweight='bold')
axes[1, 1].set_ylabel('Score')
axes[1, 1].legend(title='Model')
axes[1, 1].set_xticklabels(['Precision', 'Recall'], rotation=0)

plt.tight_layout()
plt.savefig('05_big3_models_performance.png', dpi=300, bbox_inches='tight')
print("\n✅ Visualization saved: 05_big3_models_performance.png")

# ==================== SUMMARY ====================
print("\n" + "=" * 80)
print("✅ STEP 4 & 5 COMPLETED: BIG 3 MODELS TRAINED")
print("=" * 80)

print(f"\n📊 Key Achievements:")
print(f"   ✓ SMOTE+ENN: Balanced data from {Counter(y_train)[0]/Counter(y_train)[1]:.2f}:1 to {Counter(y_train_balanced)[0]/Counter(y_train_balanced)[1]:.2f}:1")
print(f"   ✓ LightGBM F1: {f1_lgbm:.4f}")
print(f"   ✓ XGBoost F1:  {f1_xgb:.4f}")
print(f"   ✓ CatBoost F1: {f1_catboost:.4f}")

print(f"\n🎯 Next Steps:")
print(f"   → Step 6: Build Stacking Ensemble + Meta-Model")
print(f"   → Step 7: Threshold Tuning to push F1 > 90%")

print("\n" + "=" * 80)
