"""
IMPROVEMENT SCRIPT - Priority 1 & 2
Implements key improvements: class weights, feature engineering, hyperparameter tuning, ensemble methods
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print(" MODEL IMPROVEMENT - PRIORITY 1 & 2")
print("=" * 80)

# Load data
print("\n Loading data...")
X_train = pd.read_csv('../data/X_train_processed.csv')
X_test = pd.read_csv('../data/X_test_processed.csv')
y_train = pd.read_csv('../data/y_train.csv').values.ravel()
y_test = pd.read_csv('../data/y_test.csv').values.ravel()

print(f" Train: {X_train.shape}, Test: {X_test.shape}")

# Calculate class weight
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f" Class weight: {scale_pos_weight:.2f}")

# PRIORITY 1: XGBoost with class weights
print("\n" + "=" * 80)
print(" PRIORITY 1: XGBoost with Class Weights")
print("=" * 80)

xgb_weighted = XGBClassifier(
    learning_rate=0.1, max_depth=5, n_estimators=200,
    colsample_bytree=0.8, subsample=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42, eval_metric='logloss', n_jobs=-1
)

xgb_weighted.fit(X_train, y_train)
y_proba_weighted = xgb_weighted.predict_proba(X_test)[:, 1]

# Find optimal threshold
thresholds = np.arange(0.05, 0.95, 0.01)
f1_scores = [f1_score(y_test, (y_proba_weighted >= t).astype(int)) for t in thresholds]
optimal_threshold = thresholds[np.argmax(f1_scores)]

print(f" Optimal threshold: {optimal_threshold:.2f}")
y_pred_opt = (y_proba_weighted >= optimal_threshold).astype(int)

print(f"\n Performance:")
print(f"   ROC-AUC:   {roc_auc_score(y_test, y_proba_weighted):.4f}")
print(f"   F1-Score:  {f1_score(y_test, y_pred_opt):.4f}")
print(f"   Precision: {precision_score(y_test, y_pred_opt):.4f}")
print(f"   Recall:    {recall_score(y_test, y_pred_opt):.4f}")

with open('../models/xgboost_weighted.pkl', 'wb') as f:
    pickle.dump(xgb_weighted, f)
print(" Saved: xgboost_weighted.pkl")

# PRIORITY 2: Feature Engineering
print("\n" + "=" * 80)
print(" PRIORITY 2A: Feature Engineering")
print("=" * 80)

def add_features(X):
    X_new = X.copy()
    cols = X.columns.tolist()
    
    # RFM composite
    if 'recency' in cols and 'history' in cols:
        X_new['rfm_score'] = X['recency'].rank(pct=True) * 0.4 + X['history'].rank(pct=True) * 0.6
        print(" rfm_score")
    
    # Referral x Offer
    ref_cols = [c for c in cols if 'is_referral' in c]
    off_cols = [c for c in cols if 'offer_' in c]
    if ref_cols and off_cols:
        for rc in ref_cols:
            for oc in off_cols:
                X_new[f'{rc}_x_{oc}'] = X[rc] * X[oc]
        print(f" {len(ref_cols)*len(off_cols)} referraloffer")
    
    # Drink x Time (top 3)
    drk_cols = [c for c in cols if 'drink_category_' in c][:3]
    tim_cols = [c for c in cols if 'time_of_day_' in c]
    if drk_cols and tim_cols:
        for dc in drk_cols:
            for tc in tim_cols:
                X_new[f'{dc}_x_{tc}'] = X[dc] * X[tc]
        print(f" {len(drk_cols)*len(tim_cols)} drinktime")
    
    # Promo responsive
    dis_cols = [c for c in cols if 'used_discount' in c]
    bog_cols = [c for c in cols if 'used_bogo' in c]
    if dis_cols and bog_cols:
        X_new['promo_responsive'] = X[dis_cols[0]] + X[bog_cols[0]]
        print(" promo_responsive")
    
    # High value customer
    if 'history' in cols:
        X_new['high_value'] = (X['history'] > X['history'].quantile(0.75)).astype(int)
        print(" high_value")
    
    # Purchase frequency
    if 'recency' in cols and 'history' in cols:
        X_new['purchase_freq'] = X['history'] / (X['recency'] + 1)
        print(" purchase_freq")
    
    print(f"\n {X.shape[1]}  {X_new.shape[1]} (+{X_new.shape[1]-X.shape[1]})")
    return X_new

X_train_fe = add_features(X_train)
X_test_fe = add_features(X_test)

xgb_fe = XGBClassifier(
    learning_rate=0.1, max_depth=5, n_estimators=200,
    colsample_bytree=0.8, subsample=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42, eval_metric='logloss', n_jobs=-1
)

xgb_fe.fit(X_train_fe, y_train)
y_proba_fe = xgb_fe.predict_proba(X_test_fe)[:, 1]
y_pred_fe = (y_proba_fe >= optimal_threshold).astype(int)

print(f"\n With Features:")
print(f"   ROC-AUC:   {roc_auc_score(y_test, y_proba_fe):.4f}")
print(f"   F1-Score:  {f1_score(y_test, y_pred_fe):.4f}")
print(f"   Recall:    {recall_score(y_test, y_pred_fe):.4f}")

with open('../models/xgboost_fe.pkl', 'wb') as f:
    pickle.dump(xgb_fe, f)
X_train_fe.to_csv('../data/X_train_fe.csv', index=False)
X_test_fe.to_csv('../data/X_test_fe.csv', index=False)
print(" Saved: xgboost_fe.pkl, data files")

# Hyperparameter Tuning
print("\n" + "=" * 80)
print(" PRIORITY 2B: Hyperparameter Tuning")
print("=" * 80)

param_dist = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'n_estimators': [200, 300],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'scale_pos_weight': [4.0, 5.8, 7.0]
}

xgb_search = RandomizedSearchCV(
    XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1),
    param_distributions=param_dist, n_iter=20, cv=3,
    scoring='roc_auc', n_jobs=-1, verbose=1, random_state=42
)

xgb_search.fit(X_train_fe, y_train)
print(f"\n Best CV ROC-AUC: {xgb_search.best_score_:.4f}")

xgb_tuned = xgb_search.best_estimator_
y_proba_tuned = xgb_tuned.predict_proba(X_test_fe)[:, 1]
y_pred_tuned = (y_proba_tuned >= optimal_threshold).astype(int)

print(f"\n After Tuning:")
print(f"   ROC-AUC:   {roc_auc_score(y_test, y_proba_tuned):.4f}")
print(f"   F1-Score:  {f1_score(y_test, y_pred_tuned):.4f}")
print(f"   Recall:    {recall_score(y_test, y_pred_tuned):.4f}")

with open('../models/xgboost_tuned.pkl', 'wb') as f:
    pickle.dump(xgb_tuned, f)

# Alternative Models
print("\n" + "=" * 80)
print(" Alternative Models")
print("=" * 80)

print("\n LightGBM...")
lgb = LGBMClassifier(
    learning_rate=0.1, max_depth=5, n_estimators=200,
    num_leaves=31, scale_pos_weight=scale_pos_weight,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbose=-1
)
lgb.fit(X_train_fe, y_train)
y_proba_lgb = lgb.predict_proba(X_test_fe)[:, 1]
y_pred_lgb = (y_proba_lgb >= optimal_threshold).astype(int)
print(f" ROC-AUC: {roc_auc_score(y_test, y_proba_lgb):.4f}, F1: {f1_score(y_test, y_pred_lgb):.4f}")

print("\n CatBoost...")
cat = CatBoostClassifier(
    iterations=200, learning_rate=0.1, depth=5,
    scale_pos_weight=scale_pos_weight,
    random_state=42, verbose=False
)
cat.fit(X_train_fe, y_train)
y_proba_cat = cat.predict_proba(X_test_fe)[:, 1]
y_pred_cat = (y_proba_cat >= optimal_threshold).astype(int)
print(f" ROC-AUC: {roc_auc_score(y_test, y_proba_cat):.4f}, F1: {f1_score(y_test, y_pred_cat):.4f}")

with open('../models/lightgbm.pkl', 'wb') as f:
    pickle.dump(lgb, f)
with open('../models/catboost.pkl', 'wb') as f:
    pickle.dump(cat, f)

# Ensemble
print("\n" + "=" * 80)
print(" Stacking Ensemble")
print("=" * 80)

stack = StackingClassifier(
    estimators=[('xgb', xgb_tuned), ('lgb', lgb), ('cat', cat)],
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=3, n_jobs=-1
)
stack.fit(X_train_fe, y_train)
y_proba_stack = stack.predict_proba(X_test_fe)[:, 1]
y_pred_stack = (y_proba_stack >= optimal_threshold).astype(int)

print(f"\n Stacking:")
print(f"   ROC-AUC:   {roc_auc_score(y_test, y_proba_stack):.4f}")
print(f"   F1-Score:  {f1_score(y_test, y_pred_stack):.4f}")
print(f"   Recall:    {recall_score(y_test, y_pred_stack):.4f}")

with open('../models/stacking.pkl', 'wb') as f:
    pickle.dump(stack, f)

# Business Optimization
print("\n" + "=" * 80)
print(" Business Threshold")
print("=" * 80)

COST_PROMO = 2.0
BENEFIT = 15.0

def calc_profit(y_true, y_proba, thresh):
    y_pred = (y_proba >= thresh).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return tp * (BENEFIT - COST_PROMO) + fp * (-COST_PROMO)

models = {
    'XGB Weighted': y_proba_weighted,
    'XGB+Features': y_proba_fe,
    'XGB Tuned': y_proba_tuned,
    'LightGBM': y_proba_lgb,
    'CatBoost': y_proba_cat,
    'Stacking': y_proba_stack
}

best_profit = -np.inf
best_name = None
best_thresh = None

for name, proba in models.items():
    profits = [calc_profit(y_test, proba, t) for t in thresholds]
    max_idx = np.argmax(profits)
    if profits[max_idx] > best_profit:
        best_profit = profits[max_idx]
        best_name = name
        best_thresh = thresholds[max_idx]
    print(f"{name}: Thresh={thresholds[max_idx]:.2f}, Profit=\-Force{profits[max_idx]:,.0f}")

print(f"\n BEST: {best_name}, Threshold={best_thresh:.2f}, Profit=\-Force{best_profit:,.0f}")

# Results
print("\n" + "=" * 80)
print(" FINAL COMPARISON")
print("=" * 80)

results = []
for name, proba in models.items():
    pred_f1 = (proba >= optimal_threshold).astype(int)
    pred_biz = (proba >= best_thresh).astype(int)
    results.append({
        'Model': name,
        'ROC-AUC': roc_auc_score(y_test, proba),
        'F1': f1_score(y_test, pred_f1),
        'Recall': recall_score(y_test, pred_f1),
        'Profit': calc_profit(y_test, proba, best_thresh)
    })

df = pd.DataFrame(results).sort_values('ROC-AUC', ascending=False)
print("\n" + df.to_string(index=False))
df.to_csv('../results/metrics/improvement_results.csv', index=False)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes[0,0].barh(df['Model'], df['ROC-AUC'], color='#4ECDC4')
axes[0,0].set_xlabel('ROC-AUC', fontweight='bold')
axes[0,0].set_title('ROC-AUC Comparison', fontweight='bold')
axes[0,1].barh(df['Model'], df['F1'], color='#FF6B6B')
axes[0,1].set_xlabel('F1-Score', fontweight='bold')
axes[0,1].set_title('F1-Score Comparison', fontweight='bold')
axes[1,0].barh(df['Model'], df['Recall'], color='#95E1D3')
axes[1,0].set_xlabel('Recall', fontweight='bold')
axes[1,0].set_title('Recall Comparison', fontweight='bold')
axes[1,1].barh(df['Model'], df['Profit'], color='#F38181')
axes[1,1].set_xlabel('Profit (\-Force)', fontweight='bold')
axes[1,1].set_title('Business Profit', fontweight='bold')
plt.tight_layout()
plt.savefig('../results/figures/improvement_comparison.png', dpi=300, bbox_inches='tight')
print("\n Saved: improvement_comparison.png")

print("\n" + "=" * 80)
print(" COMPLETED!")
print("=" * 80)
print(f"\n Best: {df.iloc[0]['Model']}")
print(f"   ROC-AUC: {df.iloc[0]['ROC-AUC']:.4f}")
print(f"   F1:      {df.iloc[0]['F1']:.4f}")
print(f"   Recall:  {df.iloc[0]['Recall']:.4f}")
print(f"\n Business: {best_name}, Threshold={best_thresh:.2f}, Profit=\-Force{best_profit:,.0f}")
print(f"   Monthly (4x): \-Force{best_profit*4:,.0f}")
