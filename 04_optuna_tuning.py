"""
Hyperparameter Tuning with Optuna
Find best parameters for LightGBM, XGBoost, and Random Forest
"""

import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("HYPERPARAMETER TUNING WITH OPTUNA")
print("=" * 80)

# Load data
print("\n[1] Loading data...")
df = pd.read_csv('coffee_shop_revenue1.csv')
X = df.drop('Daily_Revenue', axis=1)
y = df['Daily_Revenue']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}")

# ============================================================================
# OPTUNA OBJECTIVES
# ============================================================================

def objective_lightgbm(trial):
    """Optuna objective for LightGBM"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    model = lgb.LGBMRegressor(**params)

    # Cross-validation
    scores = cross_val_score(model, X_train, y_train, cv=5,
                            scoring='r2', n_jobs=-1)

    return scores.mean()

def objective_xgboost(trial):
    """Optuna objective for XGBoost"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'random_state': 42,
        'n_jobs': -1
    }

    model = xgb.XGBRegressor(**params)

    # Cross-validation
    scores = cross_val_score(model, X_train, y_train, cv=5,
                            scoring='r2', n_jobs=-1)

    return scores.mean()

def objective_random_forest(trial):
    """Optuna objective for Random Forest"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'random_state': 42,
        'n_jobs': -1
    }

    model = RandomForestRegressor(**params)

    # Cross-validation
    scores = cross_val_score(model, X_train, y_train, cv=5,
                            scoring='r2', n_jobs=-1)

    return scores.mean()

# ============================================================================
# OPTIMIZE
# ============================================================================

print("\n" + "=" * 80)
print("OPTIMIZING HYPERPARAMETERS")
print("=" * 80)

n_trials = 50  # Number of trials per model

# 1. LightGBM
print("\n[2] Optimizing LightGBM...")
print(f"   Running {n_trials} trials...")

study_lgb = optuna.create_study(direction='maximize', study_name='LightGBM')
study_lgb.optimize(objective_lightgbm, n_trials=n_trials, show_progress_bar=True)

print(f"\n✓ Best R² Score: {study_lgb.best_value:.4f}")
print(f"✓ Best Parameters:")
for key, value in study_lgb.best_params.items():
    print(f"   {key}: {value}")

# 2. XGBoost
print("\n[3] Optimizing XGBoost...")
print(f"   Running {n_trials} trials...")

study_xgb = optuna.create_study(direction='maximize', study_name='XGBoost')
study_xgb.optimize(objective_xgboost, n_trials=n_trials, show_progress_bar=True)

print(f"\n✓ Best R² Score: {study_xgb.best_value:.4f}")
print(f"✓ Best Parameters:")
for key, value in study_xgb.best_params.items():
    print(f"   {key}: {value}")

# 3. Random Forest
print("\n[4] Optimizing Random Forest...")
print(f"   Running {n_trials} trials...")

study_rf = optuna.create_study(direction='maximize', study_name='RandomForest')
study_rf.optimize(objective_random_forest, n_trials=n_trials, show_progress_bar=True)

print(f"\n✓ Best R² Score: {study_rf.best_value:.4f}")
print(f"✓ Best Parameters:")
for key, value in study_rf.best_params.items():
    print(f"   {key}: {value}")

# ============================================================================
# TRAIN FINAL MODELS WITH BEST PARAMS
# ============================================================================

print("\n" + "=" * 80)
print("TRAINING FINAL MODELS WITH BEST PARAMETERS")
print("=" * 80)

# LightGBM
print("\n[5] Training LightGBM with best params...")
lgb_best = lgb.LGBMRegressor(**study_lgb.best_params)
lgb_best.fit(X_train, y_train)
y_pred_lgb = lgb_best.predict(X_test)

r2_lgb = r2_score(y_test, y_pred_lgb)
mape_lgb = mean_absolute_percentage_error(y_test, y_pred_lgb) * 100
rmse_lgb = np.sqrt(mean_squared_error(y_test, y_pred_lgb))

print(f"   R²:   {r2_lgb:.4f}")
print(f"   MAPE: {mape_lgb:.2f}%")
print(f"   RMSE: ${rmse_lgb:.2f}")

# XGBoost
print("\n[6] Training XGBoost with best params...")
xgb_best = xgb.XGBRegressor(**study_xgb.best_params)
xgb_best.fit(X_train, y_train)
y_pred_xgb = xgb_best.predict(X_test)

r2_xgb = r2_score(y_test, y_pred_xgb)
mape_xgb = mean_absolute_percentage_error(y_test, y_pred_xgb) * 100
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))

print(f"   R²:   {r2_xgb:.4f}")
print(f"   MAPE: {mape_xgb:.2f}%")
print(f"   RMSE: ${rmse_xgb:.2f}")

# Random Forest
print("\n[7] Training Random Forest with best params...")
rf_best = RandomForestRegressor(**study_rf.best_params)
rf_best.fit(X_train, y_train)
y_pred_rf = rf_best.predict(X_test)

r2_rf = r2_score(y_test, y_pred_rf)
mape_rf = mean_absolute_percentage_error(y_test, y_pred_rf) * 100
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

print(f"   R²:   {r2_rf:.4f}")
print(f"   MAPE: {mape_rf:.2f}%")
print(f"   RMSE: ${rmse_rf:.2f}")

# ============================================================================
# COMPARISON
# ============================================================================

print("\n" + "=" * 80)
print("FINAL COMPARISON - TUNED MODELS")
print("=" * 80)

results = pd.DataFrame({
    'Model': ['LightGBM', 'XGBoost', 'Random Forest'],
    'R2': [r2_lgb, r2_xgb, r2_rf],
    'MAPE': [mape_lgb, mape_xgb, mape_rf],
    'RMSE': [rmse_lgb, rmse_xgb, rmse_rf]
}).sort_values('R2', ascending=False)

print("\n" + results.to_string(index=False))

# ============================================================================
# SAVE BEST MODEL
# ============================================================================

print("\n[8] Saving best model...")

best_model_name = results.iloc[0]['Model']
models = {
    'LightGBM': lgb_best,
    'XGBoost': xgb_best,
    'Random Forest': rf_best
}
best_model = models[best_model_name]

# Save model
with open('models/best_model_tuned.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Save model info
model_info = {
    'model_name': best_model_name,
    'metrics': {
        'R2': results.iloc[0]['R2'],
        'MAPE': results.iloc[0]['MAPE'],
        'RMSE': results.iloc[0]['RMSE']
    },
    'feature_names': X.columns.tolist(),
    'best_params': study_lgb.best_params if best_model_name == 'LightGBM'
                   else study_xgb.best_params if best_model_name == 'XGBoost'
                   else study_rf.best_params
}

with open('models/model_info_tuned.pkl', 'wb') as f:
    pickle.dump(model_info, f)

# Save all best params
best_params = {
    'LightGBM': study_lgb.best_params,
    'XGBoost': study_xgb.best_params,
    'Random Forest': study_rf.best_params
}

with open('models/best_params.pkl', 'wb') as f:
    pickle.dump(best_params, f)

print(f"✓ Best model ({best_model_name}) saved")
print("✓ Model info saved")
print("✓ Best parameters saved")

# Save results
results.to_csv('results/tuned_model_comparison.csv', index=False)
print("✓ Results saved to results/tuned_model_comparison.csv")

print("\n" + "=" * 80)
print("OPTIMIZATION COMPLETE!")
print("=" * 80)

print(f"\n🏆 Winner: {best_model_name}")
print(f"   R²:   {results.iloc[0]['R2']:.4f}")
print(f"   MAPE: {results.iloc[0]['MAPE']:.2f}%")
print(f"   RMSE: ${results.iloc[0]['RMSE']:.2f}")

print("\n📁 Files created:")
print("   - models/best_model_tuned.pkl")
print("   - models/model_info_tuned.pkl")
print("   - models/best_params.pkl")
print("   - results/tuned_model_comparison.csv")

print("\n" + "=" * 80)
