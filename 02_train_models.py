"""
Coffee Shop Revenue Prediction - Model Training & Evaluation
Train multiple models and compare performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import xgboost as xgb
import lightgbm as lgb
import pickle
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')

print("=" * 80)
print("COFFEE SHOP REVENUE PREDICTION - MODEL TRAINING")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1] Loading data...")
df = pd.read_csv('coffee_shop_revenue1.csv')
print(f"✓ Loaded {len(df)} rows")

# ============================================================================
# 2. PREPARE FEATURES AND TARGET
# ============================================================================
print("\n[2] Preparing features and target...")

# Separate features and target
X = df.drop('Daily_Revenue', axis=1)
y = df['Daily_Revenue']

feature_names = X.columns.tolist()
print(f"✓ Features: {len(feature_names)}")
print(f"  {feature_names}")

# ============================================================================
# 3. TRAIN-TEST SPLIT
# ============================================================================
print("\n[3] Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"✓ Train set: {len(X_train)} samples")
print(f"✓ Test set:  {len(X_test)} samples")
print(f"  Train revenue: ${y_train.mean():.2f} ± ${y_train.std():.2f}")
print(f"  Test revenue:  ${y_test.mean():.2f} ± ${y_test.std():.2f}")

# ============================================================================
# 4. FEATURE SCALING (for Linear Regression)
# ============================================================================
print("\n[4] Scaling features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Scaler saved to models/scaler.pkl")

# ============================================================================
# 5. TRAIN MODELS
# ============================================================================
print("\n" + "=" * 80)
print("[5] TRAINING MODELS")
print("=" * 80)

models = {}
results = []

# 5.1 Linear Regression
print("\n[5.1] Training Linear Regression...")
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
models['Linear Regression'] = lr

y_pred_lr = lr.predict(X_test_scaled)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
mape_lr = mean_absolute_percentage_error(y_test, y_pred_lr) * 100
r2_lr = r2_score(y_test, y_pred_lr)

results.append({
    'Model': 'Linear Regression',
    'MAE': mae_lr,
    'RMSE': rmse_lr,
    'MAPE': mape_lr,
    'R2': r2_lr
})

print(f"  MAE:  ${mae_lr:.2f}")
print(f"  RMSE: ${rmse_lr:.2f}")
print(f"  MAPE: {mape_lr:.2f}%")
print(f"  R²:   {r2_lr:.4f}")

# 5.2 Random Forest
print("\n[5.2] Training Random Forest...")
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
models['Random Forest'] = rf

y_pred_rf = rf.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mape_rf = mean_absolute_percentage_error(y_test, y_pred_rf) * 100
r2_rf = r2_score(y_test, y_pred_rf)

results.append({
    'Model': 'Random Forest',
    'MAE': mae_rf,
    'RMSE': rmse_rf,
    'MAPE': mape_rf,
    'R2': r2_rf
})

print(f"  MAE:  ${mae_rf:.2f}")
print(f"  RMSE: ${rmse_rf:.2f}")
print(f"  MAPE: {mape_rf:.2f}%")
print(f"  R²:   {r2_rf:.4f}")

# 5.3 XGBoost
print("\n[5.3] Training XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)
models['XGBoost'] = xgb_model

y_pred_xgb = xgb_model.predict(X_test)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
mape_xgb = mean_absolute_percentage_error(y_test, y_pred_xgb) * 100
r2_xgb = r2_score(y_test, y_pred_xgb)

results.append({
    'Model': 'XGBoost',
    'MAE': mae_xgb,
    'RMSE': rmse_xgb,
    'MAPE': mape_xgb,
    'R2': r2_xgb
})

print(f"  MAE:  ${mae_xgb:.2f}")
print(f"  RMSE: ${rmse_xgb:.2f}")
print(f"  MAPE: {mape_xgb:.2f}%")
print(f"  R²:   {r2_xgb:.4f}")

# 5.4 LightGBM
print("\n[5.4] Training LightGBM...")
lgb_model = lgb.LGBMRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgb_model.fit(X_train, y_train)
models['LightGBM'] = lgb_model

y_pred_lgb = lgb_model.predict(X_test)
mae_lgb = mean_absolute_error(y_test, y_pred_lgb)
rmse_lgb = np.sqrt(mean_squared_error(y_test, y_pred_lgb))
mape_lgb = mean_absolute_percentage_error(y_test, y_pred_lgb) * 100
r2_lgb = r2_score(y_test, y_pred_lgb)

results.append({
    'Model': 'LightGBM',
    'MAE': mae_lgb,
    'RMSE': rmse_lgb,
    'MAPE': mape_lgb,
    'R2': r2_lgb
})

print(f"  MAE:  ${mae_lgb:.2f}")
print(f"  RMSE: ${rmse_lgb:.2f}")
print(f"  MAPE: {mape_lgb:.2f}%")
print(f"  R²:   {r2_lgb:.4f}")

# ============================================================================
# 6. COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("[6] MODEL COMPARISON")
print("=" * 80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('R2', ascending=False)

print("\nRanked by R² Score:")
print("-" * 80)
print(f"{'Model':<20} {'MAE':<12} {'RMSE':<12} {'MAPE':<12} {'R²':<10}")
print("-" * 80)
for _, row in results_df.iterrows():
    star = "⭐" if row['R2'] == results_df['R2'].max() else "  "
    print(f"{row['Model']:<20} ${row['MAE']:<11.2f} ${row['RMSE']:<11.2f} {row['MAPE']:<11.2f}% {row['R2']:<9.4f} {star}")
print("-" * 80)

# Save results
results_df.to_csv('results/model_comparison.csv', index=False)
print("\n✓ Results saved to results/model_comparison.csv")

# ============================================================================
# 7. SAVE BEST MODEL
# ============================================================================
print("\n[7] Saving best model...")

best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]

with open('models/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print(f"✓ Best model ({best_model_name}) saved to models/best_model.pkl")

# Save model info
model_info = {
    'model_name': best_model_name,
    'metrics': results_df.iloc[0].to_dict(),
    'feature_names': feature_names
}

with open('models/model_info.pkl', 'wb') as f:
    pickle.dump(model_info, f)

print("✓ Model info saved to models/model_info.pkl")

# ============================================================================
# 8. FEATURE IMPORTANCE (for tree-based models)
# ============================================================================
print("\n[8] Feature importance analysis...")

if best_model_name in ['Random Forest', 'XGBoost', 'LightGBM']:
    if best_model_name == 'Random Forest':
        importances = best_model.feature_importances_
    elif best_model_name == 'XGBoost':
        importances = best_model.feature_importances_
    else:  # LightGBM
        importances = best_model.feature_importances_

    feature_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    print("\nTop 5 Most Important Features:")
    for i, row in feature_imp.head(5).iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.4f}")

    feature_imp.to_csv('results/feature_importance.csv', index=False)
    print("\n✓ Feature importance saved to results/feature_importance.csv")

# ============================================================================
# 9. VISUALIZATIONS
# ============================================================================
print("\n[9] Creating visualizations...")

fig = plt.figure(figsize=(18, 10))

# 9.1 Model Comparison - R²
ax1 = plt.subplot(2, 3, 1)
bars = ax1.barh(results_df['Model'], results_df['R2'], color='skyblue')
bars[0].set_color('gold')  # Highlight best
ax1.set_xlabel('R² Score')
ax1.set_title('Model Comparison - R² Score', fontweight='bold', fontsize=12)
ax1.axvline(x=0, color='black', linewidth=0.5)
for i, v in enumerate(results_df['R2']):
    ax1.text(v, i, f' {v:.4f}', va='center', fontsize=9)
ax1.grid(True, alpha=0.3)

# 9.2 Model Comparison - MAPE
ax2 = plt.subplot(2, 3, 2)
bars = ax2.barh(results_df['Model'], results_df['MAPE'], color='lightcoral')
bars[results_df['MAPE'].argmin()].set_color('gold')
ax2.set_xlabel('MAPE (%)')
ax2.set_title('Model Comparison - MAPE', fontweight='bold', fontsize=12)
for i, v in enumerate(results_df['MAPE']):
    ax2.text(v, i, f' {v:.2f}%', va='center', fontsize=9)
ax2.grid(True, alpha=0.3)

# 9.3 Model Comparison - RMSE
ax3 = plt.subplot(2, 3, 3)
bars = ax3.barh(results_df['Model'], results_df['RMSE'], color='lightgreen')
bars[results_df['RMSE'].argmin()].set_color('gold')
ax3.set_xlabel('RMSE ($)')
ax3.set_title('Model Comparison - RMSE', fontweight='bold', fontsize=12)
for i, v in enumerate(results_df['RMSE']):
    ax3.text(v, i, f' ${v:.2f}', va='center', fontsize=9)
ax3.grid(True, alpha=0.3)

# 9.4 Predictions vs Actual (Best Model)
ax4 = plt.subplot(2, 3, 4)
if best_model_name == 'Linear Regression':
    y_pred_best = best_model.predict(X_test_scaled)
else:
    y_pred_best = best_model.predict(X_test)

ax4.scatter(y_test, y_pred_best, alpha=0.5, s=20)
ax4.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax4.set_xlabel('Actual Revenue ($)')
ax4.set_ylabel('Predicted Revenue ($)')
ax4.set_title(f'Predictions vs Actual - {best_model_name}\n(R²={results_df.iloc[0]["R2"]:.4f})',
              fontweight='bold', fontsize=12)
ax4.grid(True, alpha=0.3)

# 9.5 Residual Plot
ax5 = plt.subplot(2, 3, 5)
residuals = y_test - y_pred_best
ax5.scatter(y_pred_best, residuals, alpha=0.5, s=20)
ax5.axhline(y=0, color='r', linestyle='--', lw=2)
ax5.set_xlabel('Predicted Revenue ($)')
ax5.set_ylabel('Residuals ($)')
ax5.set_title(f'Residual Plot - {best_model_name}', fontweight='bold', fontsize=12)
ax5.grid(True, alpha=0.3)

# 9.6 Feature Importance (if applicable)
ax6 = plt.subplot(2, 3, 6)
if best_model_name in ['Random Forest', 'XGBoost', 'LightGBM']:
    top_features = feature_imp.head(6)
    bars = ax6.barh(top_features['Feature'], top_features['Importance'], color='mediumpurple')
    ax6.set_xlabel('Importance')
    ax6.set_title(f'Top Features - {best_model_name}', fontweight='bold', fontsize=12)
    for i, v in enumerate(top_features['Importance']):
        ax6.text(v, i, f' {v:.4f}', va='center', fontsize=9)
    ax6.grid(True, alpha=0.3)
else:
    # Show coefficients for Linear Regression
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': abs(best_model.coef_)
    }).sort_values('Coefficient', ascending=False).head(6)

    bars = ax6.barh(coef_df['Feature'], coef_df['Coefficient'], color='mediumpurple')
    ax6.set_xlabel('Absolute Coefficient')
    ax6.set_title(f'Top Features - {best_model_name}', fontweight='bold', fontsize=12)
    ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/model_evaluation.png', dpi=150, bbox_inches='tight')
print("✓ Saved: results/model_evaluation.png")

# ============================================================================
# 10. SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)

best_metrics = results_df.iloc[0]
print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   R²:   {best_metrics['R2']:.4f}")
print(f"   MAPE: {best_metrics['MAPE']:.2f}%")
print(f"   RMSE: ${best_metrics['RMSE']:.2f}")
print(f"   MAE:  ${best_metrics['MAE']:.2f}")

print("\n📁 Files created:")
print("   - models/best_model.pkl")
print("   - models/model_info.pkl")
print("   - models/scaler.pkl")
print("   - results/model_comparison.csv")
print("   - results/feature_importance.csv")
print("   - results/model_evaluation.png")

print("\n" + "=" * 80)
