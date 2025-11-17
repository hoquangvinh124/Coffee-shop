"""
STEP 2: Model Training Pipeline
Script này huấn luyện và đánh giá 3 mô hình ML
Models: Random Forest, Gradient Boosting, XGBoost
Metrics: ROC-AUC, F1-Score, Accuracy
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, 
    classification_report, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
import os
warnings.filterwarnings('ignore')

# Get the project root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

print("=" * 80)
print("STEP 2: MODEL TRAINING & EVALUATION")
print("=" * 80)

# ============================================================================
# 1. LOAD PREPROCESSED DATA
# ============================================================================
print("\n Loading preprocessed data...")
X_train = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'X_train_processed.csv'))
X_test = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'X_test_processed.csv'))
y_train = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'y_train.csv')).values.ravel()
y_test = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'y_test.csv')).values.ravel()

print(f"✅ Train set: {X_train.shape}")
print(f"✅ Test set: {X_test.shape}")
print(f"   Features: {X_train.shape[1]}")

# ============================================================================
# 2. DEFINE MODELS AND HYPERPARAMETERS
# ============================================================================
print("\n🤖 Defining models and hyperparameters...")

models = {
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42, n_jobs=-1),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5],
            'subsample': [0.8, 1.0]
        }
    },
    'XGBoost': {
        'model': XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1),
        'params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
    }
}

# ============================================================================
# 3. TRAIN AND EVALUATE EACH MODEL
# ============================================================================
results = []
trained_models = {}

for name, config in models.items():
    print("\n" + "=" * 80)
    print(f"🔄 Training {name}...")
    print("=" * 80)
    
    # GridSearchCV with cross-validation
    print("Running GridSearchCV (5-fold CV)...")
    grid = GridSearchCV(
        config['model'],
        config['params'],
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid.fit(X_train, y_train)
    
    # Best model
    best_model = grid.best_estimator_
    trained_models[name] = best_model
    
    print(f"\n✅ Best parameters: {grid.best_params_}")
    print(f"   Best CV ROC-AUC: {grid.best_score_:.4f}")
    
    # Predictions on test set
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    
    # Store results
    results.append({
        'Model': name,
        'ROC-AUC': roc_auc,
        'F1-Score': f1,
        'Accuracy': acc,
        'Best_Params': str(grid.best_params_)
    })
    
    # Print metrics
    print(f"\n📊 Test Set Performance:")
    print(f"   ROC-AUC:  {roc_auc:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    print(f"   Accuracy: {acc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📈 Confusion Matrix:")
    print(f"   TN: {cm[0,0]:5d}  |  FP: {cm[0,1]:5d}")
    print(f"   FN: {cm[1,0]:5d}  |  TP: {cm[1,1]:5d}")
    
    # Classification Report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Conversion', 'Conversion']))
    
    # Save model
    model_filename = os.path.join(PROJECT_ROOT, 'models', f'{name.lower().replace(" ", "_")}.pkl')
    with open(model_filename, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"💾 Model saved: {model_filename}")

# ============================================================================
# 4. MODEL COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("📊 MODEL COMPARISON")
print("=" * 80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('ROC-AUC', ascending=False)
print(results_df[['Model', 'ROC-AUC', 'F1-Score', 'Accuracy']].to_string(index=False))

# Save comparison results
results_df.to_csv(os.path.join(PROJECT_ROOT, 'results', 'metrics', 'model_comparison.csv'), index=False)
print("\n💾 Comparison saved: results/metrics/model_comparison.csv")# ============================================================================
# 5. IDENTIFY BEST MODEL
# ============================================================================
best_idx = results_df['ROC-AUC'].idxmax()
best_model_name = results_df.loc[best_idx, 'Model']
best_roc_auc = results_df.loc[best_idx, 'ROC-AUC']

print("\n" + "=" * 80)
print(f"🏆 BEST MODEL: {best_model_name}")
print(f"   ROC-AUC: {best_roc_auc:.4f}")
print("=" * 80)

# Save best model separately
best_model = trained_models[best_model_name]
with open(os.path.join(PROJECT_ROOT, 'models', 'best_model.pkl'), 'wb') as f:
    pickle.dump(best_model, f)
print("💾 Best model saved as: models/best_model.pkl")

# ============================================================================
# 6. GENERATE ROC CURVES
# ============================================================================
print("\n📈 Generating ROC curves...")

plt.figure(figsize=(10, 8))

for name, model in trained_models.items():
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_ROOT, 'results', 'figures', 'roc_curves.png'), dpi=300)
print("✅ ROC curves saved: results/figures/roc_curves.png")

# ============================================================================
# 7. FEATURE IMPORTANCE (BEST MODEL)
# ============================================================================
if hasattr(best_model, 'feature_importances_'):
    print("\n📊 Extracting feature importance...")
    
    # Load feature names
    with open(os.path.join(PROJECT_ROOT, 'models', 'feature_names.pkl'), 'rb') as f:
        feature_names = pickle.load(f)
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Save feature importance
    importance_df.to_csv(os.path.join(PROJECT_ROOT, 'results', 'metrics', 'feature_importance.csv'), index=False)
    print("✅ Feature importance saved: results/metrics/feature_importance.csv")
    
    # Plot top 15 features
    plt.figure(figsize=(10, 8))
    top_features = importance_df.head(15)
    sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top 15 Feature Importance - {best_model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, 'results', 'figures', 'feature_importance_top15.png'), dpi=300)
    print("✅ Feature importance plot saved: results/figures/feature_importance_top15.png")
    
    print("\n🔝 Top 10 Most Important Features:")
    for i, row in importance_df.head(10).iterrows():
        print(f"   {row['feature']:40s} : {row['importance']:.4f}")

# ============================================================================
# 8. SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ STEP 2 COMPLETED SUCCESSFULLY!")
print("=" * 80)
print(f"Models trained: {len(models)}")
print(f"Best model: {best_model_name} (ROC-AUC: {best_roc_auc:.4f})")
print(f"Files saved:")
print(f"  - 3 trained models in models/")
print(f"  - best_model.pkl")
print(f"  - model_comparison.csv")
print(f"  - roc_curves.png")
print(f"  - feature_importance.csv & plot")
print("=" * 80)
