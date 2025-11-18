"""
Model Training - Promotional Response Prediction
================================================
Train multiple ML models và so sánh hiệu suất
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, 
    f1_score, classification_report, confusion_matrix, roc_curve
)
import lightgbm as lgb
import catboost as cb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

print("="*80)
print("MODEL TRAINING - PROMOTIONAL RESPONSE PREDICTION")
print("="*80)

# Load preprocessed data
print("\n1. LOADING PREPROCESSED DATA...")
X_train = pd.read_csv(DATA_DIR / "X_train_balanced.csv")
y_train = pd.read_csv(DATA_DIR / "y_train_balanced.csv").values.ravel()
X_test = pd.read_csv(DATA_DIR / "X_test.csv")
y_test = pd.read_csv(DATA_DIR / "y_test.csv").values.ravel()

print(f"✓ Train set: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
print(f"✓ Test set:  {X_test.shape[0]:,} samples, {X_test.shape[1]} features")
print(f"✓ Class distribution (train): {np.sum(y_train==0):,} (0) | {np.sum(y_train==1):,} (1)")
print(f"✓ Class distribution (test):  {np.sum(y_test==0):,} (0) | {np.sum(y_test==1):,} (1)")

# Model configuration
print("\n2. MODEL CONFIGURATIONS...")
print("-" * 80)

models = {
    'Logistic Regression': LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced'
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42
    ),
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=7,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
        class_weight='balanced'
    ),
    'CatBoost': cb.CatBoostClassifier(
        iterations=200,
        learning_rate=0.1,
        depth=6,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=False,
        auto_class_weights='Balanced'
    )
}

for name, model in models.items():
    print(f"✓ {name}: Configured")

# Train and evaluate models
print("\n3. TRAINING MODELS...")
print("="*80)

results = {}
trained_models = {}

for name, model in models.items():
    print(f"\n{'='*80}")
    print(f"Training: {name}")
    print(f"{'='*80}")
    
    # Train
    print("Training...")
    model.fit(X_train, y_train)
    trained_models[name] = model
    print("✓ Training complete")
    
    # Predict
    print("Making predictions...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluate
    print("Evaluating...")
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results[name] = {
        'ROC-AUC': roc_auc,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    # Print results
    print(f"\n{name} Results:")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"    TN: {cm[0,0]:,}  |  FP: {cm[0,1]:,}")
    print(f"    FN: {cm[1,0]:,}  |  TP: {cm[1,1]:,}")
    
    # Save model
    model_path = MODELS_DIR / f"{name.lower().replace(' ', '_')}.pkl"
    joblib.dump(model, model_path)
    print(f"\n✓ Model saved: {model_path}")

# Results comparison
print("\n4. MODEL COMPARISON")
print("="*80)

results_df = pd.DataFrame(results).T
results_df = results_df.drop(['y_pred', 'y_pred_proba'], axis=1)
results_df = results_df.sort_values('ROC-AUC', ascending=False)

print("\nModel Performance Ranking (by ROC-AUC):")
print("-" * 80)
print(results_df.to_string())

# Best model
best_model_name = results_df.index[0]
best_model = trained_models[best_model_name]
best_roc_auc = results_df.loc[best_model_name, 'ROC-AUC']

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   ROC-AUC: {best_roc_auc:.4f}")

# Save best model
best_model_path = MODELS_DIR / "best_model.pkl"
joblib.dump(best_model, best_model_path)
print(f"\n✓ Best model saved: {best_model_path}")

# Save results
results_path = RESULTS_DIR / "model_comparison.csv"
results_df.to_csv(results_path)
print(f"✓ Results saved: {results_path}")

# Feature importance (for tree-based models)
print("\n5. FEATURE IMPORTANCE ANALYSIS")
print("="*80)

feature_names = X_train.columns.tolist()

if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print(f"\nTop 15 Most Important Features ({best_model_name}):")
    print("-" * 80)
    for idx, row in feature_importance_df.head(15).iterrows():
        print(f"  {row['Feature']:30s}: {row['Importance']:.6f}")
    
    # Save feature importance
    importance_path = RESULTS_DIR / "feature_importance.csv"
    feature_importance_df.to_csv(importance_path, index=False)
    print(f"\n✓ Feature importance saved: {importance_path}")

# Visualizations
print("\n6. CREATING VISUALIZATIONS...")
print("-" * 80)

# Figure 1: Model comparison
fig, ax = plt.subplots(figsize=(12, 6))
metrics = ['ROC-AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(results_df.index))
width = 0.15

for i, metric in enumerate(metrics):
    values = results_df[metric].values
    ax.bar(x + i*width, values, width, label=metric, alpha=0.8)

ax.set_xlabel('Models', fontweight='bold', fontsize=12)
ax.set_ylabel('Score', fontweight='bold', fontsize=12)
ax.set_title('Model Performance Comparison', fontweight='bold', fontsize=14)
ax.set_xticks(x + width * 2)
ax.set_xticklabels(results_df.index, rotation=45, ha='right')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "model_comparison.png", dpi=300, bbox_inches='tight')
print("✓ Saved: model_comparison.png")
plt.close()

# Figure 2: ROC Curves
fig, ax = plt.subplots(figsize=(10, 8))

for name in results_df.index:
    y_pred_proba = results[name]['y_pred_proba']
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = results[name]['ROC-AUC']
    ax.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {roc_auc:.4f})')

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier (AUC = 0.5000)')
ax.set_xlabel('False Positive Rate', fontweight='bold', fontsize=12)
ax.set_ylabel('True Positive Rate', fontweight='bold', fontsize=12)
ax.set_title('ROC Curves Comparison', fontweight='bold', fontsize=14)
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "roc_curves.png", dpi=300, bbox_inches='tight')
print("✓ Saved: roc_curves.png")
plt.close()

# Figure 3: Confusion matrices
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, name in enumerate(results_df.index):
    y_pred = results[name]['y_pred']
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['No Conv', 'Conv'],
                yticklabels=['No Conv', 'Conv'])
    axes[idx].set_title(f'{name}\nROC-AUC: {results[name]["ROC-AUC"]:.4f}', 
                       fontweight='bold')
    axes[idx].set_ylabel('True Label')
    axes[idx].set_xlabel('Predicted Label')

# Hide last subplot if odd number of models
if len(results_df.index) < 6:
    axes[-1].axis('off')

plt.tight_layout()
plt.savefig(RESULTS_DIR / "confusion_matrices.png", dpi=300, bbox_inches='tight')
print("✓ Saved: confusion_matrices.png")
plt.close()

# Figure 4: Feature importance (if available)
if hasattr(best_model, 'feature_importances_'):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    top_features = feature_importance_df.head(20)
    y_pos = np.arange(len(top_features))
    
    ax.barh(y_pos, top_features['Importance'].values, color=plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features))))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features['Feature'].values)
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score', fontweight='bold', fontsize=12)
    ax.set_title(f'Top 20 Feature Importances - {best_model_name}', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_importance.png", dpi=300, bbox_inches='tight')
    print("✓ Saved: feature_importance.png")
    plt.close()

# Training summary
print("\n7. TRAINING SUMMARY")
print("="*80)

summary = f"""
MODEL TRAINING SUMMARY - PROMOTIONAL RESPONSE PREDICTION
========================================================

DATASET:
  Training samples: {len(X_train):,} (SMOTE-balanced)
  Test samples: {len(X_test):,}
  Features: {X_train.shape[1]}
  
MODELS TRAINED: {len(models)}
{chr(10).join([f"  ✓ {name}" for name in models.keys()])}

PERFORMANCE RESULTS:
{'='*80}
{results_df.to_string()}

BEST MODEL:
  🏆 Model: {best_model_name}
  📊 ROC-AUC: {best_roc_auc:.4f}
  🎯 Accuracy: {results_df.loc[best_model_name, 'Accuracy']:.4f}
  🎯 Precision: {results_df.loc[best_model_name, 'Precision']:.4f}
  🎯 Recall: {results_df.loc[best_model_name, 'Recall']:.4f}
  🎯 F1-Score: {results_df.loc[best_model_name, 'F1-Score']:.4f}

IMPROVEMENT vs BASELINE:
  Baseline XGBoost: 0.6344
  Current Best: {best_roc_auc:.4f}
  Improvement: {(best_roc_auc - 0.6344)*100:+.2f} percentage points
  {"✅ BETTER" if best_roc_auc > 0.6344 else "⚠️ NEEDS IMPROVEMENT"}

KEY INSIGHTS:
"""

# Add feature importance insights if available
if hasattr(best_model, 'feature_importances_'):
    top_5_features = feature_importance_df.head(5)
    summary += "\n  Top 5 Most Important Features:\n"
    for idx, row in top_5_features.iterrows():
        summary += f"    {idx+1}. {row['Feature']:30s}: {row['Importance']:.6f}\n"

summary += f"""
SAVED ARTIFACTS:
  ✓ All trained models saved to: {MODELS_DIR}
  ✓ Best model: {best_model_path.name}
  ✓ Performance comparison: model_comparison.csv
  ✓ Feature importance: feature_importance.csv
  ✓ Visualizations: 4 plots saved to {RESULTS_DIR}

NEXT STEPS:
  1. Run 05_model_optimization.py for hyperparameter tuning
  2. Run 06_business_strategy.py for targeting recommendations
  3. Deploy best model to production environment
  4. Set up A/B testing framework
"""

summary_path = RESULTS_DIR / "training_summary.txt"
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write(summary)

print(summary)
print(f"\n✓ Summary saved to: {summary_path}")

print("\n" + "="*80)
print("✅ MODEL TRAINING COMPLETE!")
print("="*80)
print(f"\n🏆 Best Model: {best_model_name} (ROC-AUC: {best_roc_auc:.4f})")
print(f"📁 Models saved to: {MODELS_DIR}")
print(f"📊 Results saved to: {RESULTS_DIR}")
print(f"\nNext step: Run 05_model_optimization.py for hyperparameter tuning")
