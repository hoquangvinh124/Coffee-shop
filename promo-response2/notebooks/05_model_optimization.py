"""
Model Optimization - Hyperparameter Tuning
==========================================
Tối ưu hóa hyperparameters cho best performing models
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score, make_scorer
import lightgbm as lgb
import catboost as cb
from scipy.stats import randint, uniform
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

print("="*80)
print("MODEL OPTIMIZATION - HYPERPARAMETER TUNING")
print("="*80)

# Load data
print("\n1. LOADING DATA...")
X_train = pd.read_csv(DATA_DIR / "X_train_balanced.csv")
y_train = pd.read_csv(DATA_DIR / "y_train_balanced.csv").values.ravel()
X_test = pd.read_csv(DATA_DIR / "X_test.csv")
y_test = pd.read_csv(DATA_DIR / "y_test.csv").values.ravel()

print(f"✓ Train: {X_train.shape[0]:,} samples")
print(f"✓ Test: {X_test.shape[0]:,} samples")

# Define parameter grids for top models
print("\n2. HYPERPARAMETER SEARCH SPACES...")
print("-" * 80)

# LightGBM parameters
lgb_params = {
    'n_estimators': [100, 200, 300, 400],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10, -1],
    'num_leaves': [15, 31, 63, 127],
    'min_child_samples': [10, 20, 30, 50],
    'subsample': [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'reg_alpha': [0, 0.1, 0.5, 1.0],
    'reg_lambda': [0, 0.1, 0.5, 1.0]
}

print("✓ LightGBM: 9 hyperparameters configured")

# CatBoost parameters
catboost_params = {
    'iterations': [100, 200, 300, 400],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'depth': [4, 6, 8, 10],
    'l2_leaf_reg': [1, 3, 5, 7, 9],
    'border_count': [32, 64, 128],
    'bagging_temperature': [0, 0.5, 1.0],
    'random_strength': [0, 0.5, 1.0]
}

print("✓ CatBoost: 7 hyperparameters configured")

# Optimization function
def optimize_model(model_name, model, param_grid, X_train, y_train, X_test, y_test):
    """
    Optimize model using RandomizedSearchCV
    """
    print(f"\n{'='*80}")
    print(f"OPTIMIZING: {model_name}")
    print(f"{'='*80}")
    
    # Setup RandomizedSearch
    scorer = make_scorer(roc_auc_score, needs_proba=True)
    
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=30,  # Try 30 random combinations
        scoring=scorer,
        cv=5,  # 5-fold cross-validation
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    # Fit
    print("\nSearching for best hyperparameters...")
    random_search.fit(X_train, y_train)
    
    # Best parameters
    print(f"\n✓ Best parameters found:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")
    
    # Cross-validation score
    print(f"\n✓ Best CV ROC-AUC: {random_search.best_score_:.4f}")
    
    # Test set evaluation
    best_model = random_search.best_estimator_
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    test_roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"✓ Test ROC-AUC: {test_roc_auc:.4f}")
    
    return {
        'best_model': best_model,
        'best_params': random_search.best_params_,
        'cv_score': random_search.best_score_,
        'test_score': test_roc_auc,
        'cv_results': random_search.cv_results_
    }

# Optimize LightGBM
print("\n3. OPTIMIZING LIGHTGBM...")
print("="*80)

lgb_model = lgb.LGBMClassifier(
    random_state=42,
    verbosity=-1,
    class_weight='balanced'
)

lgb_results = optimize_model('LightGBM', lgb_model, lgb_params, X_train, y_train, X_test, y_test)

# Save optimized LightGBM
lgb_optimized_path = MODELS_DIR / "lightgbm_optimized.pkl"
joblib.dump(lgb_results['best_model'], lgb_optimized_path)
print(f"\n✓ Optimized LightGBM saved: {lgb_optimized_path}")

# Optimize CatBoost
print("\n4. OPTIMIZING CATBOOST...")
print("="*80)

catboost_model = cb.CatBoostClassifier(
    random_seed=42,
    verbose=False,
    auto_class_weights='Balanced'
)

catboost_results = optimize_model('CatBoost', catboost_model, catboost_params, X_train, y_train, X_test, y_test)

# Save optimized CatBoost
catboost_optimized_path = MODELS_DIR / "catboost_optimized.pkl"
joblib.dump(catboost_results['best_model'], catboost_optimized_path)
print(f"\n✓ Optimized CatBoost saved: {catboost_optimized_path}")

# Compare with baseline
print("\n5. OPTIMIZATION RESULTS COMPARISON")
print("="*80)

# Load baseline results
baseline_results = pd.read_csv(RESULTS_DIR / "model_comparison.csv", index_col=0)

comparison = pd.DataFrame({
    'Model': ['LightGBM (Baseline)', 'LightGBM (Optimized)', 
              'CatBoost (Baseline)', 'CatBoost (Optimized)'],
    'Test ROC-AUC': [
        baseline_results.loc['LightGBM', 'ROC-AUC'],
        lgb_results['test_score'],
        baseline_results.loc['CatBoost', 'ROC-AUC'],
        catboost_results['test_score']
    ],
    'CV ROC-AUC': [
        'N/A',
        f"{lgb_results['cv_score']:.4f}",
        'N/A',
        f"{catboost_results['cv_score']:.4f}"
    ]
})

comparison['Improvement'] = ''
comparison.loc[1, 'Improvement'] = f"+{(lgb_results['test_score'] - baseline_results.loc['LightGBM', 'ROC-AUC'])*100:.2f}%"
comparison.loc[3, 'Improvement'] = f"+{(catboost_results['test_score'] - baseline_results.loc['CatBoost', 'ROC-AUC'])*100:.2f}%"

print("\nOptimization Results:")
print("-" * 80)
print(comparison.to_string(index=False))

# Determine final best model
all_scores = {
    'LightGBM (Baseline)': baseline_results.loc['LightGBM', 'ROC-AUC'],
    'LightGBM (Optimized)': lgb_results['test_score'],
    'CatBoost (Baseline)': baseline_results.loc['CatBoost', 'ROC-AUC'],
    'CatBoost (Optimized)': catboost_results['test_score'],
    'Gradient Boosting': baseline_results.loc['Gradient Boosting', 'ROC-AUC'],
    'Random Forest': baseline_results.loc['Random Forest', 'ROC-AUC']
}

best_model_name = max(all_scores, key=all_scores.get)
best_score = all_scores[best_model_name]

print(f"\n🏆 FINAL BEST MODEL: {best_model_name}")
print(f"   Test ROC-AUC: {best_score:.4f}")

# Save final best model
if 'LightGBM (Optimized)' == best_model_name:
    final_best_model = lgb_results['best_model']
    final_params = lgb_results['best_params']
elif 'CatBoost (Optimized)' == best_model_name:
    final_best_model = catboost_results['best_model']
    final_params = catboost_results['best_params']
else:
    # Load baseline best model
    final_best_model = joblib.load(MODELS_DIR / f"{best_model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.pkl")
    final_params = "Baseline parameters"

final_model_path = MODELS_DIR / "final_best_model.pkl"
joblib.dump(final_best_model, final_model_path)
print(f"\n✓ Final best model saved: {final_model_path}")

# Save optimization summary
print("\n6. SAVING OPTIMIZATION SUMMARY...")
print("-" * 80)

summary = f"""
MODEL OPTIMIZATION SUMMARY
==========================

OPTIMIZATION METHOD: RandomizedSearchCV
  - Iterations per model: 30
  - Cross-validation folds: 5
  - Scoring metric: ROC-AUC
  - Search strategy: Random sampling

LIGHTGBM OPTIMIZATION:
{'='*80}
Best Parameters:
{chr(10).join([f"  {k}: {v}" for k, v in lgb_results['best_params'].items()])}

Performance:
  - CV ROC-AUC: {lgb_results['cv_score']:.4f}
  - Test ROC-AUC: {lgb_results['test_score']:.4f}
  - Baseline: {baseline_results.loc['LightGBM', 'ROC-AUC']:.4f}
  - Improvement: {(lgb_results['test_score'] - baseline_results.loc['LightGBM', 'ROC-AUC'])*100:+.2f}%

CATBOOST OPTIMIZATION:
{'='*80}
Best Parameters:
{chr(10).join([f"  {k}: {v}" for k, v in catboost_results['best_params'].items()])}

Performance:
  - CV ROC-AUC: {catboost_results['cv_score']:.4f}
  - Test ROC-AUC: {catboost_results['test_score']:.4f}
  - Baseline: {baseline_results.loc['CatBoost', 'ROC-AUC']:.4f}
  - Improvement: {(catboost_results['test_score'] - baseline_results.loc['CatBoost', 'ROC-AUC'])*100:+.2f}%

FINAL BEST MODEL:
{'='*80}
  Model: {best_model_name}
  Test ROC-AUC: {best_score:.4f}
  
  Improvement vs Original Baseline (XGBoost 0.6344):
    {(best_score - 0.6344)*100:+.2f} percentage points
    {"✅ SIGNIFICANT IMPROVEMENT" if best_score > 0.70 else "⚠️ MODERATE IMPROVEMENT"}

SAVED MODELS:
  ✓ {lgb_optimized_path.name}
  ✓ {catboost_optimized_path.name}
  ✓ {final_model_path.name} (production-ready)

PERFORMANCE COMPARISON:
{'='*80}
{comparison.to_string(index=False)}

NEXT STEPS:
  1. Run 06_business_strategy.py for targeting recommendations
  2. Deploy final_best_model.pkl to production
  3. Set up monitoring and A/B testing
  4. Collect feedback for model retraining
"""

summary_path = RESULTS_DIR / "optimization_summary.txt"
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write(summary)

print(summary)
print(f"\n✓ Summary saved: {summary_path}")

# Save comparison table
comparison_path = RESULTS_DIR / "optimization_comparison.csv"
comparison.to_csv(comparison_path, index=False)
print(f"✓ Comparison saved: {comparison_path}")

print("\n" + "="*80)
print("✅ MODEL OPTIMIZATION COMPLETE!")
print("="*80)
print(f"\n🏆 Final Best Model: {best_model_name} (ROC-AUC: {best_score:.4f})")
print(f"📁 Production model: {final_model_path}")
print(f"\nNext step: Run 06_business_strategy.py for actionable insights")
