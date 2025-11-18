# Comprehensive Model Testing Guide

This guide explains how to test the effectiveness of models in the targeted marketing strategy project.

## 📋 Overview

The testing suite includes:

1. **Comprehensive Model Testing Notebook** (`notebooks/05_comprehensive_model_testing.ipynb`)
2. **Temporal Validation Notebook** (`notebooks/06_temporal_validation.ipynb`)
3. **Business Evaluator Module** (`src/business_evaluator.py`)
4. **Production Benchmarking Script** (`scripts/benchmark_models.py`)

## 🚀 Quick Start

### Option 1: Run All Tests (Recommended)

```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Navigate to project directory
cd targeted_mkt_statery

# Run comprehensive testing notebook
jupyter notebook notebooks/05_comprehensive_model_testing.ipynb

# Run temporal validation notebook
jupyter notebook notebooks/06_temporal_validation.ipynb

# Run production benchmarks
python scripts/benchmark_models.py
```

### Option 2: Run Individual Components

#### A. Comprehensive Model Testing

```powershell
cd targeted_mkt_statery
jupyter notebook notebooks/05_comprehensive_model_testing.ipynb
```

This notebook provides:

- ✅ Basic and per-class performance metrics
- ✅ ROC-AUC curves for all classes (multiclass one-vs-rest)
- ✅ Precision-Recall curves for minority classes
- ✅ Prediction confidence/probability analysis
- ✅ Business ROI metrics and campaign simulations
- ✅ Error analysis with top 100 misclassifications
- ✅ Comprehensive visualizations

**Expected Output Files:**

- `results/metrics/comprehensive_basic_metrics.csv`
- `results/metrics/comprehensive_per_class_metrics.csv`
- `results/metrics/roc_auc_scores.csv`
- `results/metrics/precision_recall_scores.csv`
- `results/metrics/business_roi_metrics.csv`
- `results/metrics/campaign_simulation_results.csv`
- `results/metrics/error_feature_analysis.csv`
- `results/metrics/top_100_errors.csv`
- `results/metrics/comprehensive_testing_summary.txt`
- `results/figures/*.png` (27+ visualization files)

#### B. Temporal Validation

```powershell
cd targeted_mkt_statery
jupyter notebook notebooks/06_temporal_validation.ipynb
```

This notebook provides:

- ✅ Time-based data splits (train on old, test on new customers)
- ✅ Model retraining on temporal splits
- ✅ Performance degradation analysis
- ✅ Comparison with original random-split models
- ✅ Realistic production performance estimates

**Expected Output Files:**

- `results/metrics/temporal_validation_results.csv`
- `results/metrics/temporal_degradation_analysis.csv`
- `results/metrics/temporal_vs_original_comparison.csv`
- `results/metrics/temporal_validation_summary.txt`
- `results/figures/temporal_*.png`

#### C. Production Benchmarking

```powershell
cd targeted_mkt_statery
python scripts/benchmark_models.py
```

This script provides:

- ✅ Model loading time measurements
- ✅ Inference speed (predictions/second) for batch sizes: 1, 10, 100, 1000, 5000
- ✅ Memory usage profiling
- ✅ Full dataset throughput testing
- ✅ Accuracy verification
- ✅ Production readiness recommendations

**Expected Output Files:**

- `results/metrics/production_benchmarks.csv`
- `results/metrics/production_benchmarks_report.txt`

## 📊 Understanding the Results

### 1. Model Performance Metrics

**Key Metrics:**

- **Accuracy**: Overall correctness (but misleading for imbalanced data)
- **F1 (Micro)**: Overall performance weighted by samples
- **F1 (Macro)**: Unweighted average across classes (important for minority classes)
- **F1 (Weighted)**: Weighted average by class support
- **ROC-AUC**: Area under ROC curve (per-class discrimination ability)
- **Average Precision**: Area under precision-recall curve (for minority classes)

**Interpretation:**

- F1 > 0.70: Excellent performance
- F1 0.50-0.70: Good performance
- F1 0.30-0.50: Moderate performance (needs improvement)
- F1 < 0.30: Poor performance

### 2. Business Metrics

**ROI Calculation:**

```
ROI (%) = (Net Profit / Total Cost) × 100
Net Profit = Total Revenue - Total Cost
```

**Assumptions (configurable in notebook):**

- Offer cost: $2.00 per offer sent
- Completion revenue: $10.00 per completed offer
- View value: $0.50 per offer view
- Transaction value: $5.00 per transaction

**Interpretation:**

- ROI > 100%: Strongly recommended for deployment (high profitability)
- ROI 50-100%: Recommended for deployment (good profitability)
- ROI 0-50%: Cautiously recommended (low but positive profit)
- ROI < 0%: Not recommended (losing money)

### 3. Production Performance

**Inference Speed Benchmarks:**

- **Excellent**: > 1000 predictions/second
- **Good**: 500-1000 predictions/second
- **Acceptable**: 100-500 predictions/second
- **Needs Optimization**: < 100 predictions/second

**Memory Usage:**

- Model loading memory: How much RAM to load model
- Prediction memory: Additional RAM during inference
- Total memory: Peak memory usage

### 4. Temporal Validation

**Performance Degradation:**

- < 5% degradation: Excellent generalization
- 5-10% degradation: Good generalization
- 10-20% degradation: Moderate generalization (monitor in production)
- > 20% degradation: Poor generalization (needs retraining strategy)

## 🎯 Model Comparison Summary

Based on previous analysis:

| Model                 | F1-Score | ROI    | Speed  | Minority Detection | Best For                       |
| --------------------- | -------- | ------ | ------ | ------------------ | ------------------------------ |
| **XGBoost Standard**  | 0.7021   | High   | Fast   | ❌ Poor            | High-volume, accuracy-critical |
| **XGBoost Resampled** | 0.6396   | Medium | Fast   | ✅ Good            | Balanced detection, fairness   |
| **Random Forest**     | 0.5950   | Medium | Medium | ⚠️ Moderate        | General purpose                |
| **DNN Entity Embed**  | 0.1883   | Low    | Slow   | ❌ Poor            | ❌ Not recommended             |

### Recommendations by Use Case

**1. High-Volume Campaigns (Maximize Accuracy)**

- **Model**: XGBoost Standard
- **Strategy**: Top probability targeting
- **Expected ROI**: Highest
- **Trade-off**: Ignores minority classes (Offer Viewed, Offer Completed)

**2. Balanced Detection (Fairness)**

- **Model**: XGBoost Resampled
- **Strategy**: Top probability with threshold tuning
- **Expected ROI**: Good
- **Trade-off**: Slightly lower overall accuracy

**3. Minority Class Focus (Offer Completions)**

- **Model**: XGBoost Resampled
- **Strategy**: Target class 3 specifically
- **Expected ROI**: Variable by campaign
- **Trade-off**: Better recall on completions

**4. Production Deployment**

- **Primary**: XGBoost Standard (for most predictions)
- **Secondary**: XGBoost Resampled (for minority class campaigns)
- **Ensemble**: Combine both models for hybrid approach

## 🔧 Customization

### Adjust Business Metrics

Edit `notebooks/05_comprehensive_model_testing.ipynb`:

```python
# Modify cost assumptions
evaluator = BusinessEvaluator(
    offer_cost=2.0,           # Change cost per offer
    completion_revenue=10.0,  # Change revenue per completion
    view_value=0.5,           # Change value of views
    transaction_value=5.0     # Change transaction value
)
```

### Change Campaign Budget

```python
# Modify campaign simulation budget
campaign_budget = 20000  # $20,000 instead of $10,000
```

### Adjust Temporal Splits

Edit `notebooks/06_temporal_validation.ipynb`:

```python
# Modify split percentages
train_cutoff = X_full['reg_month'].quantile(0.7)  # 70% train instead of 60%
val_cutoff = X_full['reg_month'].quantile(0.85)   # 85% val instead of 80%
```

### Add Custom Benchmarking

Edit `scripts/benchmark_models.py`:

```python
# Add custom batch sizes
batch_sizes = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000]

# Add custom iterations
n_iterations = 20  # More iterations for stable results
```

## 📈 Interpreting Visualizations

### ROC Curves

- **X-axis**: False Positive Rate (FPR)
- **Y-axis**: True Positive Rate (TPR/Recall)
- **Ideal**: Curve hugs top-left corner (high TPR, low FPR)
- **Baseline**: Diagonal line (random classifier)

### Precision-Recall Curves

- **X-axis**: Recall (sensitivity)
- **Y-axis**: Precision (positive predictive value)
- **Ideal**: Curve hugs top-right corner (high precision and recall)
- **Important for**: Imbalanced classes (minority classes)

### Confusion Matrix (Errors)

- **Rows**: True classes
- **Columns**: Predicted classes
- **Diagonal**: Correct predictions
- **Off-diagonal**: Misclassifications
- **Dark cells**: High error rates (needs attention)

### Confidence Distribution

- **Correct predictions**: Should have high confidence
- **Incorrect predictions**: Should have lower confidence
- **Separation**: Good models show clear separation between correct/incorrect confidence

## 🚨 Common Issues & Solutions

### Issue 1: TensorFlow Not Available

**Error**: `Warning: TensorFlow not available`
**Solution**:

```powershell
pip install tensorflow
```

Or skip DNN model testing (it performs poorly anyway)

### Issue 2: Out of Memory

**Error**: `MemoryError` during benchmarking
**Solution**: Reduce batch sizes in benchmark script:

```python
batch_sizes = [1, 10, 100, 1000]  # Remove 5000
```

### Issue 3: Notebooks Run Slow

**Solution**:

- Run on fewer models (comment out underperforming models)
- Reduce number of iterations in benchmarks
- Use smaller test dataset for initial testing

### Issue 4: Missing reg_month Column

**Error**: `'reg_month' column not found`
**Solution**: Temporal validation requires preprocessed data with `reg_month` feature. Check `data/processed/` directory or rerun preprocessing notebook.

## 📝 Next Steps After Testing

1. **Review Summary Reports**

   - `results/metrics/comprehensive_testing_summary.txt`
   - `results/metrics/temporal_validation_summary.txt`
   - `results/metrics/production_benchmarks_report.txt`

2. **Analyze Error Cases**

   - Open `results/metrics/top_100_errors.csv`
   - Identify patterns in misclassifications
   - Consider feature engineering improvements

3. **Business Decision**

   - Review ROI metrics in `results/metrics/business_roi_metrics.csv`
   - Compare campaign strategies in `results/metrics/campaign_simulation_results.csv`
   - Make deployment decision based on business objectives

4. **Deployment Preparation**

   - Check production benchmarks for infrastructure requirements
   - Set up monitoring based on temporal validation findings
   - Implement A/B testing framework for validation

5. **Continuous Improvement**
   - Schedule periodic retraining based on temporal degradation
   - Monitor performance in production vs validation
   - Collect feedback for model improvement

## 📚 Additional Resources

- **Project README**: `README.md` (project overview)
- **Quickstart Guide**: `QUICKSTART.md` (setup instructions)
- **Project Conclusion**: `PROJECT_CONCLUSION.md` (final analysis)
- **Config File**: `config/config.yaml` (hyperparameters and settings)

## 🤝 Support

For issues or questions:

1. Check error logs in notebook outputs
2. Review configuration in `config/config.yaml`
3. Verify data files exist in `data/processed/`
4. Check Python environment and installed packages

## ✅ Testing Checklist

Before deployment, ensure:

- [ ] Run comprehensive testing notebook successfully
- [ ] Review all performance metrics (accuracy, F1, ROC-AUC)
- [ ] Calculate business ROI and verify profitability
- [ ] Analyze error patterns and understand failures
- [ ] Run temporal validation to assess generalization
- [ ] Check performance degradation is acceptable (< 10%)
- [ ] Run production benchmarks to verify infrastructure readiness
- [ ] Review inference speed meets requirements (> 100 pred/sec)
- [ ] Document model limitations and failure cases
- [ ] Create deployment plan with monitoring strategy

---

**Happy Testing! 🎉**
