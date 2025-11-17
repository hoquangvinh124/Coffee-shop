# 🚀 Hướng Dẫn Nhanh - Quick Start Guide

## Mục Tiêu Dự Án

Dự đoán **phản ứng của khách hàng Starbucks** với các chương trình khuyến mãi:

- **5 classes**: Offer Received, Offer Viewed, Transaction, Offer Completed, Green Flag
- **Problem**: Multiclass Classification với Imbalanced Dataset
- **Goal**: Tối ưu marketing strategy và giảm chi phí

---

## 📦 Cài Đặt Nhanh

```bash
# 1. Clone/Download dự án
cd targeted_mkt_statery

# 2. Cài đặt dependencies
pip install -r requirements.txt

# 3. Kiểm tra cấu trúc data
# Đảm bảo có 3 files trong data/:
# - portfolio.json
# - profile.json
# - transcript.json
```

---

## 🎯 Workflow - 4 Bước Chính

### **Bước 1: Exploratory Data Analysis (EDA)**

```bash
jupyter notebook notebooks/01_data_loading_and_eda.ipynb
```

**Làm gì:**

- ✅ Load 3 files JSON
- ✅ Phân tích cấu trúc dữ liệu
- ✅ Thống kê mô tả (age, income, events)
- ✅ Visualizations (distributions, correlations)

**Output:**

- Hiểu rõ data structure
- Phát hiện imbalanced dataset
- Biểu đồ lưu trong `results/figures/`

---

### **Bước 2: Data Preprocessing**

```bash
jupyter notebook notebooks/02_data_preprocessing.ipynb
```

**Làm gì:**

- ✅ Xử lý missing values
- ✅ Merge 3 dataframes (transcript + profile + portfolio)
- ✅ Feature engineering (reg_month, offer_id encoding)
- ✅ Feature encoding (gender, events)
- ✅ Feature scaling (StandardScaler, MinMaxScaler)
- ✅ Train/Test split (75/25)

**Output:**

- `data/processed/X_train.csv`
- `data/processed/X_test.csv`
- `data/processed/y_train.csv`
- `data/processed/y_test.csv`
- `data/processed/metadata.pkl`

---

### **Bước 3: Model Training**

```bash
jupyter notebook notebooks/03_model_training.ipynb
```

**Làm gì:**

- ✅ Load processed data
- ✅ Handle imbalanced data (SMOTE, class weights)
- ✅ Train 3 models:
  - **DNN** với Entity Embedding
  - **XGBoost** (best overall)
  - **Random Forest**
- ✅ Hyperparameter tuning
- ✅ Save trained models

**Models:**

#### 1. DNN Architecture

```python
# Multi-input với Entity Embedding
- Embedding: offer_id (200 dims), gender (200 dims)
- Dense(32) + ReLU
- Dense(32) + ReLU
- Dropout(0.2)
- Dense(32) + ReLU
- Dense(5) + Softmax

# Training
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Epochs: 15
- Batch size: 64
```

#### 2. XGBoost (Recommended)

```python
params = {
    'max_depth': 10,
    'gamma': 5,
    'objective': 'multi:softmax',
    'num_class': 5
}
```

#### 3. Random Forest

```python
RandomForestClassifier(
    n_estimators=150,
    class_weight='balanced_subsample'
)
```

**Output:**

- `models/dnn_model.h5`
- `models/xgboost_model.pkl`
- `models/random_forest_model.pkl`

---

### **Bước 4: Model Evaluation**

```bash
jupyter notebook notebooks/04_model_evaluation.ipynb
```

**Làm gì:**

- ✅ Load trained models
- ✅ Predictions trên test set
- ✅ Performance metrics:
  - Confusion Matrix
  - F1-Score (Micro, Macro, Weighted)
  - Per-class Precision/Recall
- ✅ Feature Importance
- ✅ SHAP Analysis
- ✅ Model Comparison

**Expected Results:**

| Model                  | F1 (Imbalanced) | F1 (Balanced) |
| ---------------------- | --------------- | ------------- |
| DNN (Entity Embedding) | **63.12%**      | ~50%          |
| XGBoost                | **63.45%**      | ~50%          |
| Random Forest          | 63%             | ~50%          |

**Top Features (SHAP):**

1. offer_id
2. income
3. age
4. difficulty
5. reward

---

## 🐍 Sử Dụng Python Modules

### Option 1: Jupyter Notebooks (Recommended)

Thực hiện từng bước trong notebooks như trên.

### Option 2: Python Scripts

```python
# Load và preprocess data
from src import DataLoader, Preprocessor

# Load data
loader = DataLoader(data_path='data/')
merged_data = loader.run_pipeline()

# Preprocess
preprocessor = Preprocessor(merged_data)
X_train, X_test, y_train, y_test = preprocessor.run_pipeline()

# Lưu processed data
preprocessor.save_processed_data('data/processed/')
```

---

## ⚖️ Handling Imbalanced Data - Chiến Lược

### Problem

```
Class 3 (offer completed): 40% ← Majority
Class 1 (offer viewed):    10% ← Minority
Class 4 (green flag):       5% ← Minority
```

### Solution 1: Random Over-Sampling

```python
from imblearn.over_sampling import RandomOverSampler

sampler = RandomOverSampler(sampling_strategy='not majority')
X_train, y_train = sampler.fit_resample(X_train, y_train)
```

✅ **Result:** Tất cả classes có ~50% recall (fair prediction)

### Solution 2: Class Weights (DNN)

```python
class_weights = {
    0: 3.2,
    1: 39.0,  # High weight cho minority
    2: 1.4,
    3: 1.0,
    4: 6.0
}

model.fit(..., class_weight=class_weights)
```

---

## 📊 Quick Metrics Check

```python
from src.utils import evaluate_model, print_model_evaluation

# Đánh giá model
results = evaluate_model(y_test, y_pred, class_names=class_names)

# In kết quả
print_model_evaluation(results, model_name="XGBoost")
```

---

## 🎓 Key Takeaways

### 1. Best Model

- **XGBoost** performs best (63.45% F1)
- Tree-based models > DNN cho tabular data này
- Entity Embedding quan trọng cho DNN

### 2. Imbalanced Data

- **KHÔNG ignore** minority classes!
- SMOTE: Accuracy ↓ nhưng Fairness ↑
- Trade-off phụ thuộc business goal

### 3. Feature Importance

- `offer_id` là feature quan trọng nhất
- `income` và `age` ảnh hưởng lớn
- `reg_month` ảnh hưởng thấp nhất

### 4. Business Value

```
Balanced Model Benefits:
✓ Fair prediction cho tất cả customer segments
✓ Better targeting → ROI cao hơn
✓ Nhận diện "green flag customers" → Giảm marketing cost
```

---

## 🔥 Common Issues & Solutions

### Issue 1: Missing Data Files

```
Error: File not found
```

**Solution:** Đảm bảo có 3 files JSON trong `data/`:

- portfolio.json
- profile.json
- transcript.json

### Issue 2: Memory Error

```
MemoryError: Unable to allocate array
```

**Solution:**

- Giảm `batch_size` trong DNN
- Sample data nhỏ hơn để test
- Sử dụng `n_jobs=1` trong Random Forest

### Issue 3: Import Error

```
ModuleNotFoundError: No module named 'src'
```

**Solution:**

```bash
# Thêm project root vào PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/Mac
set PYTHONPATH=%PYTHONPATH%;%CD%  # Windows
```

---

## 📞 Hỗ Trợ

Nếu gặp vấn đề:

1. Kiểm tra `requirements.txt` đã cài đầy đủ
2. Xem lại logs trong notebooks
3. Đọc docstrings trong các modules (`src/`)

---

**Happy Modeling! 🚀**

---

## 📚 Tài Liệu Tham Khảo

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [TensorFlow Keras Guide](https://www.tensorflow.org/guide/keras)
- [Imbalanced-learn](https://imbalanced-learn.org/stable/)
- [SHAP Documentation](https://shap.readthedocs.io/)
