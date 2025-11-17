# Chiến Lược Marketing Có Mục Tiêu cho Starbucks

# Targeted Marketing Strategy for Starbucks

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-green.svg)](https://scikit-learn.org/)

## 📋 Mô Tả Dự Án

Dự án Machine Learning dự đoán **phản ứng của khách hàng** với các chương trình khuyến mãi (offers) được gửi qua ứng dụng Starbucks. Đây là bài toán **multiclass classification** với 5 loại phản ứng khác nhau.

### 🎯 Mục Tiêu

Xây dựng model ML để:

- Dự đoán hành vi khách hàng khi nhận offers
- Tối ưu hóa chiến lược marketing
- Giảm chi phí bằng cách nhận diện "green flag customers" (không cần gửi offer)

### 📊 Target Classes

| Class | Event           | Mô Tả                          |
| ----- | --------------- | ------------------------------ |
| 0     | Offer Received  | Khách hàng nhận được offer     |
| 1     | Offer Viewed    | Khách hàng xem offer           |
| 2     | Transaction     | Giao dịch không dùng offer     |
| 3     | Offer Completed | Hoàn thành giao dịch với offer |
| 4     | Green Flag      | Giao dịch mà không cần offer   |

---

## 📁 Cấu Trúc Dự Án

```
targeted_mkt_statery/
│
├── data/                           # Dữ liệu thô và đã xử lý
│   ├── portfolio.json              # Thông tin offers
│   ├── profile.json                # Thông tin khách hàng
│   ├── transcript.json             # Lịch sử tương tác
│   └── processed/                  # Dữ liệu đã preprocessing
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       ├── y_test.csv
│       └── metadata.pkl
│
├── notebooks/                      # Jupyter Notebooks
│   ├── 01_data_loading_and_eda.ipynb       # EDA và phân tích
│   ├── 02_data_preprocessing.ipynb         # Tiền xử lý dữ liệu
│   ├── 03_model_training.ipynb             # Training models
│   └── 04_model_evaluation.ipynb           # Đánh giá và so sánh
│
├── src/                            # Source code modules
│   ├── __init__.py
│   ├── data_loader.py              # Load và merge data
│   ├── preprocessor.py             # Preprocessing pipeline
│   ├── models.py                   # Model definitions
│   └── utils.py                    # Utility functions
│
├── models/                         # Trained models
│   ├── dnn_model.h5
│   ├── xgboost_model.pkl
│   └── random_forest_model.pkl
│
├── results/                        # Kết quả và visualizations
│   ├── figures/                    # Biểu đồ, plots
│   └── metrics/                    # Performance metrics
│
├── config/                         # Configuration files
│   └── config.yaml
│
├── requirements.txt                # Python dependencies
└── README.md                       # File này
```

---

## 🚀 Hướng Dẫn Sử Dụng

### 1. Cài Đặt

```bash
# Clone repository
git clone <repo-url>
cd targeted_mkt_statery

# Tạo virtual environment (khuyến nghị)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

### 2. Chạy Notebooks

Thực hiện theo thứ tự:

#### **Notebook 01 - Data Loading & EDA**

```bash
jupyter notebook notebooks/01_data_loading_and_eda.ipynb
```

- Load và khám phá dữ liệu
- Phân tích thống kê mô tả
- Visualizations

#### **Notebook 02 - Data Preprocessing**

```bash
jupyter notebook notebooks/02_data_preprocessing.ipynb
```

- Xử lý missing values
- Merge 3 dataframes
- Feature engineering
- Feature encoding & scaling
- Train/Test split

#### **Notebook 03 - Model Training**

```bash
jupyter notebook notebooks/03_model_training.ipynb
```

- Handle imbalanced dataset
- Train DNN, XGBoost, Random Forest
- Hyperparameter tuning
- Save models

#### **Notebook 04 - Model Evaluation**

```bash
jupyter notebook notebooks/04_model_evaluation.ipynb
```

- Đánh giá performance
- Confusion matrix analysis
- Feature importance
- SHAP analysis
- Model comparison

---

## 📊 Dataset

### Portfolio (10 offers)

- **offer_id**: ID của offer
- **offer_type**: BOGO, Discount, Informational
- **reward**: Phần thưởng ($)
- **difficulty**: Số tiền cần chi để nhận reward
- **duration**: Thời hạn offer (days)

### Profile (~17K customers)

- **id**: Customer ID
- **gender**: F, M, O
- **age**: Tuổi
- **income**: Thu nhập hàng năm ($)
- **became_member_on**: Ngày đăng ký (YYYYMMDD)

### Transcript (~300K transactions)

- **person**: Customer ID
- **event**: offer received, viewed, completed, transaction
- **value**: Offer ID hoặc transaction amount
- **time**: Thời gian (hours)

---

## 🤖 Models

### 1. Deep Neural Network (DNN)

- **Architecture**: Multi-input với Entity Embedding
- **Embedding**: offer_id (200 dims), gender (200 dims)
- **Layers**: Dense(32) → Dense(32) → Dropout(0.2) → Dense(32) → Dense(5)
- **Activation**: ReLU (hidden), Softmax (output)
- **Optimizer**: Adam
- **Loss**: Sparse Categorical Crossentropy

### 2. XGBoost

- **Type**: Gradient Boosting Tree
- **Objective**: multi:softmax
- **Parameters**: max_depth=10, gamma=5
- **Advantages**: Handle imbalanced data tốt hơn

### 3. Random Forest

- **n_estimators**: 150 trees
- **class_weight**: 'balanced_subsample'
- **Advantages**: Built-in class balancing

---

## ⚖️ Handling Imbalanced Dataset

### Problem

Dataset bị **imbalanced** nghiêm trọng:

- Class 3 (offer completed): ~40% (majority)
- Class 1 (offer viewed): ~10% (minority)
- Class 4 (green flag): ~5% (minority)

### Solutions

#### 1. Random Over-Sampling (SMOTE)

```python
from imblearn.over_sampling import RandomOverSampler
sm = RandomOverSampler(sampling_strategy='not majority')
X_train, y_train = sm.fit_sample(X_train, y_train)
```

- ✅ Tất cả classes có recall ~50%
- ⚠️ Trade-off: Overall accuracy giảm nhưng fair hơn

#### 2. Class Weight Adjustment

```python
class_weights = {
    0: 3.2,
    1: 39.0,  # Minority class
    2: 1.4,
    3: 1.0,   # Majority class
    4: 6.0
}
model.fit(..., class_weight=class_weights)
```

---

## 📈 Evaluation Metrics

### Primary Metrics

- **Micro-averaged F1-score**: Best cho imbalanced multiclass
- **Confusion Matrix**: Detailed class-wise performance
- **Recall per class**: Quan trọng cho minority classes

### Results Summary

| Model                  | Imbalanced F1 | Balanced F1 | Best For         |
| ---------------------- | ------------- | ----------- | ---------------- |
| DNN (Label Encoded)    | 61.89%        | -           | Baseline         |
| DNN (Entity Embedding) | **63.12%**    | ~50%        | Best DNN         |
| XGBoost                | 63.45%        | ~50%        | **Best Overall** |
| XGBoost (SMOTE)        | ~50%          | ~50%        | Fair prediction  |
| Random Forest          | 63%           | ~50%        | Ensemble         |

### Key Findings

- **XGBoost** performs best trên imbalanced data
- **Entity Embedding** > One-hot encoding cho categorical features
- **SMOTE** giúp model học fair hơn cho tất cả classes

---

## 🔍 Feature Importance (SHAP Analysis)

Top features ảnh hưởng đến predictions:

1. **offer_id** - Highest impact (loại offer quan trọng nhất)
2. **income** - Strong predictor
3. **age** - Medium impact
4. **difficulty** - Medium impact
5. **reward** - Medium impact
6. **reg_month** - Lowest impact

---

## 📝 Key Takeaways

### 1. Model Selection

- XGBoost > DNN cho tabular data này
- Entity Embedding essential cho categorical features trong DNN
- Tree-based models handle imbalanced data tốt hơn

### 2. Imbalanced Data

- **KHÔNG thể ignore** minority classes trong business context
- RandomOverSampler trade-off: accuracy ↓ nhưng fairness ↑
- Class weights effective nhưng cần tuning cẩn thận

### 3. Business Value

```
Scenario 1 (Imbalanced Model):
✓ High accuracy (63%)
✗ Minority customers ignored
✗ Lost marketing opportunities

Scenario 2 (Balanced Model):
✓ Fair prediction (50% all classes)
✓ Better customer segmentation
✓ Targeted marketing hiệu quả hơn
```

### 4. Next Steps

- Collect thêm data cho minority classes
- Ensemble methods: XGBoost + Random Forest
- Hyperparameter tuning cho oversampled data
- A/B testing trên production data

---

## 🛠️ Tech Stack

- **Python** 3.8+
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **ML Libraries**:
  - scikit-learn (preprocessing, Random Forest)
  - XGBoost (gradient boosting)
  - TensorFlow/Keras (Deep Learning)
  - imbalanced-learn (SMOTE)
- **Model Interpretation**: SHAP
- **Development**: Jupyter Notebook

---

## 👥 Contributors

- **Senior Data Scientist**: Project Lead & Development

---

## 📄 License

This project is for educational purposes.

---

## 📞 Contact

For questions or feedback, please open an issue in the repository.

---

**Happy Modeling! 🚀**
