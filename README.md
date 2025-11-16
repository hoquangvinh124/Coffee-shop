# Coffee Shop Revenue Prediction API

API dự đoán doanh thu quán cafe dựa trên các chỉ số kinh doanh.

## Model Information

- **Model**: XGBoost Regressor (tuned with Optuna)
- **Metrics**:
  - R² Score: 0.953
  - MAPE: 12.93%
  - RMSE: 208.95

## Installation

```bash
pip install -r requirements.txt
```

## Running the API

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Server sẽ chạy tại: `http://localhost:8000`

## API Endpoints

### 1. Health Check
```bash
GET /
GET /health
```

### 2. Model Information
```bash
GET /model-info
```

Trả về thông tin về model đang được sử dụng.

### 3. Predict Revenue
```bash
POST /predict
```

**Request Body:**
```json
{
  "number_of_customers_per_day": 150,
  "average_order_value": 7.5,
  "operating_hours_per_day": 12,
  "number_of_employees": 4,
  "marketing_spend_per_day": 100,
  "location_foot_traffic": 80
}
```

**Response:**
```json
{
  "predicted_revenue": 1547.81
}
```

## Example Usage

### Using cURL
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "number_of_customers_per_day": 150,
    "average_order_value": 7.5,
    "operating_hours_per_day": 12,
    "number_of_employees": 4,
    "marketing_spend_per_day": 100,
    "location_foot_traffic": 80
  }'
```

### Using Python
```python
import requests

url = "http://localhost:8000/predict"
data = {
    "number_of_customers_per_day": 150,
    "average_order_value": 7.5,
    "operating_hours_per_day": 12,
    "number_of_employees": 4,
    "marketing_spend_per_day": 100,
    "location_foot_traffic": 80
}

response = requests.post(url, json=data)
print(response.json())
```

## Interactive API Documentation

FastAPI tự động tạo interactive docs:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Input Features

| Feature | Type | Description | Range |
|---------|------|-------------|-------|
| `number_of_customers_per_day` | float | Số lượng khách hàng dự kiến | > 0 |
| `average_order_value` | float | Giá trị đơn hàng trung bình | > 0 |
| `operating_hours_per_day` | float | Số giờ mở cửa mỗi ngày | 0-24 |
| `number_of_employees` | int | Số nhân viên | > 0 |
| `marketing_spend_per_day` | float | Chi phí marketing mỗi ngày | >= 0 |
| `location_foot_traffic` | int | Lượng người qua lại khu vực | >= 0 |

## Project Structure

```
Coffee-shop/
├── app.py                           # FastAPI application
├── requirements.txt                  # Python dependencies
├── coffee_shop_revenue1.csv         # Training data
├── Coffee_Shop_ML_Pipeline.ipynb    # ML training notebook
├── models/
│   ├── best_model_tuned.pkl         # Trained XGBoost model
│   ├── scaler.pkl                   # Feature scaler
│   └── model_info_tuned.pkl         # Model metadata
└── results/                         # Training results
```

## Notes

- Model được train với dữ liệu từ file `coffee_shop_revenue1.csv`
- Sử dụng StandardScaler để chuẩn hóa features
- Hyperparameters được tối ưu hóa bằng Optuna
