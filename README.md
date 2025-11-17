# Coffee Shop Sales Forecasting - Prophet Model

Dự án dự báo doanh thu Coffee Shop sử dụng mô hình Prophet của Facebook cho forecast 8 năm (2018-2025).

## Cấu trúc Project

```
Coffee-shop/
├── data/                           # Dữ liệu
│   ├── daily_sales_cafe.csv       # Dữ liệu doanh thu hàng ngày (2013-2017)
│   └── holidays_prepared.csv      # Dữ liệu ngày lễ
├── notebooks/                      # Jupyter notebooks
│   └── prophet_forecasting.ipynb  # Notebook chính - Toàn bộ quy trình Prophet
├── models/                         # Trained models
│   └── prophet_model.pkl          # Trained Prophet model
├── results/                        # Kết quả forecast và visualizations
├── app.py                         # FastAPI application
├── requirements.txt               # Python dependencies
├── test_prophet.py                # Script test nhanh Prophet model
└── README.md                      # File này
```

## Mô hình Prophet

### Đặc điểm:
- **Không có biến ngoại sinh (regressors)**: Chỉ dùng date và doanh thu
- **Holidays**: Sử dụng Ecuador country holidays + custom holidays
- **Seasonality**: Multiplicative mode với yearly và weekly patterns

### Cấu hình:
```python
config = {
    'growth': 'linear',
    'changepoint_prior_scale': 0.05,
    'seasonality_mode': 'multiplicative',
    'yearly_seasonality': 20,
    'weekly_seasonality': 10,
    'daily_seasonality': False,
    'interval_width': 0.95
}
```

## Cách chạy

### Option 1: Train Model với Jupyter Notebook
```bash
# Mở notebook
jupyter notebook notebooks/prophet_forecasting.ipynb

# Hoặc với JupyterLab
jupyter lab notebooks/prophet_forecasting.ipynb
```

Notebook bao gồm:
1. Load và khám phá dữ liệu (EDA)
2. Train mô hình Prophet
3. Đánh giá mô hình (MAE, MAPE, RMSE, Coverage)
4. Dự báo 8 năm
5. Phân tích kết quả với visualizations
6. Export kết quả ra CSV và lưu model vào `models/prophet_model.pkl`

### Option 2: Chạy FastAPI (Serving model)
```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Chạy API server
python app.py

# Hoặc với uvicorn
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

API sẽ chạy tại: http://localhost:8000

#### API Endpoints:
- `GET /` - API info và danh sách endpoints
- `GET /health` - Health check
- `POST /forecast` - Dự báo với số ngày
- `GET /forecast/range` - Dự báo theo khoảng thời gian

#### Ví dụ sử dụng:
```bash
# Dự báo 30 ngày
curl -X POST "http://localhost:8000/forecast" \
  -H "Content-Type: application/json" \
  -d '{"days": 30, "start_date": "2025-01-01"}'

# Dự báo theo range
curl "http://localhost:8000/forecast/range?start_date=2025-01-01&end_date=2025-01-31"
```

Swagger UI: http://localhost:8000/docs

### Option 3: Chạy test script (Quick test)
```bash
python test_prophet.py
```

Script này test logic chính của Prophet mà không cần Jupyter.

## Kết quả

Từ test run mẫu:

### Hiệu suất mô hình (In-Sample):
- **MAE**: $11,623
- **RMSE**: $16,332
- **Coverage (95% CI)**: 93.54%

### Dự báo 8 năm (2018-2025):
- **Projected CAGR**: 11.19%
- **Total 8-Year Revenue**: $1,216.42M
- **Average Daily Sales**: Tăng từ $246K (2017) lên $576K (2025)

### Output files (trong folder `results/`):
- `prophet_forecast_full.csv` - Full forecast với tất cả components
- `forecast_2018_2025.csv` - Forecast tương lai (2018-2025)
- `yearly_forecast_summary.csv` - Tổng hợp theo năm
- `model_metrics.csv` - Metrics đánh giá mô hình
- `prophet_model.pkl` - Trained model (để load lại sau)
- `*.png` - 9 visualization plots

## Yêu cầu

Packages cần thiết (đã có trong pyproject.toml):
- pandas
- numpy
- prophet
- matplotlib
- seaborn

## Notes

1. **Mô hình này KHÔNG sử dụng biến ngoại sinh** - chỉ dựa trên:
   - Time series pattern (trend + seasonality)
   - Holidays effects

2. **Để thêm biến ngoại sinh** (regressors) vào Prophet:
   - Cần sửa code để add regressors như `is_weekend`, `month_sin/cos`, etc.
   - Sử dụng `model.add_regressor('feature_name')`

3. **Forecast period**: 8 years (2920 days)

4. **Training data**: 2013-2017 (1688 days)

## Tác giả

Dự án thực hiện với Claude Code
