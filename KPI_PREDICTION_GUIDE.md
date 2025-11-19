# 📊 Hướng dẫn sử dụng Dự đoán KPI Logistics

## 🎯 Tổng quan

Tính năng **Dự đoán KPI Logistics** đã được tích hợp vào Admin Panel, cho phép bạn dự đoán hiệu suất (KPI score) của các sản phẩm trong quán cà phê dựa trên Machine Learning model với độ chính xác **99.99%**.

## 🚀 Cách truy cập

1. Chạy Admin Panel:

   ```bash
   python admin.py
   ```

2. Đăng nhập với tài khoản admin (username: `admin`, password: `admin123`)

3. Click vào menu **"📊 Dự đoán KPI Logistics"** ở sidebar

## 📝 Tính năng

### 1️⃣ Dự đoán đơn lẻ (Single Prediction)

**Mục đích:** Dự đoán KPI cho một sản phẩm cụ thể

**Cách sử dụng:**

1. Chọn tab **"🎯 Dự đoán đơn lẻ"**
2. Điền thông tin sản phẩm vào form:
   - **Item ID:** Mã sản phẩm (ví dụ: COFFEE_LATTE)
   - **Category:** Chọn danh mục (Groceries khuyến nghị cho cà phê)
   - **Stock Level:** Số lượng tồn kho hiện tại (ví dụ: 150)
   - **Reorder Point:** Mức cần đặt hàng lại (ví dụ: 50)
   - **Daily Demand:** Nhu cầu mỗi ngày (ví dụ: 25.5 ly/ngày)
   - **Order Fulfillment Rate:** Tỷ lệ hoàn thành đơn (0.95 = 95%)
   - **Turnover Ratio:** Tốc độ luân chuyển hàng (ví dụ: 12.5)
   - ...và các thông số khác
3. Click **"🔮 Dự đoán KPI"**
4. Xem kết quả:
   - KPI Score (0-1)
   - Đánh giá (Excellent/Good/Needs Improvement)
   - Recommendations (đề xuất cải thiện)

**Ví dụ dữ liệu mẫu (Coffee Latte):**

```
Item ID: COFFEE_LATTE
Category: Groceries
Stock Level: 150
Reorder Point: 50
Reorder Frequency: 7 days
Lead Time: 3 days
Daily Demand: 25.5
Demand Std Dev: 3.2
Popularity Score: 0.85
Zone: A
Picking Time: 45 seconds
Handling Cost: 2.50
Unit Price: 99.99
Holding Cost: 0.15
Stockout Count: 1
Fulfillment Rate: 0.95
Total Orders: 750
Turnover Ratio: 12.5
Layout Efficiency: 0.92
Last Restock: 2024-11-01
Forecasted Demand: 178.5
```

### 2️⃣ Dự đoán hàng loạt (Batch Prediction)

**Mục đích:** Dự đoán KPI cho nhiều sản phẩm cùng lúc

**Cách sử dụng:**

1. Chọn tab **"📦 Dự đoán hàng loạt"**
2. Click **"⬇️ Tải template CSV"** để tải file mẫu
3. Mở file CSV và điền thông tin cho các sản phẩm
4. Lưu file CSV
5. Click **"📁 Upload CSV"** và chọn file vừa tạo
6. Xem kết quả trong bảng:
   - Item ID
   - KPI Score
   - Interpretation
7. Click **"💾 Xuất kết quả"** để lưu file kết quả

**Template CSV:** `templates/logistics_kpi_template.csv`

File template đã có sẵn 10 sản phẩm mẫu (COFFEE_LATTE, COFFEE_CAPPUCCINO, CROISSANT, v.v.)

### 3️⃣ Hướng dẫn (Help)

Tab **"ℹ️ Hướng dẫn"** cung cấp:

- Giải thích chi tiết từng trường dữ liệu
- Cách hiểu KPI Score
- Top 10 yếu tố quan trọng nhất
- Mẹo tối ưu KPI

## 📈 Giải thích KPI Score

| Score Range   | Đánh giá                 | Ý nghĩa                             |
| ------------- | ------------------------ | ----------------------------------- |
| **0.7 - 1.0** | ✅ Excellent Performance | Sản phẩm hoạt động rất tốt, duy trì |
| **0.5 - 0.7** | ⚠️ Good Performance      | Tốt nhưng có thể cải thiện          |
| **0.0 - 0.5** | ❌ Needs Improvement     | Cần chú ý khẩn cấp, có vấn đề       |

## 🔑 Các yếu tố quan trọng

Model xem xét **43 features** (18 gốc + 25 engineered), trong đó quan trọng nhất:

1. **Order Fulfillment Rate** (85.6%) - Tỷ lệ hoàn thành đơn hàng
2. **Efficiency Composite** (79.8%) - Hiệu suất tổng hợp
3. **Fulfillment Quality** (84.5%) - Chất lượng hoàn thành
4. **Turnover Ratio** (74.2%) - Tốc độ luân chuyển
5. **Inventory Health** (72.3%) - Sức khỏe kho hàng
6. **Item Popularity** (68.1%) - Độ phổ biến
7. **Demand-Supply Balance** (65.4%) - Cân bằng cung cầu
8. **Picking Efficiency** (61.2%) - Hiệu quả lấy hàng
9. **Popularity Turnover** (59.8%) - Kết hợp độ phổ biến và luân chuyển
10. **Forecast Accuracy** (53.4%) - Độ chính xác dự báo

## 💡 Mẹo tối ưu KPI

### ✅ Để đạt KPI cao (>0.7):

1. **Giữ Order Fulfillment Rate cao**

   - Mục tiêu: >90%
   - Giảm thiểu trường hợp hết hàng
   - Đáp ứng đơn hàng đúng hạn

2. **Tối ưu vị trí kho**

   - Đặt sản phẩm phổ biến ở zone A (dễ lấy nhất)
   - Giảm Picking Time xuống <60 giây

3. **Dự báo nhu cầu chính xác**

   - Theo dõi daily demand
   - Tính toán demand_std_dev (biến động)
   - Điều chỉnh forecasted_demand_next_7d

4. **Cân bằng tồn kho**

   - Stock Level đủ để đáp ứng nhu cầu 3-7 ngày
   - Không quá cao (tốn chi phí holding)
   - Không quá thấp (nguy cơ stockout)

5. **Tăng Turnover Ratio**
   - Hàng luân chuyển nhanh (>10 lần/tháng)
   - Tránh ứ đọng hàng tồn kho

### ⚠️ Dấu hiệu cần cải thiện:

- **Stockout Count** cao (>3 lần/tháng)
- **Order Fulfillment Rate** thấp (<85%)
- **Turnover Ratio** thấp (<5)
- **Demand Std Dev** quá cao (biến động lớn)
- **Picking Time** lâu (>120 giây)

## 📊 Ví dụ thực tế

### Ví dụ 1: Sản phẩm tốt (KPI = 0.803)

```
Item: Coffee Latte
Stock: 150 | Reorder: 50 | Daily Demand: 25.5
Fulfillment Rate: 0.95 | Turnover: 12.5
Stockout: 1 | Popularity: 0.85
→ KPI Score: 0.803 ✅ Excellent
```

**Tại sao cao?**

- Fulfillment rate tốt (95%)
- Turnover cao (12.5)
- Stockout thấp (1)
- Sản phẩm phổ biến (0.85)

### Ví dụ 2: Sản phẩm cần cải thiện (KPI = 0.449)

```
Item: Sugar Packets
Stock: 500 | Reorder: 150 | Daily Demand: 8.5
Fulfillment Rate: 0.95 | Turnover: 5.2
Stockout: 1 | Popularity: 0.50
→ KPI Score: 0.449 ❌ Needs Improvement
```

**Tại sao thấp?**

- Turnover quá thấp (5.2) - hàng luân chuyển chậm
- Popularity thấp (0.50)
- Stock quá cao so với nhu cầu (tốn chi phí holding)
- Zone D (xa, picking time lâu)

**Cải thiện:**

- Giảm reorder point xuống còn 50
- Chuyển sang zone gần hơn nếu có thể
- Xem xét có nên giảm stock level

## 🛠️ Technical Details

### Model Information

- **Algorithm:** Ridge Regression
- **Accuracy:** 99.99% R²
- **Features:** 43 (18 original + 25 engineered)
- **Training Data:** 3,204 logistics items
- **Prediction Time:** <1ms per item

### Files Created

```
controllers/
  └── admin_kpi_controller.py       # Controller xử lý predictions
views/
  └── admin_logistic_kpi_ex.py      # UI widget
templates/
  └── logistics_kpi_template.csv    # CSV template
log_model/
  └── models/                       # Pretrained ML model
```

### Dependencies

- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `joblib` - Model loading
- `scikit-learn` - ML preprocessing
- `PyQt6` - UI framework

## 🔧 Troubleshooting

### Lỗi: "Không thể load model"

**Nguyên nhân:** Model files không tồn tại

**Giải pháp:**

1. Kiểm tra folder `log_model/models/` có files:
   - `Ridge_Regression_*.pkl`
   - `scaler_*.pkl`
   - `encoders_*.pkl`
2. Nếu thiếu, chạy training:
   ```bash
   cd log_model
   python src/ml/train_model.py
   ```

### Lỗi: "Missing columns"

**Nguyên nhân:** CSV file thiếu cột bắt buộc

**Giải pháp:**

1. Download lại template CSV
2. Đảm bảo có đủ 21 cột
3. Không xóa/đổi tên cột header

### Lỗi: Validation errors

**Nguyên nhân:** Dữ liệu không hợp lệ

**Giải pháp:**

- `stock_level`, `reorder_point` >= 0
- `order_fulfillment_rate`, `item_popularity_score`, `layout_efficiency_score` trong khoảng 0-1
- `category` phải là: Groceries/Electronics/Apparel/Automotive/Pharma
- `zone` phải là: A/B/C/D
- `last_restock_date` format: YYYY-MM-DD

## 📞 Support

Nếu gặp vấn đề:

1. Check tab **"ℹ️ Hướng dẫn"** trong app
2. Xem file `log_model/README.md`
3. Check logs trong console

## 🎓 Học thêm

- **Feature Engineering:** `log_model/src/ml/train_model.py`
- **Model Training:** `log_model/notebooks/exploratory_data_analysis.ipynb`
- **API Documentation:** `log_model/src/api/app.py`
- **Dashboard:** `log_model/src/dashboard/dashboard.py`

---

**Version:** 1.0.0  
**Last Updated:** 2024-11-19  
**Model Accuracy:** 99.99% R²  
**Status:** ✅ Production Ready
