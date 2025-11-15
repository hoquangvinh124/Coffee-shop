# ☕ Coffee Shop Application

Ứng dụng đặt hàng Coffee Shop đầy đủ tính năng được xây dựng bằng PyQt6 và MySQL.

## 🎯 Tính năng

### ✅ Đã hoàn thành

1. **Đăng nhập & Tài khoản**
   - ✅ Đăng ký tài khoản với email/số điện thoại
   - ✅ Đăng nhập
   - ✅ Xác thực OTP (infrastructure)
   - ✅ Thẻ thành viên (Bronze/Silver/Gold)
   - ✅ Hệ thống điểm thưởng

2. **Menu & Trải nghiệm đặt món**
   - ✅ Xem danh sách sản phẩm theo danh mục
   - ✅ Tìm kiếm món
   - ✅ Lọc theo nhiệt độ, caffeine
   - ✅ Thông tin sản phẩm chi tiết
   - ✅ Tùy chỉnh sản phẩm (size, sugar, ice, toppings)
   - ✅ Tính giá realtime

3. **Giỏ hàng**
   - ✅ Thêm/Xóa/Sửa món
   - ✅ Áp mã giảm giá/voucher
   - ✅ Tính toán tổng tiền

4. **Đơn hàng**
   - ✅ Tạo đơn hàng (Pickup/Delivery/Dine-in)
   - ✅ Theo dõi trạng thái đơn hàng
   - ✅ Lịch sử đơn hàng

5. **Loyalty System**
   - ✅ Tích điểm theo đơn hàng
   - ✅ Hệ thống hạng thành viên
   - ✅ Voucher và khuyến mãi

### 🚧 Đang phát triển (Placeholders)

- Payment integration (MoMo, ZaloPay, ShopeePay)
- AI-based product recommendations
- Real-time order tracking with GPS
- Push notifications
- QR code table ordering
- Review system UI
- Cart, Profile, Orders UI (đã có backend)

## 📋 Yêu cầu hệ thống

- Python 3.8+
- MySQL 8.0+
- PyQt6

## 🚀 Cài đặt

### 1. Clone repository

```bash
git clone <repository-url>
cd Coffee-shop
```

### 2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 3. Cấu hình database

#### Tạo database MySQL:

```bash
mysql -u root -p < database/schema.sql
```

Hoặc import thủ công:

```sql
mysql -u root -p
source database/schema.sql
```

#### Cấu hình kết nối database:

Chỉnh sửa file `utils/config.py`:

```python
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'your_password',
    'database': 'coffee_shop',
    'port': 3306
}
```

Hoặc sử dụng environment variables:

```bash
export DB_HOST=localhost
export DB_USER=root
export DB_PASSWORD=your_password
export DB_NAME=coffee_shop
export DB_PORT=3306
```

### 4. Chạy ứng dụng

```bash
python main.py
```

## 📁 Cấu trúc Project

```
Coffee-shop/
├── ui/                      # UI files (.ui)
├── ui_generated/            # Generated Python files from UI
├── views/                   # Logic files (_ex.py)
│   ├── login_ex.py
│   ├── register_ex.py
│   ├── main_window_ex.py
│   └── menu_ex.py
├── models/                  # Database models
│   ├── user.py
│   ├── product.py
│   ├── cart.py
│   ├── order.py
│   └── ...
├── controllers/             # Business logic
│   ├── auth_controller.py
│   ├── menu_controller.py
│   ├── cart_controller.py
│   └── ...
├── utils/                   # Utilities
│   ├── database.py
│   ├── config.py
│   ├── validators.py
│   └── helpers.py
├── resources/               # Images, icons, styles
│   └── styles/
│       └── style.qss
├── database/
│   └── schema.sql
├── main.py                  # Entry point
└── requirements.txt
```

## 🎨 Thiết kế

Giao diện được thiết kế theo phong cách Highland Coffee - clean, hiện đại với:
- Color palette: Coffee tones (#c7a17a, #d4691e)
- Rounded corners và shadows
- Responsive layout
- User-friendly navigation

## 💾 Database Schema

Database gồm các bảng chính:
- `users` - Thông tin người dùng
- `products` - Sản phẩm
- `categories` - Danh mục
- `toppings` - Topping
- `cart` - Giỏ hàng
- `orders` - Đơn hàng
- `order_items` - Chi tiết đơn hàng
- `vouchers` - Mã giảm giá
- `reviews` - Đánh giá
- `notifications` - Thông báo
- `loyalty_points_history` - Lịch sử điểm
- Và nhiều bảng khác...

## 🔐 Tài khoản Demo

Sau khi chạy schema.sql, bạn có thể đăng ký tài khoản mới hoặc tạo tài khoản demo:

```sql
-- Tạo user demo (password: Demo@123)
INSERT INTO users (email, password_hash, full_name, membership_tier, loyalty_points)
VALUES ('demo@coffeeshop.com',
        '8d969eef6ecad3c29a3a629280e686cf0c3f5d5a86aff3ca12020c923adc6c92',
        'Demo User', 'Gold', 6000);
```

## 📝 To-do List

- [ ] Implement cart UI
- [ ] Implement profile UI
- [ ] Implement order tracking UI with timeline
- [ ] Add product customization dialog
- [ ] Integrate payment gateways
- [ ] Add image upload for products
- [ ] Implement notification system
- [ ] Add QR code generation for table orders
- [ ] Build admin panel
- [ ] Add data analytics dashboard

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 📧 Contact

For questions or support, please contact: [your-email@example.com]

---

**Note**: Đây là project demo/educational. Một số tính năng như payment integration, Google/Apple login cần API keys và configuration bổ sung.
