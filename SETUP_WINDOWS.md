# Hướng dẫn Setup trên Windows

Hướng dẫn chi tiết cài đặt Coffee Shop Application trên Windows.

## 📋 Yêu cầu

- ✅ Python 3.8+ ([Download](https://www.python.org/downloads/))
- ✅ MySQL 8.0+ hoặc XAMPP ([MySQL Download](https://dev.mysql.com/downloads/mysql/) | [XAMPP Download](https://www.apachefriends.org/download.html))
- ✅ PyQt6 và dependencies (sẽ cài bằng pip)

## 🚀 Cài đặt

### Bước 1: Clone hoặc Download project

```cmd
git clone <repository-url>
cd Coffee-shop
```

Hoặc download ZIP và giải nén.

### Bước 2: Cài đặt Python dependencies

Mở **Command Prompt** hoặc **PowerShell** tại thư mục project:

```cmd
pip install PyQt6
pip install mysql-connector-python
```

### Bước 3: Setup Database

Bạn có **3 cách** để setup database:

---

## Cách 1: Dùng Script tự động (Khuyên dùng) ⭐

### Thêm MySQL vào PATH (nếu chưa có)

1. Tìm thư mục cài MySQL, thường là:
   - `C:\Program Files\MySQL\MySQL Server 8.0\bin`
   - Hoặc nếu dùng XAMPP: `C:\xampp\mysql\bin`

2. Thêm vào PATH:
   - Right-click **This PC** → **Properties**
   - **Advanced system settings** → **Environment Variables**
   - Trong **System variables**, chọn **Path** → **Edit**
   - Click **New** → Paste đường dẫn MySQL bin
   - Click **OK** → **OK** → **OK**

3. **Restart Command Prompt** (quan trọng!)

4. Test:
   ```cmd
   mysql --version
   ```

### Chạy script setup:

```cmd
setup_database.bat
```

Script sẽ tự động:
- ✅ Tạo database `coffee_shop`
- ✅ Tạo tất cả bảng
- ✅ Import sample data
- ✅ Tạo bảng admin
- ✅ Tạo tài khoản admin mặc định

Nhập password MySQL root khi được hỏi.

---

## Cách 2: Dùng MySQL Workbench (GUI)

### Bước 1: Mở MySQL Workbench

1. Kết nối tới MySQL server (localhost)
2. Nhập password root

### Bước 2: Chạy schema chính

1. Mở file: **File** → **Open SQL Script**
2. Chọn file: `database/schema.sql`
3. Click icon **⚡ Execute** (hoặc Ctrl+Shift+Enter)
4. Đợi hoàn tất

### Bước 3: Chạy schema updates

1. Mở file: `database/schema_updates.sql`
2. Click **⚡ Execute**

### Bước 4: Chạy admin schema

1. Mở file: `database/admin_schema.sql`
2. Click **⚡ Execute**

### Bước 5: Verify

Chạy query:
```sql
USE coffee_shop;
SHOW TABLES;
```

Phải thấy 25+ bảng.

---

## Cách 3: Dùng phpMyAdmin (nếu dùng XAMPP)

### Bước 1: Khởi động XAMPP

1. Mở **XAMPP Control Panel**
2. Start **Apache** và **MySQL**

### Bước 2: Mở phpMyAdmin

1. Mở browser: http://localhost/phpmyadmin
2. Tab **SQL**

### Bước 3: Import files

1. Copy toàn bộ nội dung file `database/schema.sql`
2. Paste vào phpMyAdmin → Click **Go**
3. Lặp lại với `database/schema_updates.sql`
4. Lặp lại với `database/admin_schema.sql`

---

## ⚙️ Cấu hình

### Sửa file `utils/config.py`:

```python
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',  # ← Nhập password MySQL của bạn (XAMPP thì để trống)
    'database': 'coffee_shop',
    'port': 3306
}
```

**Lưu ý XAMPP:**
- User: `root`
- Password: `` (để trống)
- Port: `3306`

---

## 🎮 Chạy ứng dụng

### Customer App:

```cmd
python main.py
```

### Admin Panel:

```cmd
python admin.py
```

**Tài khoản admin mặc định:**
- Username: `admin`
- Password: `admin123`

---

## 🐛 Troubleshooting

### Lỗi: 'mysql' is not recognized

**Nguyên nhân:** MySQL chưa có trong PATH

**Giải pháp:**
- Thêm MySQL bin vào PATH (xem hướng dẫn trên)
- Hoặc dùng MySQL Workbench/phpMyAdmin

### Lỗi: Access denied for user 'root'

**Nguyên nhân:** Sai password

**Giải pháp:**
- Kiểm tra password MySQL root
- Nếu dùng XAMPP mặc định password là rỗng: `password: ''`
- Sửa lại trong `utils/config.py`

### Lỗi: No module named 'PyQt6'

**Giải pháp:**
```cmd
pip install PyQt6
```

### Lỗi: No module named 'mysql'

**Giải pháp:**
```cmd
pip install mysql-connector-python
```

### Lỗi: Can't connect to MySQL server

**Giải pháp:**
1. Kiểm tra MySQL đang chạy:
   - Task Manager → Services → MySQL80 (hoặc MySQL)
   - Hoặc XAMPP Control Panel → MySQL running
2. Kiểm tra port 3306 không bị chặn
3. Test kết nối:
   ```cmd
   mysql -u root -p
   ```

### Database đã tạo nhưng app lỗi

**Giải pháp:**
1. Kiểm tra đã chạy đủ 3 file SQL:
   - `schema.sql`
   - `schema_updates.sql` ⭐ (quan trọng!)
   - `admin_schema.sql`

2. Verify trong MySQL:
   ```sql
   USE coffee_shop;

   -- Kiểm tra bảng mới
   SHOW TABLES LIKE 'voucher_usage';
   SHOW TABLES LIKE 'order_status_history';
   SHOW TABLES LIKE 'favorites';

   -- Kiểm tra field mới
   SHOW COLUMNS FROM categories LIKE 'icon';
   SHOW COLUMNS FROM products LIKE 'is_new';
   ```

### App chạy nhưng admin không đăng nhập được

**Giải pháp:**
```sql
-- Kiểm tra admin account
SELECT * FROM admin_users WHERE username = 'admin';

-- Nếu không có, tạo lại:
INSERT INTO admin_users (username, email, password_hash, full_name, role)
VALUES ('admin', 'admin@coffeeshop.com', SHA2('admin123', 256), 'Administrator', 'super_admin');
```

---

## 📁 Cấu trúc thư mục

```
Coffee-shop/
├── main.py                    # Customer app entry point
├── admin.py                   # Admin panel entry point
├── setup_database.bat         # ⭐ Setup script cho Windows
├── setup_admin.bat            # Admin setup script
├── database/
│   ├── schema.sql             # Main schema
│   ├── schema_updates.sql     # ⭐ Updates (PHẢI CHẠY!)
│   └── admin_schema.sql       # Admin tables
├── utils/
│   ├── config.py              # ⚙️ Cấu hình database
│   └── database.py            # Database connection
├── controllers/               # Business logic
├── views/                     # UI logic
├── ui_generated/             # Generated UI files
└── resources/                # Styles, images
```

---

## ✅ Checklist

Trước khi chạy app, đảm bảo:

- [ ] Python 3.8+ đã cài
- [ ] MySQL/XAMPP đang chạy
- [ ] Đã chạy `schema.sql`
- [ ] Đã chạy `schema_updates.sql` ⭐
- [ ] Đã chạy `admin_schema.sql`
- [ ] Đã sửa password trong `utils/config.py`
- [ ] Test kết nối database thành công
- [ ] Đã cài PyQt6 và mysql-connector-python

---

## 🎯 Quick Start (TL;DR)

```cmd
# 1. Cài dependencies
pip install PyQt6 mysql-connector-python

# 2. Setup database (chọn 1 trong 3 cách)
setup_database.bat
# HOẶC dùng MySQL Workbench
# HOẶC dùng phpMyAdmin

# 3. Sửa config
# Edit utils/config.py → Điền password MySQL

# 4. Chạy
python main.py      # Customer app
python admin.py     # Admin panel
```

---

## 📞 Support

Nếu vẫn gặp lỗi:
1. Check MySQL error log
2. Check Python traceback
3. Đảm bảo MySQL đang chạy
4. Đảm bảo đã chạy đủ 3 file SQL
5. Test connection: `python -c "from utils.database import db; print(db.test_connection())"`

Good luck! ☕
