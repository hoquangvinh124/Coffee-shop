# ⚡ QUICK START GUIDE

## 🚀 Lệnh siêu nhanh (chỉ 1 dòng)

### Windows CMD/PowerShell:

```bash
# Khởi động (CMD)
start.bat

# Dừng (CMD)
stop.bat

# PowerShell (linh hoạt hơn)
.\quick.ps1 start
.\quick.ps1 stop
.\quick.ps1 status
.\quick.ps1 restart
```

### Từ thư mục scripts:

```bash
scripts\startup.bat   # Khởi động
scripts\shutdown.bat  # Dừng
scripts\status.bat    # Kiểm tra
scripts\restart.bat   # Khởi động lại
```

---

## 📊 Cải tiến tốc độ

**Trước đây:**

- ⏱️ Startup: ~11 giây (5s API wait + 3s Dashboard wait + 3s browser)
- 🐌 Health check chậm
- 🔄 Sequential operations

**Bây giờ:**

- ⚡ Startup: **~3 giây** (2s API + 1s Dashboard)
- ⚠️ Skip health check khi startup (check sau bằng `status.bat`)
- 🚀 Browsers mở background (không block)
- ✅ Services lên ngay, verify sau

**Tổng tiết kiệm: ~8 giây mỗi lần start!**

---

## 💡 Cách dùng tối ưu

### Workflow nhanh nhất:

```bash
# 1. Khởi động (3 giây)
start.bat

# 2. Làm việc với dự án
#    - API: http://localhost:8000/docs
#    - Dashboard: http://localhost:8501

# 3. Kiểm tra nếu cần
scripts\status.bat

# 4. Dừng khi xong
stop.bat
```

### PowerShell shortcuts (nếu thích gõ ngắn):

```powershell
# Tạo aliases (chạy 1 lần)
Set-Alias -Name start-project -Value "$PWD\start.bat"
Set-Alias -Name stop-project -Value "$PWD\stop.bat"

# Sau đó chỉ cần:
start-project  # Khởi động
stop-project   # Dừng
```

---

## 🎯 So sánh các cách khởi động

| Phương pháp             | Lệnh                                         | Thời gian | Linh hoạt  |
| ----------------------- | -------------------------------------------- | --------- | ---------- |
| **Cách 1 (Nhanh nhất)** | `start.bat`                                  | 3s        | ⭐⭐⭐     |
| **Cách 2 (PowerShell)** | `.\quick.ps1 start`                          | 3s        | ⭐⭐⭐⭐⭐ |
| **Cách 3 (Đầy đủ)**     | `scripts\startup.bat`                        | 3s        | ⭐⭐⭐⭐   |
| **Cách 4 (Manual)**     | Activate venv + python app.py + streamlit... | ~30s      | ⭐         |

---

## 🔧 Tối ưu thêm (Optional)

### 1. Windows Terminal Profile:

Thêm vào `settings.json`:

```json
{
  "name": "ML Project - Start",
  "commandline": "cmd.exe /k \"cd /d E:\\Nam3\\TaiLieuHocKi6\\ML\\finalML\\Coffee-shop\\log_model && start.bat\"",
  "icon": "🚀"
}
```

### 2. Desktop Shortcut:

```
Target: E:\Nam3\TaiLieuHocKi6\ML\finalML\Coffee-shop\log_model\start.bat
Start in: E:\Nam3\TaiLieuHocKi6\ML\finalML\Coffee-shop\log_model
Icon: Bất kỳ
```

### 3. Task Scheduler (Auto-start on login):

```powershell
# Chạy 1 lần để tạo scheduled task
$action = New-ScheduledTaskAction -Execute "$PWD\start.bat"
$trigger = New-ScheduledTaskTrigger -AtLogon
Register-ScheduledTask -TaskName "ML_Project_AutoStart" -Action $action -Trigger $trigger
```

---

## 📝 Ghi chú

- ✅ Không cần activate venv manual (script tự động)
- ✅ Không cần kiểm tra port (script tự clean)
- ✅ Không cần mở browser manual (tự động sau 5s)
- ✅ Logs tự động lưu tại `logs/`

**Thời gian setup: 0 giây | Thời gian start: 3 giây | Thời gian stop: 1 giây**

🎉 **Enjoy coding!**
