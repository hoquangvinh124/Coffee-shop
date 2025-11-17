@echo off
REM Coffee Shop Database Setup for Windows
REM Run this to setup complete database

echo ==================================
echo Coffee Shop Database Setup
echo ==================================
echo.
echo This will create:
echo - Database 'coffee_shop'
echo - All tables (users, products, orders, etc.)
echo - Sample data for testing
echo - Admin tables and default admin account
echo.

REM Check if MySQL is accessible
where mysql >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] MySQL not found in PATH!
    echo.
    echo Please add MySQL to your PATH:
    echo 1. Right-click 'This PC' ^> Properties ^> Advanced System Settings
    echo 2. Environment Variables ^> Path ^> Edit
    echo 3. Add: C:\Program Files\MySQL\MySQL Server 8.0\bin
    echo 4. Restart Command Prompt
    echo.
    pause
    exit /b 1
)

set /p MYSQL_PASSWORD="Enter MySQL root password: "
echo.

echo ========================================
echo Step 1/3: Creating main database schema
echo ========================================
mysql -u root -p%MYSQL_PASSWORD% < database\schema.sql
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to create main schema
    pause
    exit /b 1
)
echo [OK] Main schema created

echo.
echo ========================================
echo Step 2/3: Applying schema updates
echo ========================================
mysql -u root -p%MYSQL_PASSWORD% < database\schema_updates.sql
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to apply updates
    pause
    exit /b 1
)
echo [OK] Schema updates applied

echo.
echo ========================================
echo Step 3/3: Creating admin tables
echo ========================================
mysql -u root -p%MYSQL_PASSWORD% coffee_shop < database\admin_schema.sql
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to create admin tables
    pause
    exit /b 1
)
echo [OK] Admin tables created

echo.
echo ==================================
echo Setup completed successfully!
echo ==================================
echo.
echo Database: coffee_shop
echo Tables: 25+ tables created
echo.
echo Default Admin Account:
echo   Username: admin
echo   Password: admin123
echo.
echo [WARNING] Change admin password after first login!
echo.
echo To start the application:
echo   Customer App: python main.py
echo   Admin Panel:  python admin.py
echo.
pause
