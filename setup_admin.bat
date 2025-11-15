@echo off
REM Coffee Shop Admin Panel Setup for Windows
REM This script creates admin tables in the coffee_shop database

echo ==================================
echo Coffee Shop Admin Panel Setup
echo ==================================
echo.

REM Check if MySQL is accessible
where mysql >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] MySQL not found in PATH!
    echo Please install MySQL or add it to your PATH.
    echo Example: C:\Program Files\MySQL\MySQL Server 8.0\bin
    pause
    exit /b 1
)

set /p MYSQL_PASSWORD="Enter MySQL root password: "
echo.

echo Creating admin tables...
mysql -u root -p%MYSQL_PASSWORD% coffee_shop < database\admin_schema.sql

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [SUCCESS] Admin tables created successfully!
    echo.
    echo ==================================
    echo Default Admin Account:
    echo ==================================
    echo Username: admin
    echo Password: admin123
    echo.
    echo [WARNING] Please change the default password after first login!
    echo.
    echo To start admin panel, run:
    echo   python admin.py
) else (
    echo.
    echo [ERROR] Failed to create admin tables
    echo Please check your MySQL password and database.
)

echo.
pause
