#!/usr/bin/env python3
"""
Coffee Shop Application - Main Entry Point
A full-featured coffee shop ordering system built with PyQt6
"""

import sys
from pathlib import Path
from PyQt6.QtWidgets import QApplication, QStackedWidget, QMessageBox, QSplashScreen
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QIcon

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from views.login_ex import LoginWindow
from views.register_ex import RegisterWindow
from views.main_window_ex import MainWindow
from views.menu_ex import MenuWidget
from utils.database import db
from utils.config import STYLES_DIR, APP_NAME


class CoffeeShopApp:
    """Main application class"""

    def __init__(self):
        self.app = QApplication(sys.argv)
        self.app.setApplicationName(APP_NAME)

        # Load stylesheet
        self.load_stylesheet()

        # Initialize windows
        self.login_window = None
        self.register_window = None
        self.main_window = None

        # Create stacked widget for window management
        self.stacked_widget = QStackedWidget()
        self.stacked_widget.setWindowTitle(APP_NAME)
        self.stacked_widget.resize(1200, 800)

        # Test database connection
        self.check_database_connection()

        # Setup windows
        self.setup_windows()

        # Show login window
        self.show_login()

    def load_stylesheet(self):
        """Load application stylesheet"""
        try:
            style_file = STYLES_DIR / 'style.qss'
            if style_file.exists():
                with open(style_file, 'r', encoding='utf-8') as f:
                    self.app.setStyleSheet(f.read())
        except Exception as e:
            print(f"Warning: Could not load stylesheet: {e}")

    def check_database_connection(self):
        """Check database connection on startup"""
        try:
            if not db.test_connection():
                QMessageBox.critical(
                    None,
                    "Lỗi Kết Nối Database",
                    "Không thể kết nối đến database MySQL.\n\n"
                    "Vui lòng kiểm tra:\n"
                    "1. MySQL server đang chạy\n"
                    "2. Thông tin kết nối trong utils/config.py\n"
                    "3. Database 'coffee_shop' đã được tạo\n\n"
                    "Chạy file database/schema.sql để tạo database."
                )
                sys.exit(1)
        except Exception as e:
            QMessageBox.critical(
                None,
                "Lỗi Database",
                f"Lỗi khi kiểm tra database: {str(e)}"
            )
            sys.exit(1)

    def setup_windows(self):
        """Setup all application windows"""
        # Login window
        self.login_window = LoginWindow()
        self.login_window.login_successful.connect(self.handle_login_successful)
        self.login_window.switch_to_register.connect(self.show_register)
        self.stacked_widget.addWidget(self.login_window)

        # Register window
        self.register_window = RegisterWindow()
        self.register_window.switch_to_login.connect(self.show_login)
        self.stacked_widget.addWidget(self.register_window)

    def show_login(self):
        """Show login window"""
        self.stacked_widget.setCurrentWidget(self.login_window)
        self.stacked_widget.show()

    def show_register(self):
        """Show register window"""
        self.stacked_widget.setCurrentWidget(self.register_window)

    def handle_login_successful(self, user_data):
        """Handle successful login"""
        # Create main window
        self.main_window = MainWindow()
        self.main_window.logout_requested.connect(self.handle_logout)

        # Create and add menu widget
        menu_widget = MenuWidget()
        menu_widget.cart_updated.connect(self.main_window.update_cart_count)
        self.main_window.add_content_page(menu_widget)

        # TODO: Add other content pages (cart, orders, profile, etc.)
        # Placeholder pages
        from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout

        # Cart page placeholder
        cart_page = QWidget()
        cart_layout = QVBoxLayout(cart_page)
        cart_layout.addWidget(QLabel("🛒 Giỏ hàng - Đang phát triển"))
        self.main_window.add_content_page(cart_page)

        # Orders page placeholder
        orders_page = QWidget()
        orders_layout = QVBoxLayout(orders_page)
        orders_layout.addWidget(QLabel("📦 Đơn hàng - Đang phát triển"))
        self.main_window.add_content_page(orders_page)

        # Favorites page placeholder
        favorites_page = QWidget()
        favorites_layout = QVBoxLayout(favorites_page)
        favorites_layout.addWidget(QLabel("❤️ Yêu thích - Đang phát triển"))
        self.main_window.add_content_page(favorites_page)

        # Profile page placeholder
        profile_page = QWidget()
        profile_layout = QVBoxLayout(profile_page)
        profile_layout.addWidget(QLabel("👤 Tài khoản - Đang phát triển"))
        self.main_window.add_content_page(profile_page)

        # Notifications page placeholder
        notifications_page = QWidget()
        notifications_layout = QVBoxLayout(notifications_page)
        notifications_layout.addWidget(QLabel("🔔 Thông báo - Đang phát triển"))
        self.main_window.add_content_page(notifications_page)

        # Switch to menu page
        self.main_window.switch_page(0)

        # Hide stacked widget and show main window
        self.stacked_widget.hide()
        self.main_window.show()

    def handle_logout(self):
        """Handle user logout"""
        # Close main window
        if self.main_window:
            self.main_window.close()
            self.main_window = None

        # Clear login form and show login window
        self.login_window.clear_form()
        self.show_login()

    def run(self):
        """Run the application"""
        return self.app.exec()


def main():
    """Main entry point"""
    try:
        app = CoffeeShopApp()
        sys.exit(app.run())
    except Exception as e:
        print(f"Critical error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
