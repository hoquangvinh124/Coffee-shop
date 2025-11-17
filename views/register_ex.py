"""
Register Window - Extended Logic
"""
from PyQt6.QtWidgets import QWidget, QMessageBox
from PyQt6.QtCore import pyqtSignal
from ui_generated.register import Ui_RegisterWindow
from controllers.auth_controller import AuthController


class RegisterWindow(QWidget, Ui_RegisterWindow):
    """Register window with business logic"""

    # Signals
    register_successful = pyqtSignal()
    switch_to_login = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.auth_controller = AuthController()

        # Connect signals
        self.registerButton.clicked.connect(self.handle_register)
        self.backToLoginButton.clicked.connect(self.switch_to_login.emit)

    def handle_register(self):
        """Handle register button click"""
        # Get form data
        full_name = self.fullNameLineEdit.text().strip()
        email = self.emailLineEdit.text().strip()
        phone = self.phoneLineEdit.text().strip()
        password = self.passwordLineEdit.text()
        confirm_password = self.confirmPasswordLineEdit.text()

        # Validate inputs
        if not full_name:
            QMessageBox.warning(self, "Lỗi", "Vui lòng nhập họ và tên")
            self.fullNameLineEdit.setFocus()
            return

        if not email:
            QMessageBox.warning(self, "Lỗi", "Vui lòng nhập email")
            self.emailLineEdit.setFocus()
            return

        if not password:
            QMessageBox.warning(self, "Lỗi", "Vui lòng nhập mật khẩu")
            self.passwordLineEdit.setFocus()
            return

        if password != confirm_password:
            QMessageBox.warning(self, "Lỗi", "Mật khẩu xác nhận không khớp")
            self.confirmPasswordLineEdit.clear()
            self.confirmPasswordLineEdit.setFocus()
            return

        if not self.termsCheckBox.isChecked():
            QMessageBox.warning(self, "Lỗi", "Vui lòng đồng ý với Điều khoản sử dụng")
            return

        # Disable register button during processing
        self.registerButton.setEnabled(False)
        self.registerButton.setText("Đang đăng ký...")

        # Attempt registration
        phone_value = phone if phone else None
        success, message, user_id = self.auth_controller.register(
            email, password, full_name, phone_value
        )

        # Re-enable button
        self.registerButton.setEnabled(True)
        self.registerButton.setText("Đăng ký")

        if success:
            QMessageBox.information(self, "Thành công", message)
            self.clear_form()
            self.register_successful.emit()
            self.switch_to_login.emit()
        else:
            QMessageBox.warning(self, "Lỗi", message)

    def clear_form(self):
        """Clear all form fields"""
        self.fullNameLineEdit.clear()
        self.emailLineEdit.clear()
        self.phoneLineEdit.clear()
        self.passwordLineEdit.clear()
        self.confirmPasswordLineEdit.clear()
        self.termsCheckBox.setChecked(False)
