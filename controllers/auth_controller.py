"""
Authentication Controller
Handles login, registration, password reset, OTP verification
"""
from typing import Optional, Dict, Any, Tuple
from models.user import User
from utils.validators import (validate_email, validate_password, validate_phone,
                              validate_full_name, validate_otp)
from utils.helpers import generate_otp, session
from utils.database import db
from datetime import datetime, timedelta
from utils.config import OTP_EXPIRY_MINUTES


class AuthController:
    """Authentication controller"""

    @staticmethod
    def register(email: str, password: str, full_name: str, phone: Optional[str] = None) -> Tuple[bool, str, Optional[int]]:
        """
        Register a new user
        Returns: (success, message, user_id)
        """
        # Validate email
        is_valid, error = validate_email(email)
        if not is_valid:
            return False, error, None

        # Validate password
        is_valid, error = validate_password(password)
        if not is_valid:
            return False, error, None

        # Validate full name
        is_valid, error = validate_full_name(full_name)
        if not is_valid:
            return False, error, None

        # Validate phone if provided
        if phone:
            is_valid, error = validate_phone(phone)
            if not is_valid:
                return False, error, None

            # Check if phone exists
            if User.phone_exists(phone):
                return False, "Số điện thoại đã được sử dụng", None

        # Check if email exists
        if User.email_exists(email):
            return False, "Email đã được sử dụng", None

        # Create user
        user_id = User.create(email, password, full_name, phone)

        if user_id:
            return True, "Đăng ký thành công!", user_id
        else:
            return False, "Đăng ký thất bại. Vui lòng thử lại.", None

    @staticmethod
    def login(email: str, password: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Login user
        Returns: (success, message, user_data)
        """
        # Validate email
        is_valid, error = validate_email(email)
        if not is_valid:
            return False, error, None

        # Authenticate
        user = User.authenticate(email, password)

        if user:
            # Update session
            session.login(user)
            return True, "Đăng nhập thành công!", user
        else:
            return False, "Email hoặc mật khẩu không đúng", None

    @staticmethod
    def logout():
        """Logout user"""
        session.logout()

    @staticmethod
    def send_otp(identifier: str, purpose: str = 'registration') -> Tuple[bool, str]:
        """
        Send OTP to email or phone
        Returns: (success, message)
        """
        # Determine if identifier is email or phone
        is_email = '@' in identifier

        if is_email:
            is_valid, error = validate_email(identifier)
            if not is_valid:
                return False, error
        else:
            is_valid, error = validate_phone(identifier)
            if not is_valid:
                return False, error

        # Generate OTP
        otp_code = generate_otp()

        # Calculate expiry
        expires_at = datetime.now() + timedelta(minutes=OTP_EXPIRY_MINUTES)

        # Save OTP to database
        query = """
            INSERT INTO otp_codes (email, phone, otp_code, purpose, expires_at)
            VALUES (%s, %s, %s, %s, %s)
        """

        email = identifier if is_email else None
        phone = identifier if not is_email else None

        otp_id = db.insert(query, (email, phone, otp_code, purpose, expires_at))

        if otp_id:
            # In production, send OTP via email/SMS service
            # For now, just log it (in real app, you'd use SMTP or SMS gateway)
            print(f"OTP Code: {otp_code} for {identifier}")
            return True, f"Mã OTP đã được gửi đến {identifier}"
        else:
            return False, "Không thể gửi mã OTP. Vui lòng thử lại."

    @staticmethod
    def verify_otp(identifier: str, otp_code: str, purpose: str = 'registration') -> Tuple[bool, str]:
        """
        Verify OTP code
        Returns: (success, message)
        """
        # Validate OTP
        is_valid, error = validate_otp(otp_code)
        if not is_valid:
            return False, error

        # Check if identifier is email or phone
        is_email = '@' in identifier

        # Query OTP
        if is_email:
            query = """
                SELECT * FROM otp_codes
                WHERE email = %s AND otp_code = %s AND purpose = %s
                  AND is_used = FALSE AND expires_at > NOW()
                ORDER BY created_at DESC
                LIMIT 1
            """
        else:
            query = """
                SELECT * FROM otp_codes
                WHERE phone = %s AND otp_code = %s AND purpose = %s
                  AND is_used = FALSE AND expires_at > NOW()
                ORDER BY created_at DESC
                LIMIT 1
            """

        otp = db.fetch_one(query, (identifier, otp_code, purpose))

        if otp:
            # Mark OTP as used
            db.execute_query("UPDATE otp_codes SET is_used = TRUE WHERE id = %s", (otp['id'],))

            # If verifying for registration, mark email/phone as verified
            if purpose == 'registration':
                if is_email:
                    user = User.get_by_email(identifier)
                    if user:
                        User.verify_email(user['id'])
                else:
                    user = User.get_by_phone(identifier)
                    if user:
                        User.verify_phone(user['id'])

            return True, "Xác thực thành công!"
        else:
            return False, "Mã OTP không đúng hoặc đã hết hạn"

    @staticmethod
    def reset_password(email: str, new_password: str, otp_code: str) -> Tuple[bool, str]:
        """
        Reset password with OTP verification
        Returns: (success, message)
        """
        # Verify OTP first
        success, message = AuthController.verify_otp(email, otp_code, 'password_reset')
        if not success:
            return False, message

        # Validate new password
        is_valid, error = validate_password(new_password)
        if not is_valid:
            return False, error

        # Get user
        user = User.get_by_email(email)
        if not user:
            return False, "Không tìm thấy tài khoản"

        # Update password
        if User.update_password(user['id'], new_password):
            return True, "Đặt lại mật khẩu thành công!"
        else:
            return False, "Không thể đặt lại mật khẩu. Vui lòng thử lại."

    @staticmethod
    def change_password(user_id: int, old_password: str, new_password: str) -> Tuple[bool, str]:
        """
        Change password (requires old password)
        Returns: (success, message)
        """
        # Get user
        user = User.get_by_id(user_id)
        if not user:
            return False, "Không tìm thấy tài khoản"

        # Verify old password
        from utils.helpers import verify_password
        user_full = db.fetch_one("SELECT password_hash FROM users WHERE id = %s", (user_id,))
        if not user_full or not verify_password(old_password, user_full['password_hash']):
            return False, "Mật khẩu hiện tại không đúng"

        # Validate new password
        is_valid, error = validate_password(new_password)
        if not is_valid:
            return False, error

        # Update password
        if User.update_password(user_id, new_password):
            return True, "Đổi mật khẩu thành công!"
        else:
            return False, "Không thể đổi mật khẩu. Vui lòng thử lại."

    @staticmethod
    def is_logged_in() -> bool:
        """Check if user is logged in"""
        return session.is_logged_in

    @staticmethod
    def get_current_user() -> Optional[Dict[str, Any]]:
        """Get current logged in user"""
        return session.user_data

    @staticmethod
    def get_current_user_id() -> Optional[int]:
        """Get current user ID"""
        return session.user_id
