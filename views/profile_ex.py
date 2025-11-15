"""
Profile Widget - Extended Logic
User profile management (Placeholder)
"""
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel

# TODO: Implement full profile UI with:
# - User information display and edit
# - Avatar upload
# - Membership tier and points
# - Preferences (favorite size, sugar level, etc.)
# - Password change
# - Saved payment methods
# - Badges display


class ProfileWidget(QWidget):
    """User profile widget (placeholder)"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        """Setup UI"""
        layout = QVBoxLayout(self)
        label = QLabel("👤 Tài khoản\n\nTính năng đang được phát triển...")
        label.setStyleSheet("font-size: 18px; padding: 50px;")
        layout.addWidget(label)
