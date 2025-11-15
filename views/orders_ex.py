"""
Orders Widget - Extended Logic
Order history and tracking (Placeholder)
"""
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel

# TODO: Implement full orders UI with:
# - Order history list
# - Order details view
# - Order tracking with status timeline
# - Reorder functionality
# - Cancel order
# - Review products


class OrdersWidget(QWidget):
    """Orders history widget (placeholder)"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        """Setup UI"""
        layout = QVBoxLayout(self)
        label = QLabel("📦 Đơn hàng\n\nTính năng đang được phát triển...")
        label.setStyleSheet("font-size: 18px; padding: 50px;")
        layout.addWidget(label)
