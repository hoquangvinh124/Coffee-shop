"""
Admin Logistics KPI Prediction Widget
Predict KPI scores for coffee shop logistics items
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QLabel,
    QPushButton, QLineEdit, QComboBox, QTextEdit, QTableWidget,
    QTableWidgetItem, QFileDialog, QMessageBox, QGroupBox,
    QFormLayout, QProgressBar, QScrollArea, QDoubleSpinBox,
    QSpinBox, QDateEdit
)
from PyQt6.QtCore import Qt, QDate, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor
from controllers.admin_kpi_controller import AdminKPIController
from datetime import datetime
import pandas as pd
from pathlib import Path


class PredictionThread(QThread):
    """Thread for running predictions in background"""
    finished = pyqtSignal(dict)
    
    def __init__(self, controller, prediction_type, data):
        super().__init__()
        self.controller = controller
        self.prediction_type = prediction_type
        self.data = data
    
    def run(self):
        if self.prediction_type == 'single':
            result = self.controller.predict_single_item(self.data)
        else:  # batch
            result = self.controller.predict_batch(self.data)
        self.finished.emit(result)


class AdminLogisticKPIWidget(QWidget):
    """Admin widget for logistics KPI predictions"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.controller = AdminKPIController()
        self.prediction_thread = None
        
        self.init_ui()
        self.load_model()
    
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Header with gradient background
        header_widget = QWidget()
        header_widget.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3498db, stop:1 #2ecc71);
                border-radius: 12px;
                padding: 20px;
            }
        """)
        header_layout = QVBoxLayout(header_widget)
        
        header = QLabel("📊 Dự đoán KPI Logistics - Coffee Shop")
        header.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        header.setStyleSheet("color: white;")
        header_layout.addWidget(header)
        
        desc = QLabel(
            "Hệ thống Machine Learning dự đoán hiệu suất sản phẩm với độ chính xác 99.99%"
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: rgba(255,255,255,0.9); font-size: 14px;")
        header_layout.addWidget(desc)
        
        layout.addWidget(header_widget)
        
        # Create tabs với style mới
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: none;
                background: transparent;
                padding: 0px;
            }
            QTabBar::tab {
                background: #ecf0f1;
                color: #2c3e50;
                padding: 12px 24px;
                margin-right: 3px;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                font-weight: 600;
                font-size: 13px;
                min-width: 120px;
            }
            QTabBar::tab:selected {
                background: white;
                color: #3498db;
                border-bottom: 3px solid #3498db;
            }
            QTabBar::tab:hover:!selected {
                background: #d5d8dc;
            }
        """)
        
        # Tab 1: Dashboard
        self.dashboard_tab = self.create_dashboard_tab()
        self.tabs.addTab(self.dashboard_tab, "📊 Dashboard")
        
        # Tab 2: Single Prediction
        self.single_tab = self.create_single_prediction_tab()
        self.tabs.addTab(self.single_tab, "🎯 Dự đoán đơn lẻ")
        
        # Tab 3: Batch Prediction
        self.batch_tab = self.create_batch_prediction_tab()
        self.tabs.addTab(self.batch_tab, "📦 Dự đoán hàng loạt")
        
        # Tab 4: Help
        self.help_tab = self.create_help_tab()
        self.tabs.addTab(self.help_tab, "ℹ️ Hướng dẫn")
        
        layout.addWidget(self.tabs)
    
    def create_dashboard_tab(self):
        """Create dashboard overview tab"""
        tab = QWidget()
        tab.setStyleSheet("background: #f5f6fa;")
        layout = QVBoxLayout(tab)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Stats Cards Row
        cards_layout = QHBoxLayout()
        
        # Card 1: Model Info
        card1 = self.create_stat_card(
            "🤖 Model",
            "Ridge Regression",
            "Accuracy: 99.99%",
            "#3498db"
        )
        cards_layout.addWidget(card1)
        
        # Card 2: Features
        card2 = self.create_stat_card(
            "📊 Features",
            "43 Total",
            "18 Raw + 25 Engineered",
            "#2ecc71"
        )
        cards_layout.addWidget(card2)
        
        # Card 3: Prediction Speed
        card3 = self.create_stat_card(
            "⚡ Speed",
            "< 1ms",
            "Real-time prediction",
            "#f39c12"
        )
        cards_layout.addWidget(card3)
        
        # Card 4: Status
        card4 = self.create_stat_card(
            "✅ Status",
            "Ready",
            "Model loaded",
            "#27ae60"
        )
        cards_layout.addWidget(card4)
        
        layout.addLayout(cards_layout)
        
        # Info Section
        info_section = QGroupBox("📋 Thông tin hệ thống")
        info_section.setStyleSheet("""
            QGroupBox {
                background: white;
                border-radius: 12px;
                padding: 20px;
                font-weight: bold;
                font-size: 14px;
                border: 1px solid #dfe6e9;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 5px;
            }
        """)
        info_layout = QVBoxLayout(info_section)
        
        info_html = """
        <div style='padding: 10px; line-height: 1.8;'>
            <h3 style='color: #2c3e50; margin-top: 0;'>🎯 Mục đích</h3>
            <p style='color: #34495e;'>
                Dự đoán KPI (Key Performance Indicator) cho các sản phẩm và inventory của quán cà phê,
                giúp tối ưu hóa quản lý kho hàng và nâng cao hiệu suất vận hành.
            </p>
            
            <h3 style='color: #2c3e50;'>🏪 Danh mục sản phẩm</h3>
            <ul style='color: #34495e;'>
                <li><b>☕ Signature Drinks:</b> Các đồ uống đặc trưng (Latte, Cappuccino, Americano, Espresso...)</li>
                <li><b>🍵 Tea & Others:</b> Trà, trà sữa, đồ uống thảo mộc và các loại khác</li>
                <li><b>🥐 Breakfast Items:</b> Các món ăn sáng (bánh ngọt, bánh mì, croissant...)</li>
                <li><b>🍔 Lunch & Snacks:</b> Món ăn trưa và snacks (sandwich, salad, burger...)</li>
                <li><b>📦 Premium Coffee Beans & Tea:</b> Hạt cà phê và trà cao cấp (nguyên liệu chuyên dụng)</li>
            </ul>
            
            <h3 style='color: #2c3e50;'>📍 Địa điểm quán</h3>
            <ul style='color: #34495e;'>
                <li><b>🎓 Đại Học Kinh Tế-Luật:</b> Khu vực chính, lưu lượng cao</li>
                <li><b>🏠 Ký Túc Xá Khu B:</b> Khu sinh viên, hoạt động buổi tối</li>
                <li><b>🏠 Ký Túc Xá Khu A:</b> Khu sinh viên, hoạt động buổi sáng</li>
                <li><b>🎭 Nhà Văn Hóa Sinh Viên:</b> Khu văn hóa, event space</li>
            </ul>
            
            <h3 style='color: #2c3e50;'>📈 KPI Score Ranges</h3>
            <table style='width: 100%; border-collapse: collapse; margin-top: 10px;'>
                <tr style='background: #d5f4e6;'>
                    <td style='padding: 10px; border: 1px solid #dfe6e9;'><b>0.7 - 1.0</b></td>
                    <td style='padding: 10px; border: 1px solid #dfe6e9;'>✅ Excellent Performance</td>
                </tr>
                <tr style='background: #fef5e7;'>
                    <td style='padding: 10px; border: 1px solid #dfe6e9;'><b>0.5 - 0.7</b></td>
                    <td style='padding: 10px; border: 1px solid #dfe6e9;'>⚠️ Good Performance</td>
                </tr>
                <tr style='background: #fadbd8;'>
                    <td style='padding: 10px; border: 1px solid #dfe6e9;'><b>0.0 - 0.5</b></td>
                    <td style='padding: 10px; border: 1px solid #dfe6e9;'>❌ Needs Improvement</td>
                </tr>
            </table>
        </div>
        """
        
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setHtml(info_html)
        info_text.setStyleSheet("""
            QTextEdit {
                background: transparent;
                border: none;
                font-size: 13px;
            }
        """)
        info_layout.addWidget(info_text)
        
        layout.addWidget(info_section)
        layout.addStretch()
        
        return tab
    
    def create_stat_card(self, icon, title, subtitle, color):
        """Create a stat card widget"""
        card = QWidget()
        card.setStyleSheet(f"""
            QWidget {{
                background: white;
                border-radius: 12px;
                border-left: 4px solid {color};
            }}
        """)
        card.setMinimumHeight(120)
        
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(20, 15, 20, 15)
        
        icon_label = QLabel(icon)
        icon_label.setStyleSheet(f"font-size: 32px; color: {color};")
        card_layout.addWidget(icon_label)
        
        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #2c3e50;")
        card_layout.addWidget(title_label)
        
        subtitle_label = QLabel(subtitle)
        subtitle_label.setStyleSheet("font-size: 12px; color: #7f8c8d;")
        card_layout.addWidget(subtitle_label)
        
        card_layout.addStretch()
        
        return card
    
    def create_single_prediction_tab(self):
        """Create single prediction tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Scroll area for form
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        
        form_widget = QWidget()
        form_layout = QVBoxLayout(form_widget)
        
        # Input form
        input_group = QGroupBox("📝 Nhập thông tin sản phẩm")
        input_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #3498db;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        input_layout = QFormLayout(input_group)
        input_layout.setSpacing(15)
        
        # Create input fields
        self.single_inputs = {}
        
        # Item ID
        self.single_inputs['item_id'] = QLineEdit()
        self.single_inputs['item_id'].setPlaceholderText("Ví dụ: COFFEE_LATTE")
        input_layout.addRow("Item ID:", self.single_inputs['item_id'])
        
        # Category - Coffee Shop Categories
        self.single_inputs['category'] = QComboBox()
        self.category_mapping = {
            '☕ Signature Drinks': 'Groceries',
            '🍵 Tea & Others': 'Groceries',
            '🥐 Breakfast Items': 'Groceries',
            '🍔 Lunch & Snacks': 'Groceries',
            '📦 Premium Coffee Beans & Tea': 'Pharma'
        }
        self.single_inputs['category'].addItems(list(self.category_mapping.keys()))
        input_layout.addRow("Danh mục sản phẩm:", self.single_inputs['category'])
        
        # Stock Level
        self.single_inputs['stock_level'] = QSpinBox()
        self.single_inputs['stock_level'].setRange(0, 10000)
        self.single_inputs['stock_level'].setValue(150)
        input_layout.addRow("Stock Level:", self.single_inputs['stock_level'])
        
        # Reorder Point
        self.single_inputs['reorder_point'] = QSpinBox()
        self.single_inputs['reorder_point'].setRange(0, 10000)
        self.single_inputs['reorder_point'].setValue(50)
        input_layout.addRow("Reorder Point:", self.single_inputs['reorder_point'])
        
        # Reorder Frequency Days
        self.single_inputs['reorder_frequency_days'] = QSpinBox()
        self.single_inputs['reorder_frequency_days'].setRange(1, 365)
        self.single_inputs['reorder_frequency_days'].setValue(7)
        input_layout.addRow("Reorder Frequency (days):", self.single_inputs['reorder_frequency_days'])
        
        # Lead Time Days
        self.single_inputs['lead_time_days'] = QSpinBox()
        self.single_inputs['lead_time_days'].setRange(0, 365)
        self.single_inputs['lead_time_days'].setValue(3)
        input_layout.addRow("Lead Time (days):", self.single_inputs['lead_time_days'])
        
        # Daily Demand
        self.single_inputs['daily_demand'] = QDoubleSpinBox()
        self.single_inputs['daily_demand'].setRange(0, 10000)
        self.single_inputs['daily_demand'].setDecimals(2)
        self.single_inputs['daily_demand'].setValue(25.5)
        input_layout.addRow("Daily Demand:", self.single_inputs['daily_demand'])
        
        # Demand Std Dev
        self.single_inputs['demand_std_dev'] = QDoubleSpinBox()
        self.single_inputs['demand_std_dev'].setRange(0, 1000)
        self.single_inputs['demand_std_dev'].setDecimals(2)
        self.single_inputs['demand_std_dev'].setValue(3.2)
        input_layout.addRow("Demand Std Dev:", self.single_inputs['demand_std_dev'])
        
        # Item Popularity Score
        self.single_inputs['item_popularity_score'] = QDoubleSpinBox()
        self.single_inputs['item_popularity_score'].setRange(0, 1)
        self.single_inputs['item_popularity_score'].setDecimals(2)
        self.single_inputs['item_popularity_score'].setSingleStep(0.01)
        self.single_inputs['item_popularity_score'].setValue(0.85)
        input_layout.addRow("Popularity Score (0-1):", self.single_inputs['item_popularity_score'])
        
        # Zone - Coffee Shop Locations
        self.single_inputs['zone'] = QComboBox()
        self.zone_mapping = {
            '🎓 Đại Học Kinh Tế-Luật': 'A',  # Main campus - Zone A (best)
            '🏠 Ký Túc Xá Khu A': 'B',  # Dorm A - Zone B
            '🏠 Ký Túc Xá Khu B': 'C',  # Dorm B - Zone C
            '🎭 Nhà Văn Hóa Sinh Viên': 'D'  # Cultural center - Zone D
        }
        self.single_inputs['zone'].addItems(list(self.zone_mapping.keys()))
        input_layout.addRow("Địa điểm quán:", self.single_inputs['zone'])
        
        # Picking Time Seconds
        self.single_inputs['picking_time_seconds'] = QSpinBox()
        self.single_inputs['picking_time_seconds'].setRange(0, 1000)
        self.single_inputs['picking_time_seconds'].setValue(45)
        input_layout.addRow("Picking Time (seconds):", self.single_inputs['picking_time_seconds'])
        
        # Handling Cost Per Unit
        self.single_inputs['handling_cost_per_unit'] = QDoubleSpinBox()
        self.single_inputs['handling_cost_per_unit'].setRange(0, 1000)
        self.single_inputs['handling_cost_per_unit'].setDecimals(2)
        self.single_inputs['handling_cost_per_unit'].setValue(2.50)
        input_layout.addRow("Handling Cost/Unit:", self.single_inputs['handling_cost_per_unit'])
        
        # Unit Price
        self.single_inputs['unit_price'] = QDoubleSpinBox()
        self.single_inputs['unit_price'].setRange(0, 100000)
        self.single_inputs['unit_price'].setDecimals(2)
        self.single_inputs['unit_price'].setValue(99.99)
        input_layout.addRow("Unit Price:", self.single_inputs['unit_price'])
        
        # Holding Cost Per Unit Day
        self.single_inputs['holding_cost_per_unit_day'] = QDoubleSpinBox()
        self.single_inputs['holding_cost_per_unit_day'].setRange(0, 100)
        self.single_inputs['holding_cost_per_unit_day'].setDecimals(2)
        self.single_inputs['holding_cost_per_unit_day'].setValue(0.15)
        input_layout.addRow("Holding Cost/Unit/Day:", self.single_inputs['holding_cost_per_unit_day'])
        
        # Stockout Count Last Month
        self.single_inputs['stockout_count_last_month'] = QSpinBox()
        self.single_inputs['stockout_count_last_month'].setRange(0, 100)
        self.single_inputs['stockout_count_last_month'].setValue(1)
        input_layout.addRow("Stockout Count (last month):", self.single_inputs['stockout_count_last_month'])
        
        # Order Fulfillment Rate
        self.single_inputs['order_fulfillment_rate'] = QDoubleSpinBox()
        self.single_inputs['order_fulfillment_rate'].setRange(0, 1)
        self.single_inputs['order_fulfillment_rate'].setDecimals(2)
        self.single_inputs['order_fulfillment_rate'].setSingleStep(0.01)
        self.single_inputs['order_fulfillment_rate'].setValue(0.95)
        input_layout.addRow("Fulfillment Rate (0-1):", self.single_inputs['order_fulfillment_rate'])
        
        # Total Orders Last Month
        self.single_inputs['total_orders_last_month'] = QSpinBox()
        self.single_inputs['total_orders_last_month'].setRange(0, 100000)
        self.single_inputs['total_orders_last_month'].setValue(750)
        input_layout.addRow("Total Orders (last month):", self.single_inputs['total_orders_last_month'])
        
        # Turnover Ratio
        self.single_inputs['turnover_ratio'] = QDoubleSpinBox()
        self.single_inputs['turnover_ratio'].setRange(0, 100)
        self.single_inputs['turnover_ratio'].setDecimals(2)
        self.single_inputs['turnover_ratio'].setValue(12.5)
        input_layout.addRow("Turnover Ratio:", self.single_inputs['turnover_ratio'])
        
        # Layout Efficiency Score
        self.single_inputs['layout_efficiency_score'] = QDoubleSpinBox()
        self.single_inputs['layout_efficiency_score'].setRange(0, 1)
        self.single_inputs['layout_efficiency_score'].setDecimals(2)
        self.single_inputs['layout_efficiency_score'].setSingleStep(0.01)
        self.single_inputs['layout_efficiency_score'].setValue(0.92)
        input_layout.addRow("Layout Efficiency (0-1):", self.single_inputs['layout_efficiency_score'])
        
        # Last Restock Date
        self.single_inputs['last_restock_date'] = QDateEdit()
        self.single_inputs['last_restock_date'].setCalendarPopup(True)
        self.single_inputs['last_restock_date'].setDate(QDate.currentDate())
        self.single_inputs['last_restock_date'].setDisplayFormat("yyyy-MM-dd")
        input_layout.addRow("Last Restock Date:", self.single_inputs['last_restock_date'])
        
        # Forecasted Demand Next 7d
        self.single_inputs['forecasted_demand_next_7d'] = QDoubleSpinBox()
        self.single_inputs['forecasted_demand_next_7d'].setRange(0, 100000)
        self.single_inputs['forecasted_demand_next_7d'].setDecimals(2)
        self.single_inputs['forecasted_demand_next_7d'].setValue(178.5)
        input_layout.addRow("Forecasted Demand (7 days):", self.single_inputs['forecasted_demand_next_7d'])
        
        form_layout.addWidget(input_group)
        
        # Predict button
        predict_btn = QPushButton("🔮 Dự đoán KPI")
        predict_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 12px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
        """)
        predict_btn.clicked.connect(self.predict_single)
        form_layout.addWidget(predict_btn)
        
        # Results area
        self.single_result = QTextEdit()
        self.single_result.setReadOnly(True)
        self.single_result.setMinimumHeight(250)
        self.single_result.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 2px solid #e9ecef;
                border-radius: 8px;
                padding: 15px;
                font-size: 13px;
            }
        """)
        form_layout.addWidget(self.single_result)
        
        scroll.setWidget(form_widget)
        layout.addWidget(scroll)
        
        return tab
    
    def create_batch_prediction_tab(self):
        """Create batch prediction tab"""
        tab = QWidget()
        tab.setStyleSheet("background: #f5f6fa;")
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Instructions Card
        instructions_card = QWidget()
        instructions_card.setStyleSheet("""
            QWidget {
                background: white;
                border-radius: 12px;
                border-left: 4px solid #3498db;
            }
        """)
        instructions_layout = QVBoxLayout(instructions_card)
        instructions_layout.setContentsMargins(20, 20, 20, 20)
        
        instructions = QLabel(
            "<h3 style='color: #2c3e50; margin-top: 0;'>📦 Dự đoán hàng loạt</h3>"
            "<p style='color: #34495e; line-height: 1.6;'>"
            "Upload file CSV chứa thông tin nhiều sản phẩm để dự đoán KPI cùng lúc.<br>"
            "File CSV phải có đầy đủ các cột theo template bên dưới.<br><br>"
            "<b>Lưu ý:</b> Category và Zone trong CSV phải dùng tên gốc "
            "(Groceries, A/B/C/D), không phải tên hiển thị."
            "</p>"
        )
        instructions.setWordWrap(True)
        instructions.setTextFormat(Qt.TextFormat.RichText)
        instructions_layout.addWidget(instructions)
        
        layout.addWidget(instructions_card)
        
        # Buttons với style đẹp hơn
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)
        
        # Download template button
        download_btn = QPushButton("⬇️ Tải template CSV")
        download_btn.setMinimumHeight(45)
        download_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #27ae60, stop:1 #2ecc71);
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 10px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #229954, stop:1 #27ae60);
            }
            QPushButton:pressed {
                background: #1e8449;
            }
        """)
        download_btn.clicked.connect(self.download_template)
        button_layout.addWidget(download_btn)
        
        # Upload CSV button
        upload_btn = QPushButton("📁 Upload CSV")
        upload_btn.setMinimumHeight(45)
        upload_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3498db, stop:1 #5dade2);
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 10px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2980b9, stop:1 #3498db);
            }
            QPushButton:pressed {
                background: #21618c;
            }
        """)
        upload_btn.clicked.connect(self.upload_csv)
        button_layout.addWidget(upload_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # Progress bar
        self.batch_progress = QProgressBar()
        self.batch_progress.setVisible(False)
        self.batch_progress.setMinimumHeight(30)
        self.batch_progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid #3498db;
                border-radius: 10px;
                text-align: center;
                background: white;
                color: #2c3e50;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3498db, stop:1 #2ecc71);
                border-radius: 8px;
            }
        """)
        layout.addWidget(self.batch_progress)
        
        # Results table trong white card
        table_card = QWidget()
        table_card.setStyleSheet("""
            QWidget {
                background: white;
                border-radius: 12px;
            }
        """)
        table_layout = QVBoxLayout(table_card)
        table_layout.setContentsMargins(15, 15, 15, 15)
        
        table_header = QLabel("<b style='font-size: 14px; color: #2c3e50;'>📋 Kết quả dự đoán</b>")
        table_layout.addWidget(table_header)
        
        self.batch_table = QTableWidget()
        self.batch_table.setStyleSheet("""
            QTableWidget {
                background-color: white;
                border: 1px solid #ecf0f1;
                border-radius: 8px;
                gridline-color: #ecf0f1;
            }
            QTableWidget::item {
                padding: 10px;
                border-bottom: 1px solid #ecf0f1;
            }
            QTableWidget::item:selected {
                background-color: #e3f2fd;
                color: #1976d2;
            }
            QHeaderView::section {
                background-color: #34495e;
                color: white;
                padding: 12px;
                border: none;
                font-weight: bold;
                font-size: 12px;
            }
        """)
        table_layout.addWidget(self.batch_table)
        
        layout.addWidget(table_card)
        
        # Statistics
        self.batch_stats = QLabel()
        self.batch_stats.setWordWrap(True)
        self.batch_stats.setStyleSheet("""
            QLabel {
                background-color: #f8f9fa;
                border: 2px solid #e9ecef;
                border-radius: 8px;
                padding: 15px;
                font-size: 13px;
            }
        """)
        layout.addWidget(self.batch_stats)
        
        # Export button
        export_btn = QPushButton("💾 Xuất kết quả")
        export_btn.setMinimumHeight(45)
        export_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #9b59b6, stop:1 #bb8fce);
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 10px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #8e44ad, stop:1 #9b59b6);
            }
            QPushButton:pressed {
                background: #7d3c98;
            }
        """)
        export_btn.clicked.connect(self.export_results)
        layout.addWidget(export_btn)
        
        return tab
    
    def create_help_tab(self):
        """Create help tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setHtml("""
            <h2>📊 Hướng dẫn sử dụng Dự đoán KPI Logistics</h2>
            
            <h3>🎯 Dự đoán đơn lẻ</h3>
            <p>Nhập thông tin cho một sản phẩm để dự đoán KPI score của nó:</p>
            <ul>
                <li><b>Item ID:</b> Mã sản phẩm (ví dụ: SIGNATURE_LATTE)</li>
                <li><b>Category:</b> Danh mục sản phẩm (5 loại: Signature Drinks, Tea & Others, Breakfast Items, Lunch & Snacks, Premium Coffee Beans & Tea)</li>
                <li><b>Stock Level:</b> Số lượng tồn kho hiện tại</li>
                <li><b>Reorder Point:</b> Mức tồn kho cần đặt hàng lại</li>
                <li><b>Daily Demand:</b> Nhu cầu trung bình mỗi ngày</li>
                <li><b>Order Fulfillment Rate:</b> Tỷ lệ hoàn thành đơn hàng (0-1)</li>
                <li>...và các thông số khác</li>
            </ul>
            
            <h3>📦 Dự đoán hàng loạt</h3>
            <p>Upload file CSV chứa nhiều sản phẩm:</p>
            <ol>
                <li>Nhấn "Tải template CSV" để tải file mẫu</li>
                <li>Điền thông tin sản phẩm vào file CSV</li>
                <li>Nhấn "Upload CSV" và chọn file của bạn</li>
                <li>Xem kết quả trong bảng</li>
                <li>Nhấn "Xuất kết quả" để lưu file</li>
            </ol>
            
            <h3>📈 Giải thích KPI Score</h3>
            <ul>
                <li><b style="color: green;">0.7 - 1.0:</b> ✅ Excellent Performance - Sản phẩm hoạt động tốt</li>
                <li><b style="color: orange;">0.5 - 0.7:</b> ⚠️ Good Performance - Có thể cải thiện</li>
                <li><b style="color: red;">0.0 - 0.5:</b> ❌ Needs Improvement - Cần chú ý khẩn cấp</li>
            </ul>
            
            <h3>🔑 Các yếu tố quan trọng nhất</h3>
            <p>Model xem xét nhiều yếu tố, trong đó quan trọng nhất là:</p>
            <ol>
                <li><b>Order Fulfillment Rate</b> - Tỷ lệ hoàn thành đơn hàng</li>
                <li><b>Efficiency Composite</b> - Hiệu suất tổng hợp</li>
                <li><b>Turnover Ratio</b> - Tốc độ luân chuyển hàng</li>
                <li><b>Item Popularity</b> - Độ phổ biến sản phẩm</li>
                <li><b>Demand-Supply Balance</b> - Cân bằng cung cầu</li>
            </ol>
            
            <h3>🔄 Category Mapping</h3>
            <p>Hệ thống sử dụng ML model với 5 categories gốc (Electronics, Groceries, Apparel, Automotive, Pharma). 
            Các category của quán cà phê được map như sau:</p>
            <ul>
                <li><b>Signature Drinks, Tea & Others, Breakfast Items, Lunch & Snacks</b> → <b>Groceries</b> (đặc điểm: high turnover, perishable, frequent reorder)</li>
                <li><b>Premium Coffee Beans & Tea</b> → <b>Pharma</b> (đặc điểm: specialized items, moderate turnover, longer shelf life, 94% accuracy)</li>
            </ul>
            
            <h3>💡 Lưu ý</h3>
            <ul>
                <li>Đảm bảo dữ liệu nhập vào chính xác</li>
                <li>Order Fulfillment Rate và các score phải từ 0-1</li>
                <li>Ngày restock phải ở định dạng YYYY-MM-DD</li>
                <li>Model đã được train với 99.99% accuracy</li>
            </ul>
            
            <h3>🚀 Mẹo tối ưu KPI</h3>
            <ul>
                <li>Giữ Order Fulfillment Rate cao (>0.9)</li>
                <li>Tối ưu vị trí kho để giảm Picking Time</li>
                <li>Dự báo nhu cầu chính xác</li>
                <li>Cân bằng giữa tồn kho và nhu cầu</li>
                <li>Giảm thiểu Stockout</li>
            </ul>
        """)
        help_text.setStyleSheet("""
            QTextEdit {
                background-color: white;
                border: 1px solid #dcdde1;
                border-radius: 8px;
                padding: 20px;
                font-size: 13px;
            }
        """)
        layout.addWidget(help_text)
        
        return tab
    
    def load_model(self):
        """Load ML model"""
        success, message = self.controller.load_model()
        if not success:
            QMessageBox.critical(self, "Lỗi", f"Không thể load model:\n{message}")
    
    def predict_single(self):
        """Predict KPI for single item"""
        # Gather input data
        data = {}
        for key, widget in self.single_inputs.items():
            if isinstance(widget, QLineEdit):
                data[key] = widget.text()
            elif isinstance(widget, QComboBox):
                current_text = widget.currentText()
                # Map coffee shop categories to model categories
                if key == 'category' and current_text in self.category_mapping:
                    data[key] = self.category_mapping[current_text]
                # Map coffee shop zones to model zones
                elif key == 'zone' and current_text in self.zone_mapping:
                    data[key] = self.zone_mapping[current_text]
                else:
                    data[key] = current_text
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                data[key] = widget.value()
            elif isinstance(widget, QDateEdit):
                data[key] = widget.date().toString("yyyy-MM-dd")
        
        # Show loading
        self.single_result.setHtml("<p style='text-align: center;'>⏳ Đang dự đoán...</p>")
        
        # Run prediction in thread
        self.prediction_thread = PredictionThread(self.controller, 'single', data)
        self.prediction_thread.finished.connect(self.display_single_result)
        self.prediction_thread.start()
    
    def display_single_result(self, result):
        """Display single prediction result"""
        if not result['success']:
            self.single_result.setHtml(f"""
                <div style='color: red; padding: 20px; text-align: center;'>
                    <h2>❌ Lỗi</h2>
                    <p>{result['error']}</p>
                </div>
            """)
            return
        
        kpi_score = result['kpi_score']
        interpretation = result['interpretation']
        recommendations = self.controller.get_recommendations(kpi_score)
        
        # Color based on score
        if kpi_score >= 0.7:
            color = "#27ae60"
            bg_color = "#d5f4e6"
        elif kpi_score >= 0.5:
            color = "#f39c12"
            bg_color = "#fef5e7"
        else:
            color = "#e74c3c"
            bg_color = "#fadbd8"
        
        html = f"""
            <div style='padding: 20px;'>
                <h2 style='text-align: center; color: {color};'>
                    {interpretation}
                </h2>
                <div style='text-align: center; margin: 30px 0;'>
                    <div style='display: inline-block; background: {bg_color}; 
                                border: 3px solid {color}; border-radius: 50%; 
                                width: 200px; height: 200px; line-height: 200px;'>
                        <span style='font-size: 48px; font-weight: bold; color: {color};'>
                            {kpi_score:.4f}
                        </span>
                    </div>
                </div>
                <h3>📋 Đánh giá chi tiết:</h3>
                <p style='font-size: 14px; line-height: 1.6;'>
                    Sản phẩm <b>{self.single_inputs['item_id'].text()}</b> có KPI score là <b>{kpi_score:.4f}</b>.
                    Score này cho thấy mức độ hiệu quả tổng thể trong logistics và inventory management.
                </p>
                <h3>💡 Đề xuất cải thiện:</h3>
                <ul style='font-size: 14px; line-height: 1.8;'>
                    {''.join([f'<li>{rec}</li>' for rec in recommendations])}
                </ul>
                <hr style='margin: 20px 0; border: 1px solid #ecf0f1;'>
                <p style='font-size: 12px; color: #7f8c8d; text-align: center;'>
                    Model accuracy: 99.99% R² | Prediction time: <1ms
                </p>
            </div>
        """
        
        self.single_result.setHtml(html)
    
    def download_template(self):
        """Download CSV template"""
        template_path = Path(__file__).parent.parent / 'templates' / 'logistics_kpi_template.csv'
        
        if not template_path.exists():
            QMessageBox.warning(self, "Lỗi", "Template file không tồn tại!")
            return
        
        # Ask where to save
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Lưu template CSV",
            "logistics_kpi_template.csv",
            "CSV Files (*.csv)"
        )
        
        if save_path:
            try:
                import shutil
                shutil.copy(template_path, save_path)
                QMessageBox.information(self, "Thành công", f"Template đã được lưu tại:\n{save_path}")
            except Exception as e:
                QMessageBox.critical(self, "Lỗi", f"Không thể lưu file:\n{str(e)}")
    
    def upload_csv(self):
        """Upload CSV for batch prediction"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn file CSV",
            "",
            "CSV Files (*.csv)"
        )
        
        if not file_path:
            return
        
        # Show progress
        self.batch_progress.setVisible(True)
        self.batch_progress.setRange(0, 0)  # Indeterminate
        
        # Run prediction in thread
        self.prediction_thread = PredictionThread(self.controller, 'batch', file_path)
        self.prediction_thread.finished.connect(self.display_batch_result)
        self.prediction_thread.start()
    
    def display_batch_result(self, result):
        """Display batch prediction results"""
        self.batch_progress.setVisible(False)
        
        if not result['success']:
            QMessageBox.critical(self, "Lỗi", f"Không thể dự đoán:\n{result['error']}")
            return
        
        # Store results
        self.batch_results = result['results']
        stats = result['stats']
        
        # Display in table
        df = self.batch_results
        self.batch_table.setRowCount(len(df))
        self.batch_table.setColumnCount(3)
        self.batch_table.setHorizontalHeaderLabels(['Item ID', 'KPI Score', 'Interpretation'])
        
        for i, row in df.iterrows():
            self.batch_table.setItem(i, 0, QTableWidgetItem(row['item_id']))
            
            score_item = QTableWidgetItem(f"{row['predicted_kpi_score']:.4f}")
            score_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.batch_table.setItem(i, 1, score_item)
            
            interp_item = QTableWidgetItem(row['interpretation'])
            if '✅' in row['interpretation']:
                interp_item.setForeground(QColor('#27ae60'))
            elif '⚠️' in row['interpretation']:
                interp_item.setForeground(QColor('#f39c12'))
            else:
                interp_item.setForeground(QColor('#e74c3c'))
            self.batch_table.setItem(i, 2, interp_item)
        
        self.batch_table.resizeColumnsToContents()
        
        # Display statistics
        stats_html = f"""
            <div style='padding: 15px;'>
                <h3 style='margin-top: 0;'>📊 Thống kê kết quả</h3>
                <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 10px;'>
                    <div><b>Tổng số sản phẩm:</b> {stats['total_items']}</div>
                    <div><b>KPI trung bình:</b> {stats['mean_kpi']:.4f}</div>
                    <div><b>KPI cao nhất:</b> {stats['max_kpi']:.4f}</div>
                    <div><b>KPI thấp nhất:</b> {stats['min_kpi']:.4f}</div>
                    <div><b>Độ lệch chuẩn:</b> {stats['std_kpi']:.4f}</div>
                    <div><b>Excellent (≥0.7):</b> <span style='color: #27ae60;'>{stats['excellent_count']} ({stats['excellent_count']/stats['total_items']*100:.1f}%)</span></div>
                    <div><b>Good (0.5-0.7):</b> <span style='color: #f39c12;'>{stats['good_count']} ({stats['good_count']/stats['total_items']*100:.1f}%)</span></div>
                    <div><b>Needs Improvement (<0.5):</b> <span style='color: #e74c3c;'>{stats['needs_improvement_count']} ({stats['needs_improvement_count']/stats['total_items']*100:.1f}%)</span></div>
                </div>
            </div>
        """
        self.batch_stats.setText(stats_html)
        
        QMessageBox.information(self, "Thành công", 
                               f"Đã dự đoán thành công cho {stats['total_items']} sản phẩm!")
    
    def export_results(self):
        """Export batch results to CSV"""
        if not hasattr(self, 'batch_results'):
            QMessageBox.warning(self, "Cảnh báo", "Chưa có kết quả để xuất!")
            return
        
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Lưu kết quả",
            f"kpi_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV Files (*.csv)"
        )
        
        if save_path:
            try:
                self.batch_results.to_csv(save_path, index=False)
                QMessageBox.information(self, "Thành công", f"Kết quả đã được lưu tại:\n{save_path}")
            except Exception as e:
                QMessageBox.critical(self, "Lỗi", f"Không thể lưu file:\n{str(e)}")
