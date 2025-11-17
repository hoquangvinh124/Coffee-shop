"""
Admin ML Analytics Widget - Extended Logic
Display ML forecasting charts with day/week/month filters
"""
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                              QComboBox, QLabel, QMessageBox, QDateEdit, QGroupBox)
from PyQt6.QtCore import Qt, QDate, QThread, pyqtSignal
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import requests
import json


class ForecastWorker(QThread):
    """Worker thread for fetching forecast data"""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, api_url, start_date, end_date):
        super().__init__()
        self.api_url = api_url
        self.start_date = start_date
        self.end_date = end_date

    def run(self):
        """Fetch forecast data from API"""
        try:
            url = f"{self.api_url}/forecast/range"
            params = {
                'start_date': self.start_date,
                'end_date': self.end_date
            }
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            self.finished.emit(data)
        except Exception as e:
            self.error.emit(str(e))


class MLAnalyticsChart(FigureCanvas):
    """Matplotlib chart widget"""

    def __init__(self, parent=None, width=8, height=5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

        # Configure chart style
        self.fig.patch.set_facecolor('#f5f5f5')
        self.axes.set_facecolor('#ffffff')
        self.axes.grid(True, alpha=0.3, linestyle='--')

    def plot_forecast(self, data):
        """Plot forecast data"""
        self.axes.clear()

        if not data or 'forecasts' not in data:
            self.axes.text(0.5, 0.5, 'Không có dữ liệu',
                          ha='center', va='center', fontsize=14)
            self.draw()
            return

        forecasts = data['forecasts']
        dates = [f['date'] for f in forecasts]
        values = [f['forecast'] for f in forecasts]
        lower_bounds = [f['lower_bound'] for f in forecasts]
        upper_bounds = [f['upper_bound'] for f in forecasts]

        # Plot main forecast line
        self.axes.plot(dates, values, 'b-', linewidth=2, label='Dự đoán')

        # Plot confidence interval
        self.axes.fill_between(range(len(dates)), lower_bounds, upper_bounds,
                               alpha=0.2, color='blue', label='Khoảng tin cậy')

        # Configure axes
        self.axes.set_xlabel('Ngày', fontsize=10, fontweight='bold')
        self.axes.set_ylabel('Doanh thu (VNĐ)', fontsize=10, fontweight='bold')
        self.axes.set_title('Dự Đoán Doanh Thu Cửa Hàng', fontsize=12, fontweight='bold')

        # Rotate x-axis labels for better readability
        if len(dates) > 10:
            step = len(dates) // 10
            self.axes.set_xticks(range(0, len(dates), step))
            self.axes.set_xticklabels([dates[i] for i in range(0, len(dates), step)],
                                     rotation=45, ha='right')
        else:
            self.axes.set_xticks(range(len(dates)))
            self.axes.set_xticklabels(dates, rotation=45, ha='right')

        # Add legend
        self.axes.legend(loc='upper left')

        # Format y-axis as currency
        from matplotlib.ticker import FuncFormatter
        def currency_formatter(x, p):
            return f'{int(x):,}'
        self.axes.yaxis.set_major_formatter(FuncFormatter(currency_formatter))

        # Adjust layout
        self.fig.tight_layout()
        self.draw()


class AdminMLAnalyticsWidget(QWidget):
    """Admin ML Analytics widget"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.api_url = "http://localhost:8000"
        self.worker = None
        self.setup_ui()

    def setup_ui(self):
        """Setup the user interface"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # Title
        title_label = QLabel("📊 ML Analytics - Dự Đoán Doanh Thu")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
                padding: 10px;
            }
        """)
        main_layout.addWidget(title_label)

        # Control panel
        control_group = QGroupBox("Cài Đặt Dự Đoán")
        control_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                border: 2px solid #c7a17a;
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
        control_layout = QHBoxLayout(control_group)

        # Period selector
        period_label = QLabel("Khoảng thời gian:")
        period_label.setStyleSheet("font-weight: bold;")
        control_layout.addWidget(period_label)

        self.period_combo = QComboBox()
        self.period_combo.addItems(["7 Ngày", "14 Ngày", "30 Ngày", "90 Ngày", "Tùy Chỉnh"])
        self.period_combo.setStyleSheet("""
            QComboBox {
                padding: 8px;
                border: 2px solid #c7a17a;
                border-radius: 4px;
                font-size: 13px;
                min-width: 120px;
            }
            QComboBox:hover {
                border-color: #a0826d;
            }
        """)
        self.period_combo.currentIndexChanged.connect(self.on_period_changed)
        control_layout.addWidget(self.period_combo)

        # Date range (hidden by default)
        date_label = QLabel("Từ ngày:")
        date_label.setStyleSheet("font-weight: bold;")
        self.date_label = date_label
        self.date_label.hide()
        control_layout.addWidget(date_label)

        self.start_date_edit = QDateEdit()
        self.start_date_edit.setCalendarPopup(True)
        self.start_date_edit.setDate(QDate.currentDate())
        self.start_date_edit.setStyleSheet("""
            QDateEdit {
                padding: 8px;
                border: 2px solid #c7a17a;
                border-radius: 4px;
                font-size: 13px;
            }
        """)
        self.start_date_edit.hide()
        control_layout.addWidget(self.start_date_edit)

        to_label = QLabel("Đến ngày:")
        to_label.setStyleSheet("font-weight: bold;")
        self.to_label = to_label
        self.to_label.hide()
        control_layout.addWidget(to_label)

        self.end_date_edit = QDateEdit()
        self.end_date_edit.setCalendarPopup(True)
        self.end_date_edit.setDate(QDate.currentDate().addDays(7))
        self.end_date_edit.setStyleSheet("""
            QDateEdit {
                padding: 8px;
                border: 2px solid #c7a17a;
                border-radius: 4px;
                font-size: 13px;
            }
        """)
        self.end_date_edit.hide()
        control_layout.addWidget(self.end_date_edit)

        # Fetch button
        self.fetch_button = QPushButton("🔄 Tải Dữ Liệu")
        self.fetch_button.setStyleSheet("""
            QPushButton {
                background-color: #c7a17a;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #a0826d;
            }
            QPushButton:pressed {
                background-color: #8b6f5a;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.fetch_button.clicked.connect(self.fetch_forecast)
        control_layout.addWidget(self.fetch_button)

        control_layout.addStretch()
        main_layout.addWidget(control_group)

        # Status label
        self.status_label = QLabel("Sẵn sàng")
        self.status_label.setStyleSheet("""
            QLabel {
                padding: 8px;
                background-color: #e8f5e9;
                border-left: 4px solid #4caf50;
                border-radius: 4px;
                font-size: 13px;
            }
        """)
        main_layout.addWidget(self.status_label)

        # Chart
        self.chart = MLAnalyticsChart(self)
        main_layout.addWidget(self.chart)

        # Summary panel
        summary_group = QGroupBox("Thống Kê Tóm Tắt")
        summary_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                border: 2px solid #c7a17a;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
        """)
        summary_layout = QHBoxLayout(summary_group)

        self.avg_label = QLabel("Trung bình: --")
        self.total_label = QLabel("Tổng: --")
        self.min_label = QLabel("Thấp nhất: --")
        self.max_label = QLabel("Cao nhất: --")

        for label in [self.avg_label, self.total_label, self.min_label, self.max_label]:
            label.setStyleSheet("""
                QLabel {
                    padding: 10px;
                    background-color: #f5f5f5;
                    border-radius: 4px;
                    font-size: 13px;
                }
            """)
            summary_layout.addWidget(label)

        main_layout.addWidget(summary_group)

        # Load initial data
        self.fetch_forecast()

    def on_period_changed(self, index):
        """Handle period selection change"""
        if index == 4:  # Custom
            self.date_label.show()
            self.start_date_edit.show()
            self.to_label.show()
            self.end_date_edit.show()
        else:
            self.date_label.hide()
            self.start_date_edit.hide()
            self.to_label.hide()
            self.end_date_edit.hide()

    def fetch_forecast(self):
        """Fetch forecast data from API"""
        # Disable button during fetch
        self.fetch_button.setEnabled(False)
        self.status_label.setText("⏳ Đang tải dữ liệu...")
        self.status_label.setStyleSheet("""
            QLabel {
                padding: 8px;
                background-color: #fff3e0;
                border-left: 4px solid #ff9800;
                border-radius: 4px;
                font-size: 13px;
            }
        """)

        # Calculate date range
        period_index = self.period_combo.currentIndex()
        today = datetime.now()

        if period_index == 4:  # Custom
            start_date = self.start_date_edit.date().toPyDate()
            end_date = self.end_date_edit.date().toPyDate()
        else:
            days_map = {0: 7, 1: 14, 2: 30, 3: 90}
            days = days_map[period_index]
            start_date = today.date()
            end_date = (today + timedelta(days=days)).date()

        # Format dates
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        # Create worker thread
        self.worker = ForecastWorker(self.api_url, start_str, end_str)
        self.worker.finished.connect(self.on_forecast_loaded)
        self.worker.error.connect(self.on_forecast_error)
        self.worker.start()

    def on_forecast_loaded(self, data):
        """Handle successful forecast data load"""
        self.fetch_button.setEnabled(True)
        self.status_label.setText("✅ Dữ liệu đã tải thành công")
        self.status_label.setStyleSheet("""
            QLabel {
                padding: 8px;
                background-color: #e8f5e9;
                border-left: 4px solid #4caf50;
                border-radius: 4px;
                font-size: 13px;
            }
        """)

        # Update chart
        self.chart.plot_forecast(data)

        # Update summary
        if 'summary' in data:
            summary = data['summary']
            self.avg_label.setText(f"Trung bình: {summary['avg_daily_forecast']:,.0f} VNĐ")
            self.total_label.setText(f"Tổng: {summary['total_forecast']:,.0f} VNĐ")
            self.min_label.setText(f"Thấp nhất: {summary['min_forecast']:,.0f} VNĐ")
            self.max_label.setText(f"Cao nhất: {summary['max_forecast']:,.0f} VNĐ")

    def on_forecast_error(self, error_msg):
        """Handle forecast data load error"""
        self.fetch_button.setEnabled(True)
        self.status_label.setText(f"❌ Lỗi: {error_msg}")
        self.status_label.setStyleSheet("""
            QLabel {
                padding: 8px;
                background-color: #ffebee;
                border-left: 4px solid #f44336;
                border-radius: 4px;
                font-size: 13px;
            }
        """)

        QMessageBox.warning(
            self,
            "Lỗi Tải Dữ Liệu",
            f"Không thể tải dữ liệu dự đoán:\n{error_msg}\n\n"
            "Vui lòng kiểm tra:\n"
            "- API server đang chạy (http://localhost:8000)\n"
            "- Model đã được load thành công"
        )
