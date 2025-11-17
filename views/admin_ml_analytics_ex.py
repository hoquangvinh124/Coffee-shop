"""
Admin ML Analytics Widget - Extended Logic
Display ML forecasting charts with store comparison and optimization insights
"""
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                              QComboBox, QLabel, QMessageBox, QDateEdit, QGroupBox,
                              QScrollArea, QFrame, QGridLayout, QCheckBox)
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

    def __init__(self, api_url, endpoint, params=None):
        super().__init__()
        self.api_url = api_url
        self.endpoint = endpoint
        self.params = params or {}

    def run(self):
        """Fetch forecast data from API"""
        try:
            url = f"{self.api_url}{self.endpoint}"
            response = requests.get(url, params=self.params, timeout=30)
            response.raise_for_status()
            data = response.json()
            self.finished.emit(data)
        except Exception as e:
            self.error.emit(str(e))


class StoreListWorker(QThread):
    """Worker thread for fetching store list"""
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, api_url):
        super().__init__()
        self.api_url = api_url

    def run(self):
        """Fetch store list from API"""
        try:
            url = f"{self.api_url}/stores/top/10"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            self.finished.emit(data['stores'])
        except Exception as e:
            self.error.emit(str(e))


class MLAnalyticsChart(FigureCanvas):
    """Matplotlib chart widget with fixed height"""

    def __init__(self, parent=None, width=10, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

        # Set fixed height to prevent overflow
        self.setMinimumHeight(400)
        self.setMaximumHeight(400)

        # Configure chart style
        self.fig.patch.set_facecolor('#ffffff')
        self.axes.set_facecolor('#ffffff')
        self.axes.grid(True, alpha=0.3, linestyle='--')

    def plot_forecast(self, forecasts_data):
        """Plot forecast data for one or multiple stores"""
        self.axes.clear()

        if not forecasts_data:
            self.axes.text(0.5, 0.5, 'Không có dữ liệu',
                          ha='center', va='center', fontsize=14)
            self.draw()
            return

        colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']

        for idx, store_data in enumerate(forecasts_data):
            if 'forecasts' not in store_data:
                continue

            store_name = store_data.get('store_name', f"Store {idx+1}")
            forecasts = store_data['forecasts']
            dates = [f['date'] for f in forecasts]
            values = [f['forecast'] for f in forecasts]

            color = colors[idx % len(colors)]
            self.axes.plot(dates, values, '-', linewidth=2,
                          label=store_name, color=color)

        # Configure axes
        self.axes.set_xlabel('Ngày', fontsize=10, fontweight='bold')
        self.axes.set_ylabel('Doanh thu (VNĐ)', fontsize=10, fontweight='bold')
        self.axes.set_title('So Sánh Dự Đoán Doanh Thu Giữa Các Cửa Hàng',
                           fontsize=12, fontweight='bold')

        # Rotate x-axis labels
        dates_list = forecasts_data[0]['forecasts']
        all_dates = [f['date'] for f in dates_list]
        if len(all_dates) > 10:
            step = len(all_dates) // 10
            self.axes.set_xticks(range(0, len(all_dates), step))
            self.axes.set_xticklabels([all_dates[i] for i in range(0, len(all_dates), step)],
                                     rotation=45, ha='right', fontsize=8)
        else:
            self.axes.set_xticks(range(len(all_dates)))
            self.axes.set_xticklabels(all_dates, rotation=45, ha='right', fontsize=8)

        # Add legend
        if len(forecasts_data) > 1:
            self.axes.legend(loc='upper left', fontsize=9)

        # Format y-axis as currency
        from matplotlib.ticker import FuncFormatter
        def currency_formatter(x, p):
            if x >= 1000000:
                return f'{x/1000000:.1f}M'
            elif x >= 1000:
                return f'{x/1000:.0f}K'
            return f'{int(x)}'
        self.axes.yaxis.set_major_formatter(FuncFormatter(currency_formatter))

        # Adjust layout
        self.fig.tight_layout()
        self.draw()


class AdminMLAnalyticsWidget(QWidget):
    """Admin ML Analytics widget with store comparison"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.api_url = "http://localhost:8000"
        self.worker = None
        self.stores_list = []
        self.selected_stores = []
        self.setup_ui()
        self.load_stores()

    def setup_ui(self):
        """Setup the user interface"""
        # Main scroll area to prevent overflow
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # Title
        title_label = QLabel("🤖 ML Analytics - Dự Đoán & So Sánh Cửa Hàng")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 22px;
                font-weight: bold;
                color: #2c3e50;
                padding: 10px;
                background-color: #f8f9fa;
                border-left: 5px solid #c7a17a;
                border-radius: 4px;
            }
        """)
        main_layout.addWidget(title_label)

        # Control panel
        control_group = self.create_control_panel()
        main_layout.addWidget(control_group)

        # Status label
        self.status_label = QLabel("Sẵn sàng")
        self.status_label.setStyleSheet("""
            QLabel {
                padding: 8px;
                background-color: #e8f5e9;
                border-left: 4px solid #4caf50;
                border-radius: 4px;
                font-size: 12px;
            }
        """)
        main_layout.addWidget(self.status_label)

        # Chart - with fixed height
        self.chart = MLAnalyticsChart(self)
        main_layout.addWidget(self.chart)

        # Comparison panel
        comparison_group = self.create_comparison_panel()
        main_layout.addWidget(comparison_group)

        # Insights panel
        insights_group = self.create_insights_panel()
        main_layout.addWidget(insights_group)

        scroll.setWidget(main_widget)

        # Set scroll area as main layout
        wrapper_layout = QVBoxLayout(self)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.addWidget(scroll)

    def create_control_panel(self):
        """Create control panel"""
        control_group = QGroupBox("⚙️ Cài Đặt Phân Tích")
        control_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 13px;
                border: 2px solid #c7a17a;
                border-radius: 8px;
                margin-top: 10px;
                padding: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        control_layout = QGridLayout(control_group)
        control_layout.setSpacing(10)

        # Store selector
        row = 0
        store_label = QLabel("Chọn cửa hàng:")
        store_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        control_layout.addWidget(store_label, row, 0)

        self.store_combo = QComboBox()
        self.store_combo.addItem("Đang tải danh sách...")
        self.store_combo.setStyleSheet("""
            QComboBox {
                padding: 6px;
                border: 2px solid #c7a17a;
                border-radius: 4px;
                font-size: 12px;
                min-width: 200px;
            }
        """)
        control_layout.addWidget(self.store_combo, row, 1)

        # Period selector
        row += 1
        period_label = QLabel("Khoảng thời gian:")
        period_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        control_layout.addWidget(period_label, row, 0)

        self.period_combo = QComboBox()
        self.period_combo.addItems(["7 Ngày", "14 Ngày", "30 Ngày", "90 Ngày"])
        self.period_combo.setCurrentIndex(2)  # Default 30 days
        self.period_combo.setStyleSheet("""
            QComboBox {
                padding: 6px;
                border: 2px solid #c7a17a;
                border-radius: 4px;
                font-size: 12px;
            }
        """)
        control_layout.addWidget(self.period_combo, row, 1)

        # Fetch button
        row += 1
        self.fetch_button = QPushButton("📊 Phân Tích Dự Đoán")
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
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.fetch_button.clicked.connect(self.fetch_forecast)
        control_layout.addWidget(self.fetch_button, row, 0, 1, 2)

        return control_group

    def create_comparison_panel(self):
        """Create comparison statistics panel"""
        comparison_group = QGroupBox("📊 Thống Kê So Sánh")
        comparison_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 13px;
                border: 2px solid #2196F3;
                border-radius: 8px;
                margin-top: 10px;
                padding: 15px;
            }
        """)
        comparison_layout = QHBoxLayout(comparison_group)

        # Create stat cards
        self.avg_label = self.create_stat_card("Trung bình", "--", "#2196F3")
        self.total_label = self.create_stat_card("Tổng dự đoán", "--", "#4CAF50")
        self.min_label = self.create_stat_card("Thấp nhất", "--", "#FF9800")
        self.max_label = self.create_stat_card("Cao nhất", "--", "#9C27B0")

        comparison_layout.addWidget(self.avg_label)
        comparison_layout.addWidget(self.total_label)
        comparison_layout.addWidget(self.min_label)
        comparison_layout.addWidget(self.max_label)

        return comparison_group

    def create_stat_card(self, title, value, color):
        """Create a stat card widget"""
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background-color: white;
                border-left: 4px solid {color};
                border-radius: 6px;
                padding: 10px;
            }}
        """)
        layout = QVBoxLayout(card)
        layout.setSpacing(5)

        title_label = QLabel(title)
        title_label.setStyleSheet(f"""
            QLabel {{
                color: {color};
                font-size: 11px;
                font-weight: bold;
            }}
        """)
        layout.addWidget(title_label)

        value_label = QLabel(value)
        value_label.setObjectName("value")
        value_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #2c3e50;
            }
        """)
        layout.addWidget(value_label)

        return card

    def create_insights_panel(self):
        """Create AI insights and recommendations panel"""
        insights_group = QGroupBox("💡 Gợi Ý Tối Ưu Hoạt Động")
        insights_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 13px;
                border: 2px solid #FF9800;
                border-radius: 8px;
                margin-top: 10px;
                padding: 15px;
            }
        """)
        insights_layout = QVBoxLayout(insights_group)

        self.insights_label = QLabel("Chưa có dữ liệu để phân tích")
        self.insights_label.setWordWrap(True)
        self.insights_label.setStyleSheet("""
            QLabel {
                padding: 15px;
                background-color: #fff3e0;
                border-radius: 6px;
                font-size: 12px;
                line-height: 1.6;
            }
        """)
        insights_layout.addWidget(self.insights_label)

        return insights_group

    def load_stores(self):
        """Load store list from API"""
        self.store_combo.setEnabled(False)
        worker = StoreListWorker(self.api_url)
        worker.finished.connect(self.on_stores_loaded)
        worker.error.connect(self.on_stores_error)
        worker.start()
        self.store_list_worker = worker  # Keep reference

    def on_stores_loaded(self, stores):
        """Handle store list loaded"""
        self.stores_list = stores
        self.store_combo.clear()

        for store in stores:
            store_nbr = store['store_nbr']
            city = store['city']
            store_type = store['type']
            growth = store['growth_percent']
            self.store_combo.addItem(
                f"Store #{store_nbr} - {city} (Type {store_type}) - Tăng trưởng: {growth:.1f}%",
                store_nbr
            )

        self.store_combo.setEnabled(True)

    def on_stores_error(self, error_msg):
        """Handle store list load error"""
        self.store_combo.clear()
        self.store_combo.addItem("Lỗi tải danh sách")
        QMessageBox.warning(self, "Lỗi", f"Không thể tải danh sách cửa hàng:\n{error_msg}")

    def fetch_forecast(self):
        """Fetch forecast data"""
        if self.store_combo.currentIndex() < 0:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn cửa hàng!")
            return

        self.fetch_button.setEnabled(False)
        self.status_label.setText("⏳ Đang tải dữ liệu dự đoán...")
        self.status_label.setStyleSheet("""
            QLabel {
                padding: 8px;
                background-color: #fff3e0;
                border-left: 4px solid #ff9800;
                border-radius: 4px;
                font-size: 12px;
            }
        """)

        # Get parameters
        store_nbr = self.store_combo.currentData()
        period_index = self.period_combo.currentIndex()
        days_map = {0: 7, 1: 14, 2: 30, 3: 90}
        days = days_map[period_index]

        # Fetch data
        endpoint = f"/stores/{store_nbr}/forecast"
        params = {'days': days}

        self.worker = ForecastWorker(self.api_url, endpoint, params)
        self.worker.finished.connect(self.on_forecast_loaded)
        self.worker.error.connect(self.on_forecast_error)
        self.worker.start()

    def on_forecast_loaded(self, data):
        """Handle successful forecast load"""
        self.fetch_button.setEnabled(True)
        self.status_label.setText("✅ Dữ liệu đã tải thành công")
        self.status_label.setStyleSheet("""
            QLabel {
                padding: 8px;
                background-color: #e8f5e9;
                border-left: 4px solid #4caf50;
                border-radius: 4px;
                font-size: 12px;
            }
        """)

        # Prepare chart data
        store_name = f"Store #{data['store_nbr']} - {data['city']}"
        chart_data = [{
            'store_name': store_name,
            'forecasts': data['forecasts']
        }]

        # Update chart
        self.chart.plot_forecast(chart_data)

        # Update statistics
        avg_val = data.get('forecast_avg_daily', 0)
        total_val = data.get('total_forecast', 0)

        forecasts = data['forecasts']
        min_val = min(f['forecast'] for f in forecasts)
        max_val = max(f['forecast'] for f in forecasts)

        self.update_stat_card(self.avg_label, f"{avg_val:,.0f} VNĐ")
        self.update_stat_card(self.total_label, f"{total_val:,.0f} VNĐ")
        self.update_stat_card(self.min_label, f"{min_val:,.0f} VNĐ")
        self.update_stat_card(self.max_label, f"{max_val:,.0f} VNĐ")

        # Generate insights
        self.generate_insights(data)

    def update_stat_card(self, card, value):
        """Update stat card value"""
        value_label = card.findChild(QLabel, "value")
        if value_label:
            value_label.setText(value)

    def generate_insights(self, data):
        """Generate AI insights based on forecast data"""
        insights = []

        # Growth analysis
        growth = data.get('growth_percent', 0)
        if growth > 30:
            insights.append(f"🚀 <b>Tăng trưởng mạnh:</b> Cửa hàng có tốc độ tăng trưởng {growth:.1f}%, rất cao so với lịch sử. Nên tăng cường nhân sự và nguồn cung để đáp ứng nhu cầu.")
        elif growth > 10:
            insights.append(f"📈 <b>Tăng trưởng ổn định:</b> Dự đoán tăng {growth:.1f}% so với trung bình lịch sử. Duy trì chiến lược hiện tại và theo dõi xu hướng.")
        elif growth < 0:
            insights.append(f"⚠️ <b>Cảnh báo giảm sút:</b> Doanh thu dự kiến giảm {abs(growth):.1f}%. Cần xem xét lại chiến lược marketing và chất lượng dịch vụ.")
        else:
            insights.append(f"➡️ <b>Ổn định:</b> Tăng trưởng {growth:.1f}%. Cửa hàng hoạt động ổn định.")

        # Revenue analysis
        avg_daily = data.get('forecast_avg_daily', 0)
        historical_avg = data.get('historical_avg_daily', 0)

        if avg_daily > historical_avg * 1.2:
            insights.append(f"💰 <b>Tiềm năng cao:</b> Doanh thu dự đoán {avg_daily:,.0f} VNĐ/ngày, cao hơn 20% so với lịch sử. Đây là thời điểm tốt để mở rộng hoạt động.")

        # Store type recommendation
        store_type = data.get('type', '')
        if store_type == 'A':
            insights.append("🏪 <b>Cửa hàng loại A:</b> Đây là cửa hàng hạng A với doanh thu cao. Ưu tiên đầu tư vào trải nghiệm khách hàng và sản phẩm cao cấp.")
        elif store_type == 'D':
            insights.append("🏪 <b>Cửa hàng loại D:</b> Cửa hàng quy mô nhỏ. Tập trung vào hiệu quả chi phí và dịch vụ nhanh.")

        # Seasonal recommendations
        insights.append("📅 <b>Lời khuyên theo mùa:</b> Dựa trên dự đoán, hãy chuẩn bị kế hoạch marketing và khuyến mãi phù hợp với từng giai đoạn trong chu kỳ dự báo.")

        # Optimization tips
        insights.append("✨ <b>Tối ưu hóa:</b><br>"
                       "• Theo dõi thời điểm cao điểm để điều phối nhân sự<br>"
                       "• Chuẩn bị nguyên liệu dựa trên dự báo để tránh lãng phí<br>"
                       "• So sánh với các cửa hàng khác để học hỏi kinh nghiệm")

        insights_html = "<br><br>".join(insights)
        self.insights_label.setText(insights_html)

    def on_forecast_error(self, error_msg):
        """Handle forecast error"""
        self.fetch_button.setEnabled(True)
        self.status_label.setText(f"❌ Lỗi: {error_msg}")
        self.status_label.setStyleSheet("""
            QLabel {
                padding: 8px;
                background-color: #ffebee;
                border-left: 4px solid #f44336;
                border-radius: 4px;
                font-size: 12px;
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
