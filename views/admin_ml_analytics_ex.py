"""
Admin ML Analytics Widget - Extended Logic
Display comprehensive revenue forecasting with store comparisons
"""
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                              QComboBox, QLabel, QMessageBox, QSpinBox, QGroupBox,
                              QScrollArea, QFrame, QGridLayout, QSizePolicy)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import requests
import json


class DataWorker(QThread):
    """Worker thread for fetching data from API"""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, api_url, endpoint, method="GET", params=None, json_data=None):
        super().__init__()
        self.api_url = api_url
        self.endpoint = endpoint
        self.method = method
        self.params = params or {}
        self.json_data = json_data

    def run(self):
        """Fetch data from API"""
        try:
            url = f"{self.api_url}{self.endpoint}"
            if self.method == "POST":
                response = requests.post(url, params=self.params, json=self.json_data, timeout=30)
            else:
                response = requests.get(url, params=self.params, timeout=30)
            response.raise_for_status()
            data = response.json()
            self.finished.emit(data)
        except Exception as e:
            self.error.emit(str(e))


class CompactChart(FigureCanvas):
    """Compact chart widget"""

    def __init__(self, parent=None, title="", width=5, height=3):
        self.fig = Figure(figsize=(width, height), dpi=80)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.chart_title = title

        # Set size policy
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(250)

        # Configure style
        self.fig.patch.set_facecolor('#ffffff')
        self.axes.set_facecolor('#ffffff')
        self.axes.grid(True, alpha=0.2, linestyle='--')

    def plot_line_forecast(self, data, title=None):
        """Plot line chart for forecast"""
        self.axes.clear()

        if not data or 'forecasts' not in data:
            self.axes.text(0.5, 0.5, 'Không có dữ liệu', ha='center', va='center')
            self.draw()
            return

        forecasts = data['forecasts']
        dates = [f['date'] for f in forecasts]
        values = [f['forecast'] for f in forecasts]

        # Plot
        self.axes.plot(dates, values, 'b-', linewidth=2, marker='o', markersize=3, alpha=0.7)

        # Title
        if title:
            self.axes.set_title(title, fontsize=10, fontweight='bold', pad=10)

        # Labels
        self.axes.set_xlabel('Ngày', fontsize=8)
        self.axes.set_ylabel('Doanh thu', fontsize=8)

        # Format x-axis
        if len(dates) > 10:
            step = len(dates) // 8
            self.axes.set_xticks(range(0, len(dates), step))
            self.axes.set_xticklabels([dates[i] for i in range(0, len(dates), step)],
                                     rotation=45, ha='right', fontsize=7)
        else:
            self.axes.set_xticks(range(len(dates)))
            self.axes.set_xticklabels(dates, rotation=45, ha='right', fontsize=7)

        # Format y-axis
        from matplotlib.ticker import FuncFormatter
        def currency_formatter(x, p):
            if x >= 1000000:
                return f'{x/1000000:.1f}M'
            elif x >= 1000:
                return f'{x/1000:.0f}K'
            return f'{int(x)}'
        self.axes.yaxis.set_major_formatter(FuncFormatter(currency_formatter))
        self.axes.tick_params(axis='y', labelsize=7)

        self.fig.tight_layout()
        self.draw()

    def plot_bar_comparison(self, stores_data, title=""):
        """Plot bar chart for store comparison"""
        self.axes.clear()

        if not stores_data:
            self.axes.text(0.5, 0.5, 'Không có dữ liệu', ha='center', va='center')
            self.draw()
            return

        # Extract data
        store_names = [f"#{s['store_nbr']}" for s in stores_data]
        revenues = [s['revenue'] for s in stores_data]

        # Plot
        bars = self.axes.bar(range(len(store_names)), revenues, color='#4CAF50', alpha=0.7)

        # Highlight
        for bar in bars:
            height = bar.get_height()
            self.axes.text(bar.get_x() + bar.get_width()/2., height,
                          f'{height/1000:.0f}K',
                          ha='center', va='bottom', fontsize=7)

        # Labels
        self.axes.set_title(title, fontsize=10, fontweight='bold', pad=10)
        self.axes.set_xlabel('Cửa hàng', fontsize=8)
        self.axes.set_ylabel('Doanh thu', fontsize=8)
        self.axes.set_xticks(range(len(store_names)))
        self.axes.set_xticklabels(store_names, fontsize=7)

        # Format y-axis
        from matplotlib.ticker import FuncFormatter
        def currency_formatter(x, p):
            if x >= 1000000:
                return f'{x/1000000:.1f}M'
            elif x >= 1000:
                return f'{x/1000:.0f}K'
            return f'{int(x)}'
        self.axes.yaxis.set_major_formatter(FuncFormatter(currency_formatter))
        self.axes.tick_params(axis='y', labelsize=7)

        self.fig.tight_layout()
        self.draw()


class AdminMLAnalyticsWidget(QWidget):
    """Admin ML Analytics widget with comprehensive forecasting"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.api_url = "http://localhost:8000"
        self.stores_list = []
        self.current_data = {}
        self.setup_ui()
        self.load_stores()

    def setup_ui(self):
        """Setup the user interface"""
        # Main scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # Title
        title_label = QLabel("📊 Dự Báo Doanh Thu")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
                padding: 10px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #c7a17a, stop:1 #f5f5f5);
                border-radius: 5px;
            }
        """)
        main_layout.addWidget(title_label)

        # Control panel
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)

        # Stat cards
        stats_group = self.create_stats_panel()
        main_layout.addWidget(stats_group)

        # Main forecasts (side by side)
        forecasts_layout = QHBoxLayout()

        # Overall forecast (left)
        self.overall_group = self.create_forecast_group("🌐 Dự Báo Tổng Thể")
        self.overall_chart = CompactChart(self, "Tổng Thể Cửa Hàng")
        self.overall_group.layout().addWidget(self.overall_chart)
        forecasts_layout.addWidget(self.overall_group)

        # Individual store forecast (right)
        self.store_group = self.create_forecast_group("🏪 Dự Báo Từng Cửa Hàng")
        self.store_chart = CompactChart(self, "Cửa Hàng Cụ Thể")
        self.store_group.layout().addWidget(self.store_chart)
        forecasts_layout.addWidget(self.store_group)

        main_layout.addLayout(forecasts_layout)

        # Comparison charts (side by side)
        comparison_layout = QHBoxLayout()

        # Top performers (left)
        self.top_group = self.create_forecast_group("🏆 Top Cửa Hàng Cao Nhất")
        self.top_chart = CompactChart(self, "Top Performers")
        self.top_group.layout().addWidget(self.top_chart)
        comparison_layout.addWidget(self.top_group)

        # Bottom performers (right)
        self.bottom_group = self.create_forecast_group("⚠️ Top Cửa Hàng Thấp Nhất")
        self.bottom_chart = CompactChart(self, "Bottom Performers")
        self.bottom_group.layout().addWidget(self.bottom_chart)
        comparison_layout.addWidget(self.bottom_group)

        main_layout.addLayout(comparison_layout)

        scroll.setWidget(main_widget)

        wrapper_layout = QVBoxLayout(self)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.addWidget(scroll)

    def create_control_panel(self):
        """Create control panel"""
        group = QGroupBox("⚙️ Cài Đặt Phân Tích")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
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

        layout = QGridLayout(group)
        layout.setSpacing(10)

        row = 0

        # Store selector
        layout.addWidget(QLabel("Chọn cửa hàng:"), row, 0)
        self.store_combo = QComboBox()
        self.store_combo.addItem("Đang tải...")
        self.store_combo.setMinimumWidth(250)
        layout.addWidget(self.store_combo, row, 1)

        # Period
        layout.addWidget(QLabel("Số ngày dự báo:"), row, 2)
        self.period_spin = QSpinBox()
        self.period_spin.setRange(7, 365)
        self.period_spin.setValue(30)
        self.period_spin.setSuffix(" ngày")
        layout.addWidget(self.period_spin, row, 3)

        row += 1

        # Top N
        layout.addWidget(QLabel("Top N cửa hàng:"), row, 0)
        self.topn_spin = QSpinBox()
        self.topn_spin.setRange(3, 20)
        self.topn_spin.setValue(10)
        layout.addWidget(self.topn_spin, row, 1)

        # Analyze button
        self.analyze_btn = QPushButton("📊 Phân Tích Dự Đoán")
        self.analyze_btn.setStyleSheet("""
            QPushButton {
                background-color: #c7a17a;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #a0826d; }
            QPushButton:disabled { background-color: #ccc; }
        """)
        self.analyze_btn.clicked.connect(self.run_analysis)
        layout.addWidget(self.analyze_btn, row, 2, 1, 2)

        return group

    def create_stats_panel(self):
        """Create statistics panel"""
        group = QGroupBox("📈 Thống Kê Tổng Quan")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #2196F3;
                border-radius: 8px;
                margin-top: 10px;
                padding: 10px;
            }
        """)

        layout = QHBoxLayout(group)

        self.stat1 = self.create_stat_card("Tổng TB/ngày", "--", "#2196F3")
        self.stat2 = self.create_stat_card("Tổng dự báo", "--", "#4CAF50")
        self.stat3 = self.create_stat_card("TB cửa hàng", "--", "#FF9800")
        self.stat4 = self.create_stat_card("Tăng trưởng", "--", "#9C27B0")

        layout.addWidget(self.stat1)
        layout.addWidget(self.stat2)
        layout.addWidget(self.stat3)
        layout.addWidget(self.stat4)

        return group

    def create_stat_card(self, title, value, color):
        """Create stat card"""
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background-color: white;
                border-left: 5px solid {color};
                border-radius: 6px;
                padding: 10px;
            }}
        """)
        layout = QVBoxLayout(card)
        layout.setSpacing(3)

        title_label = QLabel(title)
        title_label.setStyleSheet(f"color: {color}; font-size: 10px; font-weight: bold;")
        layout.addWidget(title_label)

        value_label = QLabel(value)
        value_label.setObjectName("value")
        value_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")
        layout.addWidget(value_label)

        return card

    def create_forecast_group(self, title):
        """Create forecast group box"""
        group = QGroupBox(title)
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #ddd;
                border-radius: 8px;
                margin-top: 10px;
                padding: 10px;
                font-size: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        layout = QVBoxLayout(group)
        return group

    def update_stat_card(self, card, value):
        """Update stat card value"""
        value_label = card.findChild(QLabel, "value")
        if value_label:
            value_label.setText(value)

    def load_stores(self):
        """Load all stores"""
        self.store_combo.setEnabled(False)
        worker = DataWorker(self.api_url, "/stores")
        worker.finished.connect(self.on_stores_loaded)
        worker.error.connect(self.on_error)
        worker.start()
        self.stores_worker = worker

    def on_stores_loaded(self, data):
        """Handle stores loaded"""
        self.stores_list = data.get('stores', [])
        self.store_combo.clear()

        for store in self.stores_list:
            store_nbr = store['store_nbr']
            city = store.get('city', 'N/A')
            store_type = store.get('type', 'N/A')
            self.store_combo.addItem(f"Store #{store_nbr} - {city} (Type {store_type})", store_nbr)

        self.store_combo.setEnabled(True)

    def run_analysis(self):
        """Run comprehensive analysis"""
        if self.store_combo.currentIndex() < 0:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn cửa hàng!")
            return

        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setText("⏳ Đang phân tích...")

        # Get parameters
        store_nbr = self.store_combo.currentData()
        days = self.period_spin.value()
        top_n = self.topn_spin.value()

        # Store for comparison
        self.current_params = {
            'store_nbr': store_nbr,
            'days': days,
            'top_n': top_n
        }

        # Fetch overall forecast
        self.fetch_overall_forecast(days)

    def fetch_overall_forecast(self, days):
        """Fetch overall forecast"""
        worker = DataWorker(self.api_url, "/forecast", method="POST", params={'days': days})
        worker.finished.connect(self.on_overall_loaded)
        worker.error.connect(self.on_error)
        worker.start()
        self.overall_worker = worker

    def on_overall_loaded(self, data):
        """Handle overall forecast loaded"""
        self.current_data['overall'] = data

        # Update chart
        self.overall_chart.plot_line_forecast(data, "Dự Báo Tổng Thể Hệ Thống")

        # Fetch store forecast
        store_nbr = self.current_params['store_nbr']
        days = self.current_params['days']
        self.fetch_store_forecast(store_nbr, days)

    def fetch_store_forecast(self, store_nbr, days):
        """Fetch store forecast"""
        worker = DataWorker(self.api_url, f"/stores/{store_nbr}/forecast",
                           method="POST", params={'days': days})
        worker.finished.connect(self.on_store_loaded)
        worker.error.connect(self.on_error)
        worker.start()
        self.store_worker = worker

    def on_store_loaded(self, data):
        """Handle store forecast loaded"""
        self.current_data['store'] = data

        # Update chart
        store_name = f"Store #{data['store_nbr']} - {data['city']}"
        self.store_chart.plot_line_forecast(data, store_name)

        # Update stats
        overall = self.current_data.get('overall', {})
        overall_summary = overall.get('summary', {})

        overall_avg = overall_summary.get('avg_daily_forecast', 0)
        overall_total = overall_summary.get('total_forecast', 0)
        store_avg = data.get('forecast_avg_daily', 0)
        growth = data.get('growth_percent', 0)

        self.update_stat_card(self.stat1, f"{overall_avg:,.0f} VNĐ")
        self.update_stat_card(self.stat2, f"{overall_total:,.0f} VNĐ")
        self.update_stat_card(self.stat3, f"{store_avg:,.0f} VNĐ")
        self.update_stat_card(self.stat4, f"{growth:+.1f}%")

        # Fetch comparison data
        self.fetch_comparison_data()

    def fetch_comparison_data(self):
        """Fetch data for top/bottom comparison"""
        days = self.current_params['days']
        top_n = self.current_params['top_n']

        # We'll fetch all stores and calculate
        self.fetch_all_stores_forecast(days, top_n)

    def fetch_all_stores_forecast(self, days, top_n):
        """Fetch forecasts for all stores to compare"""
        # Simplified: Get top N stores from metadata
        worker = DataWorker(self.api_url, f"/stores/top/{top_n}")
        worker.finished.connect(lambda data: self.on_top_stores_loaded(data, days))
        worker.error.connect(self.on_error)
        worker.start()
        self.comparison_worker = worker

    def on_top_stores_loaded(self, data, days):
        """Handle top stores loaded"""
        stores = data.get('stores', [])

        # Prepare data for charts
        top_stores = stores[:self.topn_spin.value()]
        bottom_stores = sorted(stores, key=lambda x: x['forecast_avg_daily'])[:self.topn_spin.value()]

        # Calculate total revenue for period
        top_data = [{
            'store_nbr': s['store_nbr'],
            'revenue': s['forecast_avg_daily'] * days
        } for s in top_stores]

        bottom_data = [{
            'store_nbr': s['store_nbr'],
            'revenue': s['forecast_avg_daily'] * days
        } for s in bottom_stores]

        # Plot
        self.top_chart.plot_bar_comparison(top_data, f"Top {len(top_data)} Cửa Hàng Cao Nhất")
        self.bottom_chart.plot_bar_comparison(bottom_data, f"Top {len(bottom_data)} Cửa Hàng Thấp Nhất")

        # Re-enable button
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("📊 Phân Tích Dự Đoán")

    def on_error(self, error_msg):
        """Handle error"""
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("📊 Phân Tích Dự Đoán")

        QMessageBox.warning(
            self,
            "Lỗi",
            f"Không thể tải dữ liệu:\n{error_msg}\n\n"
            "Vui lòng kiểm tra API server đang chạy."
        )
