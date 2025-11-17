"""
Admin ML Analytics Widget - Revenue Forecasting
Direct model prediction without API
"""
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                              QComboBox, QLabel, QMessageBox, QDateEdit, QGroupBox,
                              QScrollArea, QFrame, QGridLayout, QSizePolicy, QSpinBox)
from PyQt6.QtCore import Qt, QDate, QThread, pyqtSignal
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from predictor import get_predictor


class PredictionWorker(QThread):
    """Worker thread for running predictions"""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, predictor, task, **kwargs):
        super().__init__()
        self.predictor = predictor
        self.task = task
        self.kwargs = kwargs

    def run(self):
        """Run prediction task"""
        try:
            if self.task == 'overall':
                result = self.predictor.predict_overall(**self.kwargs)
            elif self.task == 'store':
                result = self.predictor.predict_store(**self.kwargs)
            elif self.task == 'top_stores':
                result = self.predictor.get_top_stores(**self.kwargs)
            elif self.task == 'bottom_stores':
                result = self.predictor.get_bottom_stores(**self.kwargs)
            else:
                raise ValueError(f"Unknown task: {self.task}")
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class CompactChart(FigureCanvas):
    """Compact chart widget"""

    def __init__(self, parent=None, width=5, height=3):
        self.fig = Figure(figsize=(width, height), dpi=80)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(250)

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

        self.axes.plot(dates, values, 'b-', linewidth=2, marker='o', markersize=3, alpha=0.7)

        if title:
            self.axes.set_title(title, fontsize=10, fontweight='bold', pad=10)

        self.axes.set_xlabel('Ngày', fontsize=8)
        self.axes.set_ylabel('Doanh thu', fontsize=8)

        if len(dates) > 10:
            step = len(dates) // 8
            self.axes.set_xticks(range(0, len(dates), step))
            self.axes.set_xticklabels([dates[i] for i in range(0, len(dates), step)],
                                     rotation=45, ha='right', fontsize=7)
        else:
            self.axes.set_xticks(range(len(dates)))
            self.axes.set_xticklabels(dates, rotation=45, ha='right', fontsize=7)

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

    def plot_bar_comparison(self, stores_data, days, title=""):
        """Plot bar chart for store comparison"""
        self.axes.clear()

        if not stores_data:
            self.axes.text(0.5, 0.5, 'Không có dữ liệu', ha='center', va='center')
            self.draw()
            return

        store_names = [f"#{s['store_nbr']}" for s in stores_data]
        revenues = [s['forecast_avg_daily'] * days for s in stores_data]

        bars = self.axes.bar(range(len(store_names)), revenues, color='#4CAF50', alpha=0.7)

        for bar in bars:
            height = bar.get_height()
            self.axes.text(bar.get_x() + bar.get_width()/2., height,
                          f'{height/1000:.0f}K',
                          ha='center', va='bottom', fontsize=7)

        self.axes.set_title(title, fontsize=10, fontweight='bold', pad=10)
        self.axes.set_xlabel('Cửa hàng', fontsize=8)
        self.axes.set_ylabel('Doanh thu', fontsize=8)
        self.axes.set_xticks(range(len(store_names)))
        self.axes.set_xticklabels(store_names, fontsize=7)

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
    """Admin ML Analytics widget"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.predictor = get_predictor()
        self.current_data = {}
        self.setup_ui()
        self.load_stores()

    def setup_ui(self):
        """Setup UI"""
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

        # Controls
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)

        # Stats
        stats_group = self.create_stats_panel()
        main_layout.addWidget(stats_group)

        # Charts row 1
        forecasts_layout = QHBoxLayout()

        self.overall_group = self.create_forecast_group("🌐 Dự Báo Tổng Thể")
        self.overall_chart = CompactChart(self)
        self.overall_group.layout().addWidget(self.overall_chart)
        forecasts_layout.addWidget(self.overall_group)

        self.store_group = self.create_forecast_group("🏪 Dự Báo Từng Cửa Hàng")
        self.store_chart = CompactChart(self)
        self.store_group.layout().addWidget(self.store_chart)
        forecasts_layout.addWidget(self.store_group)

        main_layout.addLayout(forecasts_layout)

        # Charts row 2
        comparison_layout = QHBoxLayout()

        self.top_group = self.create_forecast_group("🏆 Top Cửa Hàng Cao Nhất")
        self.top_chart = CompactChart(self)
        self.top_group.layout().addWidget(self.top_chart)
        comparison_layout.addWidget(self.top_group)

        self.bottom_group = self.create_forecast_group("⚠️ Top Cửa Hàng Thấp Nhất")
        self.bottom_chart = CompactChart(self)
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

        # Store selector
        layout.addWidget(QLabel("Chọn cửa hàng:"), 0, 0)
        self.store_combo = QComboBox()
        self.store_combo.addItem("Đang tải...")
        self.store_combo.setMinimumWidth(250)
        layout.addWidget(self.store_combo, 0, 1)

        # Period selector - DROPDOWN
        layout.addWidget(QLabel("Khoảng thời gian:"), 0, 2)
        self.period_combo = QComboBox()
        self.period_combo.addItems([
            "7 Ngày",
            "1 Tháng (30 ngày)",
            "1 Quý (90 ngày)",
            "1 Năm (365 ngày)",
            "Tùy chỉnh"
        ])
        self.period_combo.currentIndexChanged.connect(self.on_period_changed)
        layout.addWidget(self.period_combo, 0, 3)

        # Custom date range (hidden by default)
        self.custom_label = QLabel("Từ:")
        self.custom_label.hide()
        layout.addWidget(self.custom_label, 1, 0)

        self.start_date = QDateEdit()
        self.start_date.setCalendarPopup(True)
        self.start_date.setDate(QDate.currentDate())
        self.start_date.hide()
        layout.addWidget(self.start_date, 1, 1)

        self.to_label = QLabel("Đến:")
        self.to_label.hide()
        layout.addWidget(self.to_label, 1, 2)

        self.end_date = QDateEdit()
        self.end_date.setCalendarPopup(True)
        self.end_date.setDate(QDate.currentDate().addDays(30))
        self.end_date.hide()
        layout.addWidget(self.end_date, 1, 3)

        # Top N
        layout.addWidget(QLabel("Top N cửa hàng:"), 2, 0)
        self.topn_spin = QSpinBox()
        self.topn_spin.setRange(3, 20)
        self.topn_spin.setValue(10)
        layout.addWidget(self.topn_spin, 2, 1)

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
        layout.addWidget(self.analyze_btn, 2, 2, 1, 2)

        return group

    def on_period_changed(self, index):
        """Handle period selection change"""
        if index == 4:  # Tùy chỉnh
            self.custom_label.show()
            self.start_date.show()
            self.to_label.show()
            self.end_date.show()
        else:
            self.custom_label.hide()
            self.start_date.hide()
            self.to_label.hide()
            self.end_date.hide()

    def create_stats_panel(self):
        """Create stats panel"""
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
        """Create forecast group"""
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
        """Update stat card"""
        value_label = card.findChild(QLabel, "value")
        if value_label:
            value_label.setText(value)

    def load_stores(self):
        """Load stores list"""
        try:
            stores = self.predictor.get_all_stores()
            self.store_combo.clear()
            for store in stores:
                self.store_combo.addItem(
                    f"Store #{store['store_nbr']} - {store['city']} (Type {store['type']})",
                    store['store_nbr']
                )
        except Exception as e:
            QMessageBox.warning(self, "Lỗi", f"Không thể load danh sách cửa hàng:\n{str(e)}")

    def get_days(self):
        """Get number of days from selection"""
        period_index = self.period_combo.currentIndex()
        if period_index == 4:  # Custom
            start = self.start_date.date().toPyDate()
            end = self.end_date.date().toPyDate()
            return (end - start).days
        else:
            days_map = {0: 7, 1: 30, 2: 90, 3: 365}
            return days_map[period_index]

    def run_analysis(self):
        """Run analysis"""
        if self.store_combo.currentIndex() < 0:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn cửa hàng!")
            return

        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setText("⏳ Đang phân tích...")

        store_nbr = self.store_combo.currentData()
        days = self.get_days()
        top_n = self.topn_spin.value()

        self.current_params = {'store_nbr': store_nbr, 'days': days, 'top_n': top_n}

        # Run overall prediction
        self.worker = PredictionWorker(self.predictor, 'overall', days=days)
        self.worker.finished.connect(self.on_overall_loaded)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_overall_loaded(self, data):
        """Handle overall forecast loaded"""
        self.current_data['overall'] = data
        self.overall_chart.plot_line_forecast(data, "Dự Báo Tổng Thể Hệ Thống")

        # Run store prediction
        worker = PredictionWorker(self.predictor, 'store',
                                 store_nbr=self.current_params['store_nbr'],
                                 days=self.current_params['days'])
        worker.finished.connect(self.on_store_loaded)
        worker.error.connect(self.on_error)
        worker.start()
        self.store_worker = worker

    def on_store_loaded(self, data):
        """Handle store forecast loaded"""
        self.current_data['store'] = data

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

        # Get top/bottom stores
        self.fetch_comparison()

    def fetch_comparison(self):
        """Fetch comparison data"""
        n = self.current_params['top_n']

        # Top stores
        worker_top = PredictionWorker(self.predictor, 'top_stores', n=n)
        worker_top.finished.connect(lambda data: self.on_top_loaded(data, self.current_params['days']))
        worker_top.error.connect(self.on_error)
        worker_top.start()
        self.top_worker = worker_top

        # Bottom stores
        worker_bottom = PredictionWorker(self.predictor, 'bottom_stores', n=n)
        worker_bottom.finished.connect(lambda data: self.on_bottom_loaded(data, self.current_params['days']))
        worker_bottom.error.connect(self.on_error)
        worker_bottom.start()
        self.bottom_worker = worker_bottom

    def on_top_loaded(self, data, days):
        """Handle top stores loaded"""
        self.top_chart.plot_bar_comparison(data, days, f"Top {len(data)} Cao Nhất")

    def on_bottom_loaded(self, data, days):
        """Handle bottom stores loaded"""
        self.bottom_chart.plot_bar_comparison(data, days, f"Top {len(data)} Thấp Nhất")

        # Re-enable button
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("📊 Phân Tích Dự Đoán")

    def on_error(self, error_msg):
        """Handle error"""
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("📊 Phân Tích Dự Đoán")
        QMessageBox.warning(self, "Lỗi", f"Không thể thực hiện dự đoán:\n{error_msg}")
