"""
Admin ML Analytics Widget - Revenue Forecasting
Each chart has its own controls
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
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


class CompactChart(FigureCanvas):
    """Compact chart widget"""

    def __init__(self, parent=None, width=6, height=3.5):
        self.fig = Figure(figsize=(width, height), dpi=80)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(280)
        self.setMaximumHeight(320)

        self.fig.patch.set_facecolor('#ffffff')
        self.axes.set_facecolor('#ffffff')
        self.axes.grid(True, alpha=0.2, linestyle='--')

    def plot_line_forecast(self, data, title=None):
        """Plot line chart for forecast"""
        self.axes.clear()

        if not data or 'forecasts' not in data:
            self.axes.text(0.5, 0.5, 'Chưa có dữ liệu', ha='center', va='center')
            self.draw()
            return

        forecasts = data['forecasts']
        dates = [f['date'] for f in forecasts]
        values = [f['forecast'] for f in forecasts]

        self.axes.plot(dates, values, 'b-', linewidth=2.5, marker='o', markersize=4, alpha=0.8)

        if title:
            self.axes.set_title(title, fontsize=11, fontweight='bold', pad=10)

        self.axes.set_xlabel('Ngày', fontsize=9)
        self.axes.set_ylabel('Doanh thu (VNĐ)', fontsize=9)

        if len(dates) > 10:
            step = max(1, len(dates) // 8)
            self.axes.set_xticks(range(0, len(dates), step))
            self.axes.set_xticklabels([dates[i] for i in range(0, len(dates), step)],
                                     rotation=45, ha='right', fontsize=8)
        else:
            self.axes.set_xticks(range(len(dates)))
            self.axes.set_xticklabels(dates, rotation=45, ha='right', fontsize=8)

        from matplotlib.ticker import FuncFormatter
        def currency_formatter(x, p):
            if x >= 1000000:
                return f'{x/1000000:.1f}M'
            elif x >= 1000:
                return f'{x/1000:.0f}K'
            return f'{int(x)}'
        self.axes.yaxis.set_major_formatter(FuncFormatter(currency_formatter))
        self.axes.tick_params(axis='y', labelsize=8)

        self.fig.tight_layout()
        self.draw()

    def plot_bar_comparison(self, stores_data, days, title=""):
        """Plot bar chart for store comparison"""
        self.axes.clear()

        if not stores_data:
            self.axes.text(0.5, 0.5, 'Chưa có dữ liệu', ha='center', va='center')
            self.draw()
            return

        store_names = [f"#{s['store_nbr']}" for s in stores_data]
        revenues = [s['forecast_avg_daily'] * days for s in stores_data]

        colors = ['#4CAF50' if i < len(stores_data)//2 else '#FF9800' for i in range(len(stores_data))]
        bars = self.axes.bar(range(len(store_names)), revenues, color=colors, alpha=0.7)

        for bar in bars:
            height = bar.get_height()
            self.axes.text(bar.get_x() + bar.get_width()/2., height,
                          f'{height/1000:.0f}K',
                          ha='center', va='bottom', fontsize=7)

        self.axes.set_title(title, fontsize=11, fontweight='bold', pad=10)
        self.axes.set_xlabel('Cửa hàng', fontsize=9)
        self.axes.set_ylabel('Doanh thu (VNĐ)', fontsize=9)
        self.axes.set_xticks(range(len(store_names)))
        self.axes.set_xticklabels(store_names, fontsize=8)

        from matplotlib.ticker import FuncFormatter
        def currency_formatter(x, p):
            if x >= 1000000:
                return f'{x/1000000:.1f}M'
            elif x >= 1000:
                return f'{x/1000:.0f}K'
            return f'{int(x)}'
        self.axes.yaxis.set_major_formatter(FuncFormatter(currency_formatter))
        self.axes.tick_params(axis='y', labelsize=8)

        self.fig.tight_layout()
        self.draw()


class AdminMLAnalyticsWidget(QWidget):
    """Admin ML Analytics widget"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.predictor = get_predictor()
        self.setup_ui()
        self.load_stores()

    def setup_ui(self):
        """Setup UI"""
        # Create main layout for this widget
        widget_layout = QVBoxLayout(self)
        widget_layout.setContentsMargins(0, 0, 0, 0)
        widget_layout.setSpacing(0)

        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Create content widget for scroll area
        content_widget = QWidget()
        main_layout = QVBoxLayout(content_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # Title
        title_label = QLabel("📊 Dự Báo Doanh Thu")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
                padding: 12px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #c7a17a, stop:1 #f5f5f5);
                border-radius: 6px;
            }
        """)
        main_layout.addWidget(title_label)

        # STATS FIRST - Thống kê tổng quan
        stats_group = self.create_stats_panel()
        main_layout.addWidget(stats_group)

        # Chart 1: Overall Forecast with own controls
        overall_section = self.create_chart_section(
            "🌐 Dự Báo Tổng Thể Hệ Thống",
            has_store_selector=False,
            chart_name='overall'
        )
        main_layout.addWidget(overall_section)

        # Chart 2: Store Forecast with own controls
        store_section = self.create_chart_section(
            "🏪 Dự Báo Từng Cửa Hàng",
            has_store_selector=True,
            chart_name='store'
        )
        main_layout.addWidget(store_section)

        # Chart 3 & 4: Comparison charts with shared controls
        comparison_section = self.create_comparison_section()
        main_layout.addWidget(comparison_section)

        # Add stretch to push everything up
        main_layout.addStretch()

        # Set content widget to scroll area and add to main widget layout
        scroll_area.setWidget(content_widget)
        widget_layout.addWidget(scroll_area)

    def create_stats_panel(self):
        """Create stats panel"""
        group = QGroupBox("📈 Thống Kê Tổng Quan")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 13px;
                border: 2px solid #2196F3;
                border-radius: 8px;
                margin-top: 10px;
                padding: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
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
                padding: 15px;
                min-height: 80px;
            }}
        """)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(5, 8, 5, 8)
        layout.setSpacing(8)

        title_label = QLabel(title)
        title_label.setStyleSheet(f"color: {color}; font-size: 10px; font-weight: bold;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        layout.addWidget(title_label)

        value_label = QLabel(value)
        value_label.setObjectName("value")
        value_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50;")
        value_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(value_label)

        return card

    def create_chart_section(self, title, has_store_selector, chart_name):
        """Create section with chart and its own controls"""
        section = QGroupBox(title)
        section.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 13px;
                border: 2px solid #c7a17a;
                border-radius: 8px;
                margin-top: 10px;
                padding: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)

        layout = QVBoxLayout(section)

        # Controls
        controls_layout = QHBoxLayout()

        if has_store_selector:
            controls_layout.addWidget(QLabel("Cửa hàng:"))
            store_combo = QComboBox()
            store_combo.addItem("Đang tải...")
            store_combo.setMinimumWidth(200)
            controls_layout.addWidget(store_combo)
            setattr(self, f'{chart_name}_store_combo', store_combo)

        controls_layout.addWidget(QLabel("Thời gian:"))
        period_combo = QComboBox()
        period_combo.addItems(["7 Ngày", "1 Tháng (30)", "1 Quý (90)", "1 Năm (365)", "Tùy chỉnh"])
        controls_layout.addWidget(period_combo)
        setattr(self, f'{chart_name}_period_combo', period_combo)

        # Custom dates (hidden by default)
        start_date = QDateEdit()
        start_date.setCalendarPopup(True)
        start_date.setDate(QDate.currentDate())
        start_date.hide()
        controls_layout.addWidget(start_date)
        setattr(self, f'{chart_name}_start_date', start_date)

        end_date = QDateEdit()
        end_date.setCalendarPopup(True)
        end_date.setDate(QDate.currentDate().addDays(30))
        end_date.hide()
        controls_layout.addWidget(end_date)
        setattr(self, f'{chart_name}_end_date', end_date)

        # Connect period change
        period_combo.currentIndexChanged.connect(
            lambda idx, cn=chart_name: self.on_period_changed(idx, cn)
        )

        # Analyze button
        analyze_btn = QPushButton("📊 Phân Tích")
        analyze_btn.setStyleSheet("""
            QPushButton {
                background-color: #c7a17a;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #a0826d; }
            QPushButton:disabled { background-color: #ccc; }
        """)
        analyze_btn.clicked.connect(lambda: self.analyze_chart(chart_name))
        controls_layout.addWidget(analyze_btn)
        setattr(self, f'{chart_name}_analyze_btn', analyze_btn)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # Chart
        chart = CompactChart(self)
        layout.addWidget(chart)
        setattr(self, f'{chart_name}_chart', chart)

        return section

    def create_comparison_section(self):
        """Create comparison section with shared controls"""
        section = QGroupBox("📊 So Sánh Cửa Hàng")
        section.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 13px;
                border: 2px solid #c7a17a;
                border-radius: 8px;
                margin-top: 10px;
                padding: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)

        layout = QVBoxLayout(section)

        # Shared controls
        controls_layout = QHBoxLayout()

        controls_layout.addWidget(QLabel("Top N:"))
        self.comparison_topn = QSpinBox()
        self.comparison_topn.setRange(3, 20)
        self.comparison_topn.setValue(10)
        controls_layout.addWidget(self.comparison_topn)

        controls_layout.addWidget(QLabel("Thời gian:"))
        self.comparison_period = QComboBox()
        self.comparison_period.addItems(["7 Ngày", "1 Tháng (30)", "1 Quý (90)", "1 Năm (365)"])
        self.comparison_period.setCurrentIndex(1)
        controls_layout.addWidget(self.comparison_period)

        self.comparison_analyze_btn = QPushButton("📊 Phân Tích")
        self.comparison_analyze_btn.setStyleSheet("""
            QPushButton {
                background-color: #c7a17a;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #a0826d; }
            QPushButton:disabled { background-color: #ccc; }
        """)
        self.comparison_analyze_btn.clicked.connect(self.analyze_comparison)
        controls_layout.addWidget(self.comparison_analyze_btn)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # Charts side by side
        charts_layout = QHBoxLayout()

        # Top stores
        top_group = QGroupBox("🏆 Top Cao Nhất")
        top_layout = QVBoxLayout(top_group)
        self.top_chart = CompactChart(self)
        top_layout.addWidget(self.top_chart)
        charts_layout.addWidget(top_group)

        # Bottom stores
        bottom_group = QGroupBox("⚠️ Top Thấp Nhất")
        bottom_layout = QVBoxLayout(bottom_group)
        self.bottom_chart = CompactChart(self)
        bottom_layout.addWidget(self.bottom_chart)
        charts_layout.addWidget(bottom_group)

        layout.addLayout(charts_layout)

        return section

    def on_period_changed(self, index, chart_name):
        """Handle period change"""
        start_date = getattr(self, f'{chart_name}_start_date')
        end_date = getattr(self, f'{chart_name}_end_date')

        if index == 4:  # Tùy chỉnh
            start_date.show()
            end_date.show()
        else:
            start_date.hide()
            end_date.hide()

    def update_stat_card(self, card, value):
        """Update stat card"""
        value_label = card.findChild(QLabel, "value")
        if value_label:
            value_label.setText(value)

    def load_stores(self):
        """Load stores list"""
        try:
            stores = self.predictor.get_all_stores()

            # Update store combo for store chart
            if hasattr(self, 'store_store_combo'):
                self.store_store_combo.clear()
                for store in stores:
                    self.store_store_combo.addItem(
                        f"Store #{store['store_nbr']} - {store['city']} (Type {store['type']})",
                        store['store_nbr']
                    )
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể load danh sách cửa hàng:\n{str(e)}")

    def get_days(self, chart_name):
        """Get days for chart"""
        period_combo = getattr(self, f'{chart_name}_period_combo')
        index = period_combo.currentIndex()

        if index == 4:  # Custom
            start_date = getattr(self, f'{chart_name}_start_date')
            end_date = getattr(self, f'{chart_name}_end_date')
            start = start_date.date().toPyDate()
            end = end_date.date().toPyDate()
            return (end - start).days
        else:
            days_map = {0: 7, 1: 30, 2: 90, 3: 365}
            return days_map[index]

    def analyze_chart(self, chart_name):
        """Analyze specific chart"""
        analyze_btn = getattr(self, f'{chart_name}_analyze_btn')
        analyze_btn.setEnabled(False)
        analyze_btn.setText("⏳ Đang phân tích...")

        days = self.get_days(chart_name)

        if chart_name == 'overall':
            worker = PredictionWorker(self.predictor, 'overall', days=days)
            worker.finished.connect(lambda data: self.on_overall_loaded(data))
        elif chart_name == 'store':
            store_nbr = self.store_store_combo.currentData()
            if not store_nbr:
                QMessageBox.warning(self, "Lỗi", "Vui lòng chọn cửa hàng!")
                analyze_btn.setEnabled(True)
                analyze_btn.setText("📊 Phân Tích")
                return
            worker = PredictionWorker(self.predictor, 'store', store_nbr=store_nbr, days=days)
            worker.finished.connect(lambda data: self.on_store_loaded(data, chart_name))

        worker.error.connect(lambda e: self.on_error(e, chart_name))
        worker.start()
        setattr(self, f'{chart_name}_worker', worker)

    def on_overall_loaded(self, data):
        """Handle overall loaded"""
        self.overall_chart.plot_line_forecast(data, "Dự Báo Tổng Thể Hệ Thống")

        # Update stats
        summary = data.get('summary', {})
        avg = summary.get('avg_daily_forecast', 0)
        total = summary.get('total_forecast', 0)

        self.update_stat_card(self.stat1, f"{avg:,.0f} VNĐ")
        self.update_stat_card(self.stat2, f"{total:,.0f} VNĐ")

        self.overall_analyze_btn.setEnabled(True)
        self.overall_analyze_btn.setText("📊 Phân Tích")

    def on_store_loaded(self, data, chart_name):
        """Handle store loaded"""
        chart = getattr(self, f'{chart_name}_chart')
        store_name = f"Store #{data['store_nbr']} - {data['city']}"
        chart.plot_line_forecast(data, store_name)

        # Update stats
        store_avg = data.get('forecast_avg_daily', 0)
        growth = data.get('growth_percent', 0)

        self.update_stat_card(self.stat3, f"{store_avg:,.0f} VNĐ")
        self.update_stat_card(self.stat4, f"{growth:+.1f}%")

        analyze_btn = getattr(self, f'{chart_name}_analyze_btn')
        analyze_btn.setEnabled(True)
        analyze_btn.setText("📊 Phân Tích")

    def analyze_comparison(self):
        """Analyze comparison"""
        self.comparison_analyze_btn.setEnabled(False)
        self.comparison_analyze_btn.setText("⏳ Đang phân tích...")

        n = self.comparison_topn.value()
        period_idx = self.comparison_period.currentIndex()
        days_map = {0: 7, 1: 30, 2: 90, 3: 365}
        days = days_map[period_idx]

        # Top stores
        worker_top = PredictionWorker(self.predictor, 'top_stores', n=n)
        worker_top.finished.connect(lambda data: self.on_top_loaded(data, days))
        worker_top.error.connect(lambda e: self.on_error(e, 'comparison'))
        worker_top.start()
        self.top_worker = worker_top

        # Bottom stores
        worker_bottom = PredictionWorker(self.predictor, 'bottom_stores', n=n)
        worker_bottom.finished.connect(lambda data: self.on_bottom_loaded(data, days))
        worker_bottom.error.connect(lambda e: self.on_error(e, 'comparison'))
        worker_bottom.start()
        self.bottom_worker = worker_bottom

    def on_top_loaded(self, data, days):
        """Handle top stores loaded"""
        stores = data.get('stores', [])
        self.top_chart.plot_bar_comparison(stores, days, f"Top {len(stores)} Cao Nhất ({days} ngày)")

    def on_bottom_loaded(self, data, days):
        """Handle bottom stores loaded"""
        stores = data.get('stores', [])
        self.bottom_chart.plot_bar_comparison(stores, days, f"Top {len(stores)} Thấp Nhất ({days} ngày)")
        self.comparison_analyze_btn.setEnabled(True)
        self.comparison_analyze_btn.setText("📊 Phân Tích")

    def on_error(self, error_msg, chart_name):
        """Handle error"""
        if chart_name == 'comparison':
            self.comparison_analyze_btn.setEnabled(True)
            self.comparison_analyze_btn.setText("📊 Phân Tích")
        else:
            analyze_btn = getattr(self, f'{chart_name}_analyze_btn')
            analyze_btn.setEnabled(True)
            analyze_btn.setText("📊 Phân Tích")

        QMessageBox.critical(self, "Lỗi", f"Không thể thực hiện dự đoán:\n{error_msg}")
