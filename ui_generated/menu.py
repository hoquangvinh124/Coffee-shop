# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'menu.ui'
################################################################################

from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_MenuWidget(object):
    def setupUi(self, MenuWidget):
        MenuWidget.setObjectName("MenuWidget")
        MenuWidget.resize(900, 700)

        self.mainLayout = QtWidgets.QVBoxLayout(MenuWidget)
        self.mainLayout.setObjectName("mainLayout")

        # Header
        self.headerWidget = QtWidgets.QWidget(MenuWidget)
        self.headerWidget.setObjectName("headerWidget")

        self.headerLayout = QtWidgets.QHBoxLayout(self.headerWidget)
        self.headerLayout.setObjectName("headerLayout")

        # Title
        self.titleLabel = QtWidgets.QLabel(self.headerWidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        self.titleLabel.setFont(font)
        self.titleLabel.setObjectName("titleLabel")
        self.headerLayout.addWidget(self.titleLabel)

        # Spacer
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding,
                                           QtWidgets.QSizePolicy.Policy.Minimum)
        self.headerLayout.addItem(spacerItem)

        # Search box
        self.searchLineEdit = QtWidgets.QLineEdit(self.headerWidget)
        self.searchLineEdit.setMinimumWidth(300)
        self.searchLineEdit.setMinimumHeight(35)
        self.searchLineEdit.setObjectName("searchLineEdit")
        self.headerLayout.addWidget(self.searchLineEdit)

        self.mainLayout.addWidget(self.headerWidget)

        # Category tabs
        self.categoryTabWidget = QtWidgets.QTabWidget(MenuWidget)
        self.categoryTabWidget.setObjectName("categoryTabWidget")
        self.mainLayout.addWidget(self.categoryTabWidget)

        # Filter widget
        self.filterWidget = QtWidgets.QWidget(MenuWidget)
        self.filterWidget.setMaximumHeight(50)
        self.filterWidget.setObjectName("filterWidget")

        self.filterLayout = QtWidgets.QHBoxLayout(self.filterWidget)
        self.filterLayout.setObjectName("filterLayout")

        # Filter label
        self.filterLabel = QtWidgets.QLabel(self.filterWidget)
        self.filterLabel.setObjectName("filterLabel")
        self.filterLayout.addWidget(self.filterLabel)

        # Temperature filter
        self.hotCheckBox = QtWidgets.QCheckBox(self.filterWidget)
        self.hotCheckBox.setObjectName("hotCheckBox")
        self.filterLayout.addWidget(self.hotCheckBox)

        self.coldCheckBox = QtWidgets.QCheckBox(self.filterWidget)
        self.coldCheckBox.setObjectName("coldCheckBox")
        self.filterLayout.addWidget(self.coldCheckBox)

        # Caffeine filter
        self.caffeineCheckBox = QtWidgets.QCheckBox(self.filterWidget)
        self.caffeineCheckBox.setObjectName("caffeineCheckBox")
        self.filterLayout.addWidget(self.caffeineCheckBox)

        # Spacer
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding,
                                            QtWidgets.QSizePolicy.Policy.Minimum)
        self.filterLayout.addItem(spacerItem2)

        self.mainLayout.addWidget(self.filterWidget)

        # Products scroll area
        self.scrollArea = QtWidgets.QScrollArea(MenuWidget)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")

        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")

        # Grid layout for products
        self.productsGridLayout = QtWidgets.QGridLayout(self.scrollAreaWidgetContents)
        self.productsGridLayout.setObjectName("productsGridLayout")

        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.mainLayout.addWidget(self.scrollArea)

        self.retranslateUi(MenuWidget)
        QtCore.QMetaObject.connectSlotsByName(MenuWidget)

    def retranslateUi(self, MenuWidget):
        _translate = QtCore.QCoreApplication.translate
        MenuWidget.setWindowTitle(_translate("MenuWidget", "Menu"))
        self.titleLabel.setText(_translate("MenuWidget", "📋 Menu"))
        self.searchLineEdit.setPlaceholderText(_translate("MenuWidget", "🔍 Tìm kiếm sản phẩm..."))
        self.filterLabel.setText(_translate("MenuWidget", "Lọc:"))
        self.hotCheckBox.setText(_translate("MenuWidget", "🔥 Nóng"))
        self.coldCheckBox.setText(_translate("MenuWidget", "❄️ Lạnh"))
        self.caffeineCheckBox.setText(_translate("MenuWidget", "☕ Không caffeine"))
