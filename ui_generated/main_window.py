# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main_window.ui'
################################################################################

from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1200, 800)

        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")

        self.mainLayout = QtWidgets.QHBoxLayout(self.centralWidget)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.setSpacing(0)
        self.mainLayout.setObjectName("mainLayout")

        # Sidebar
        self.sidebarWidget = QtWidgets.QWidget(self.centralWidget)
        self.sidebarWidget.setMaximumWidth(250)
        self.sidebarWidget.setObjectName("sidebarWidget")

        self.sidebarLayout = QtWidgets.QVBoxLayout(self.sidebarWidget)
        self.sidebarLayout.setObjectName("sidebarLayout")

        # Logo
        self.logoLabel = QtWidgets.QLabel(self.sidebarWidget)
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        self.logoLabel.setFont(font)
        self.logoLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.logoLabel.setObjectName("logoLabel")
        self.sidebarLayout.addWidget(self.logoLabel)

        # User info
        self.userInfoWidget = QtWidgets.QFrame(self.sidebarWidget)
        self.userInfoWidget.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.userInfoWidget.setObjectName("userInfoWidget")

        self.userInfoLayout = QtWidgets.QVBoxLayout(self.userInfoWidget)
        self.userInfoLayout.setObjectName("userInfoLayout")

        self.userNameLabel = QtWidgets.QLabel(self.userInfoWidget)
        font = QtGui.QFont()
        font.setBold(True)
        self.userNameLabel.setFont(font)
        self.userNameLabel.setObjectName("userNameLabel")
        self.userInfoLayout.addWidget(self.userNameLabel)

        self.userTierLabel = QtWidgets.QLabel(self.userInfoWidget)
        self.userTierLabel.setObjectName("userTierLabel")
        self.userInfoLayout.addWidget(self.userTierLabel)

        self.userPointsLabel = QtWidgets.QLabel(self.userInfoWidget)
        self.userPointsLabel.setObjectName("userPointsLabel")
        self.userInfoLayout.addWidget(self.userPointsLabel)

        self.sidebarLayout.addWidget(self.userInfoWidget)

        # Navigation buttons
        self.menuButton = QtWidgets.QPushButton(self.sidebarWidget)
        self.menuButton.setMinimumHeight(45)
        self.menuButton.setObjectName("menuButton")
        self.sidebarLayout.addWidget(self.menuButton)

        self.cartButton = QtWidgets.QPushButton(self.sidebarWidget)
        self.cartButton.setMinimumHeight(45)
        self.cartButton.setObjectName("cartButton")
        self.sidebarLayout.addWidget(self.cartButton)

        self.ordersButton = QtWidgets.QPushButton(self.sidebarWidget)
        self.ordersButton.setMinimumHeight(45)
        self.ordersButton.setObjectName("ordersButton")
        self.sidebarLayout.addWidget(self.ordersButton)

        self.favoritesButton = QtWidgets.QPushButton(self.sidebarWidget)
        self.favoritesButton.setMinimumHeight(45)
        self.favoritesButton.setObjectName("favoritesButton")
        self.sidebarLayout.addWidget(self.favoritesButton)

        self.profileButton = QtWidgets.QPushButton(self.sidebarWidget)
        self.profileButton.setMinimumHeight(45)
        self.profileButton.setObjectName("profileButton")
        self.sidebarLayout.addWidget(self.profileButton)

        self.notificationsButton = QtWidgets.QPushButton(self.sidebarWidget)
        self.notificationsButton.setMinimumHeight(45)
        self.notificationsButton.setObjectName("notificationsButton")
        self.sidebarLayout.addWidget(self.notificationsButton)

        # Spacer
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum,
                                           QtWidgets.QSizePolicy.Policy.Expanding)
        self.sidebarLayout.addItem(spacerItem)

        # Logout button
        self.logoutButton = QtWidgets.QPushButton(self.sidebarWidget)
        self.logoutButton.setMinimumHeight(40)
        self.logoutButton.setObjectName("logoutButton")
        self.sidebarLayout.addWidget(self.logoutButton)

        self.mainLayout.addWidget(self.sidebarWidget)

        # Content area (StackedWidget for different views)
        self.contentStackedWidget = QtWidgets.QStackedWidget(self.centralWidget)
        self.contentStackedWidget.setObjectName("contentStackedWidget")
        self.mainLayout.addWidget(self.contentStackedWidget)

        MainWindow.setCentralWidget(self.centralWidget)

        # Status bar
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Coffee Shop"))
        self.logoLabel.setText(_translate("MainWindow", "☕ Coffee Shop"))
        self.userNameLabel.setText(_translate("MainWindow", "Xin chào!"))
        self.userTierLabel.setText(_translate("MainWindow", "🥉 Bronze Member"))
        self.userPointsLabel.setText(_translate("MainWindow", "0 điểm"))
        self.menuButton.setText(_translate("MainWindow", "📋 Menu"))
        self.cartButton.setText(_translate("MainWindow", "🛒 Giỏ hàng (0)"))
        self.ordersButton.setText(_translate("MainWindow", "📦 Đơn hàng"))
        self.favoritesButton.setText(_translate("MainWindow", "❤️ Yêu thích"))
        self.profileButton.setText(_translate("MainWindow", "👤 Tài khoản"))
        self.notificationsButton.setText(_translate("MainWindow", "🔔 Thông báo"))
        self.logoutButton.setText(_translate("MainWindow", "🚪 Đăng xuất"))
