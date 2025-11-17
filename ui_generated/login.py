# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'login.ui'
##
## Created by: Qt User Interface Compiler version 6.0.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_LoginWindow(object):
    def setupUi(self, LoginWindow):
        LoginWindow.setObjectName("LoginWindow")
        LoginWindow.resize(400, 600)

        self.verticalLayout = QtWidgets.QVBoxLayout(LoginWindow)
        self.verticalLayout.setObjectName("verticalLayout")

        # Logo
        self.logoLabel = QtWidgets.QLabel(LoginWindow)
        font = QtGui.QFont()
        font.setPointSize(24)
        font.setBold(True)
        self.logoLabel.setFont(font)
        self.logoLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.logoLabel.setObjectName("logoLabel")
        self.verticalLayout.addWidget(self.logoLabel)

        # Add spacer
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum,
                                           QtWidgets.QSizePolicy.Policy.Expanding)
        self.verticalLayout.addItem(spacerItem)

        # Title
        self.titleLabel = QtWidgets.QLabel(LoginWindow)
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        self.titleLabel.setFont(font)
        self.titleLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.titleLabel.setObjectName("titleLabel")
        self.verticalLayout.addWidget(self.titleLabel)

        # Email input
        self.emailLineEdit = QtWidgets.QLineEdit(LoginWindow)
        self.emailLineEdit.setMinimumHeight(40)
        self.emailLineEdit.setObjectName("emailLineEdit")
        self.verticalLayout.addWidget(self.emailLineEdit)

        # Password input
        self.passwordLineEdit = QtWidgets.QLineEdit(LoginWindow)
        self.passwordLineEdit.setMinimumHeight(40)
        self.passwordLineEdit.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self.passwordLineEdit.setObjectName("passwordLineEdit")
        self.verticalLayout.addWidget(self.passwordLineEdit)

        # Remember me
        self.rememberMeCheckBox = QtWidgets.QCheckBox(LoginWindow)
        self.rememberMeCheckBox.setObjectName("rememberMeCheckBox")
        self.verticalLayout.addWidget(self.rememberMeCheckBox)

        # Login button
        self.loginButton = QtWidgets.QPushButton(LoginWindow)
        self.loginButton.setMinimumHeight(45)
        self.loginButton.setObjectName("loginButton")
        self.verticalLayout.addWidget(self.loginButton)

        # Forgot password button
        self.forgotPasswordButton = QtWidgets.QPushButton(LoginWindow)
        self.forgotPasswordButton.setFlat(True)
        self.forgotPasswordButton.setObjectName("forgotPasswordButton")
        self.verticalLayout.addWidget(self.forgotPasswordButton)

        # Or label
        self.orLabel = QtWidgets.QLabel(LoginWindow)
        self.orLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.orLabel.setObjectName("orLabel")
        self.verticalLayout.addWidget(self.orLabel)

        # Google login
        self.googleLoginButton = QtWidgets.QPushButton(LoginWindow)
        self.googleLoginButton.setMinimumHeight(40)
        self.googleLoginButton.setObjectName("googleLoginButton")
        self.verticalLayout.addWidget(self.googleLoginButton)

        # Apple login
        self.appleLoginButton = QtWidgets.QPushButton(LoginWindow)
        self.appleLoginButton.setMinimumHeight(40)
        self.appleLoginButton.setObjectName("appleLoginButton")
        self.verticalLayout.addWidget(self.appleLoginButton)

        # Add spacer
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum,
                                            QtWidgets.QSizePolicy.Policy.Expanding)
        self.verticalLayout.addItem(spacerItem2)

        # Register label
        self.registerLabel = QtWidgets.QLabel(LoginWindow)
        self.registerLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.registerLabel.setObjectName("registerLabel")
        self.verticalLayout.addWidget(self.registerLabel)

        # Register button
        self.registerButton = QtWidgets.QPushButton(LoginWindow)
        self.registerButton.setMinimumHeight(40)
        self.registerButton.setObjectName("registerButton")
        self.verticalLayout.addWidget(self.registerButton)

        self.retranslateUi(LoginWindow)
        QtCore.QMetaObject.connectSlotsByName(LoginWindow)

    def retranslateUi(self, LoginWindow):
        _translate = QtCore.QCoreApplication.translate
        LoginWindow.setWindowTitle(_translate("LoginWindow", "Đăng nhập - Coffee Shop"))
        self.logoLabel.setText(_translate("LoginWindow", "☕ Coffee Shop"))
        self.titleLabel.setText(_translate("LoginWindow", "Đăng nhập"))
        self.emailLineEdit.setPlaceholderText(_translate("LoginWindow", "Email"))
        self.passwordLineEdit.setPlaceholderText(_translate("LoginWindow", "Mật khẩu"))
        self.rememberMeCheckBox.setText(_translate("LoginWindow", "Ghi nhớ đăng nhập"))
        self.loginButton.setText(_translate("LoginWindow", "Đăng nhập"))
        self.forgotPasswordButton.setText(_translate("LoginWindow", "Quên mật khẩu?"))
        self.orLabel.setText(_translate("LoginWindow", "hoặc"))
        self.googleLoginButton.setText(_translate("LoginWindow", "Đăng nhập với Google"))
        self.appleLoginButton.setText(_translate("LoginWindow", "Đăng nhập với Apple ID"))
        self.registerLabel.setText(_translate("LoginWindow", "Chưa có tài khoản?"))
        self.registerButton.setText(_translate("LoginWindow", "Đăng ký ngay"))
