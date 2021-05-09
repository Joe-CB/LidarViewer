'''
程序介绍：
可视化ATL04等数据。
模块：
1. Qt
2. Matplotlib
3. Load data
4. Config
5. 左右键消息控制
6. 处理
'''

import os
import sys
sys.path.insert(0, 'G:/pycharm/Visualizer')
print('\n'.join(sys.path))
import numpy as np
# PyQt
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget, QMainWindow, QApplication, QAction, qApp, QFileDialog
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
# Matplotlib
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
# Self Project File
from Config import *
import read_write


class SlimImage:
    def __init__(
            self,
            img=None,
            valid_range=(13, 18),
            draw_column=800,
            *,
            alt=None):
        '''
        显示一个细长的图片，由于太长每次只能显示一部分。
        :param img:要显示的图像       [row, column, channel]
        :param draw_column:每次显示的列数
        '''
        if img.shape[0] > img.shape[1]:
            img = img.T
        self.img = img
        self.r=img.shape[0]
        self.c=img.shape[1]
        self.mode = 'NRB'           # NRB CAB Density
        self.draw_column = draw_column
        self.position = 0
        self.valid_range = valid_range
        self.alt = alt

    def Former(self):
        self.position -= self.draw_column
        if self.position < 0:
            self.position = 0

    def Next(self):
        self.position += self.draw_column
        if (self.position + self.draw_column) >= self.c:
            self.position = self.c - self.draw_column

    def getNow(self):
        return self.img[:, self.position:(self.position + self.draw_column)]

    def resetCenter(self, center):
        if center >=0 and center < self.c:
            self.position = center - int(0.5 * self.draw_column)
        if self.position < 0:
            self.position = 0
        if self.position > self.img.shape[1] - self.draw_column:
            self.position = self.img.shape[1] - self.draw_column

    def getValidRange(self):
        return self.valid_range

    def getAltitude(self):
        return self.alt

class MplCanvas(FigureCanvasQTAgg):
    '''
    讲Matplotlib封装到QT的类
    '''
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)

        # fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = ax
        self.fig = fig
        # self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)



class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        '''
        初始化主窗口
        :param args:
        :param kwargs:
        '''
        super(MainWindow, self).__init__(*args, **kwargs)
        # config cache
        self.config_cache = Config()

        sc = MplCanvas(self, width=5, height=4, dpi=100)
        # sc.axes.plot([0,1,2,3,4], [10,1,20,3,40])

        self.sc = sc

        # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
        toolbar = NavigationToolbar(sc, self)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(toolbar)

        address_layout = self.initAddress()

        layout.addLayout(address_layout)
        layout.addWidget(sc)

        # Create a placeholder widget to hold our toolbar and canvas.
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        # Status Bar
        self.statusBar().showMessage('[None File.]')

        # Meanu
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&文件')
        
        '''Actions'''
        # OpenNRB
        act_NRB = QAction(QIcon('F:\Python\PyQt5\MenusAndToolbar\images\exit.png'), '&Open NRB', self)
        act_NRB.setStatusTip('Open NRB file')
        act_NRB.triggered.connect(self.openNRB)
        fileMenu.addAction(act_NRB)
        # OpenCAB
        act_CAB = QAction(QIcon('F:\Python\PyQt5\MenusAndToolbar\images\exit.png'), '&Open CAB', self)
        act_CAB.setStatusTip('Open CAB file')
        act_CAB.triggered.connect(self.openCAB)
        fileMenu.addAction(act_CAB)
        # OpenCAL_L1
        act_CAL_L1 = QAction(QIcon('F:\Python\PyQt5\MenusAndToolbar\images\exit.png'), '&Open CAL_L1', self)
        act_CAL_L1.setStatusTip('Open CAL_L1 file')
        act_CAL_L1.triggered.connect(self.openCAL_L1)
        fileMenu.addAction(act_CAL_L1)
        
        # self.setGeometry(300, 300, 300, 220)
        # self.setWindowTitle('菜单栏')
        # self.show()
        self.show()

    def initAddress(self):
        '''初始化address空间，用于跳转'''
        address_layout = QtWidgets.QHBoxLayout()
        self.address = QtWidgets.QLineEdit()
        self.address.setValidator(QtGui.QIntValidator(0, 6553500))
        self.address_enter = QtWidgets.QPushButton(text='Go')
        self.address_enter.clicked.connect(self.handleAddressRequest)
        address_layout.addWidget(self.address)
        address_layout.addWidget(self.address_enter)
        return address_layout

    def handleAddressRequest(self):
        '''
        处理显示位置跳转请求
        :return:None
        '''
        position = int(self.address.text())
        self.img_manager.resetCenter(center=position)
        self.drawImg()

    def drawImg(self):
        '''
        获得图像，并显示图像
        :return:
        '''
        img = self.img_manager.getNow()
        valid_range = self.img_manager.getValidRange()
        altitude = self.img_manager.getAltitude()
        # Get the images on an axis
        im = self.sc.axes.images
        # Assume colorbar was plotted last one plotted last
        if len(im) > 0:
            cb = im[-1].colorbar
            # Do any actions on the colorbar object (e.g. remove it)
            cb.remove()
        self.sc.axes.cla()

        pos = self.sc.axes.imshow(img, cmap='jet', vmin=valid_range[0], vmax=valid_range[1], interpolation='nearest')
        plt.colorbar(pos, ax=self.sc.axes)
        if altitude is not None:
            y_label_pos = np.arange(0, altitude.shape[0], 20).astype(int)
            self.sc.axes.set_yticks(y_label_pos)
            self.sc.axes.set_yticklabels(altitude[y_label_pos])
        self.sc.draw()

    def openNRB(self):
        '''
        打开文件，并将数据给SlimImage类处理
        :return:
        '''
        filename = self.config_cache.getConfig('NRB_OPEN_DIR')
        filename = QFileDialog.getOpenFileName(directory=filename)[0]
        if len(filename) == 0:
            return
        else:
            self.config_cache.addConfig('NRB_OPEN_DIR', filename)

            open_result = read_write.AtlasReader.readNRB(filename)
            img = open_result['data']
            altitude = open_result['Altitude']

            self.img_manager = SlimImage(img.T, valid_range=(13, 18), alt=altitude)
            self.drawImg()
            return

    def openCAB(self):
        '''
        打开文件，并将数据给SlimImage类处理
        :return:
        '''
        filename = self.config_cache.getConfig('CAB_OPEN_DIR')
        filename = QFileDialog.getOpenFileName(directory=filename)[0]
        if len(filename) == 0:
            return
        else:
            self.config_cache.addConfig('CAB_OPEN_DIR', filename)

            open_result = read_write.AtlasReader.readCAB(filename)
            img = open_result['data']
            altitude = open_result['Altitude']

            self.img_manager = SlimImage(img.T, valid_range=(-6, -1), alt=altitude)
            self.drawImg()
            return

    def openCAL_L1(self):
        '''
                打开文件，并将数据给SlimImage类处理
                :return:
                '''
        filename = self.config_cache.getConfig('CAL_L1_OPEN_DIR')
        filename = QFileDialog.getOpenFileName(directory=filename)[0]
        if len(filename) == 0:
            return
        else:
            self.config_cache.addConfig('CAL_L1_OPEN_DIR', filename)

            open_result = read_write.CaliopReader.read_TAB_532(filename)
            img = open_result['data']
            altitude = open_result['Altitude']

            self.img_manager = SlimImage(img.T, valid_range=(-6, -1), alt=altitude)
            self.drawImg()
            return
    
    def handleKeypoardLeft(self):
        self.img_manager.Former()
        self.drawImg()

    def handleKeypoardRight(self):
        self.img_manager.Next()
        self.drawImg()

    # 重新实现各事件处理程序
    def keyPressEvent(self, event):
        key = event.key()
        if key == 16777234:
            self.handleKeypoardLeft()
        if key == 16777236:
            self.handleKeypoardRight()
        print(key)
        if Qt.Key_A <= key <= Qt.Key_Z:
            if event.modifiers() & Qt.ShiftModifier:  # Shift 键被按下
                self.statusBar().showMessage('"Shift+%s" pressed' % chr(key), 500)
            elif event.modifiers() & Qt.ControlModifier:  # Ctrl 键被按下
                self.statusBar().showMessage('"Control+%s" pressed' % chr(key), 500)
            elif event.modifiers() & Qt.AltModifier:  # Alt 键被按下
                self.statusBar().showMessage('"Alt+%s" pressed' % chr(key), 500)
            else:
                self.statusBar().showMessage('"%s" pressed' % chr(key), 500)

        elif key == Qt.Key_Home:
            self.statusBar().showMessage('"Home" pressed', 500)
        elif key == Qt.Key_End:
            self.statusBar().showMessage('"End" pressed', 500)
        elif key == Qt.Key_PageUp:
            self.statusBar().showMessage('"PageUp" pressed', 500)
        elif key == Qt.Key_PageDown:
            self.statusBar().showMessage('"PageDown" pressed', 500)
        else:  # 其它未设定的情况
            QWidget.keyPressEvent(self, event)  # 留给基类处理
        '''
        其它常用按键：
        Qt.Key_Escape,Qt.Key_Tab,Qt.Key_Backspace,Qt.Key_Return,Qt.Key_Enter,
        Qt.Key_Insert,Qt.Key_Delete,Qt.Key_Pause,Qt.Key_Print,Qt.Key_F1...Qt.Key_F12,
        Qt.Key_Space,Qt.Key_0...Qt.Key_9,Qt.Key_Colon,Qt.Key_Semicolon,Qt.Key_Equal
        ...
        '''

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    app.exec_()