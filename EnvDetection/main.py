"""
Author: Haowen Wang
Last Edit: 2021/11/14
"""

from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5 import uic
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap, QIcon
from utils import TSD_env_utils as Tool
from TSD_env_test import TSD_env_predict
import cv2

class TSD_env:

    def __init__(self):
        self.ui = uic.loadUi('ui/env_ui.ui')
        self.tool = Tool(self.ui)
        self.ui.btn_load.clicked.connect(self.loadGraphics)
        self.ui.btn_show.clicked.connect(self.detect)
        self.ui.btn_next.clicked.connect(self.showNext)
        self.ui.btn_save.clicked.connect(self.save)
        self.loadPath = None
        self.img_save = None

    def loadGraphics(self):
        self.loadPath = self.tool.getFilePath()
        pixmap = self.tool.img2pix(self.loadPath)
        self.tool.disGraphics(pixmap)

    def detect(self):
        if self.loadPath is None:
            QMessageBox.critical(self.ui, 'Input error', 'Please load an image first!')
        elif self.loadPath == '':
            QMessageBox.critical(self.ui, 'Input error', "Please choose an image from the folder!")
        else:
            figure = cv2.imread(self.loadPath)
            self.detect_img, self.img_save, category_dict, crop = TSD_env_predict(figure)
            detect_img = cv2.cvtColor(self.detect_img, cv2.COLOR_BGR2RGB)

            frame = self.tool.frame2img(detect_img)
            pixmap = self.tool.img2pix(frame)
            self.tool.disGraphics(pixmap)

            self.category_dict = category_dict
            self.crop = crop
            self.total_num = len(self.category_dict)
            self.num = 1
            self.showNext()
            
    def showNext(self):
        imgShow = self.crop[self.num-1]
        imgShow = cv2.cvtColor(imgShow, cv2.COLOR_BGR2RGB)
        frame_init = self.tool.frame2img(imgShow)
        pixmap_init = self.tool.img2pix(frame_init)
        self.tool.disGraphicsCrop(pixmap_init)
        self.tool.disResults(self.num, self.category_dict[self.num])
        self.num = self.num + 1
        if self.num > self.total_num:
            self.num = 1

    def save(self):
        filePath = self.tool.saveFilePath()
        if self.img_save is None:
            QMessageBox.critical(self.ui, 'Save error', 'Please load an image and click detect!')
        elif filePath is '':
            QMessageBox.critical(self.ui, 'Save error', 'Please choose a save path!')
        else:
            cv2.imwrite(filePath, self.img_save)

app = QApplication([])
app.setWindowIcon(QIcon('./figs/signal.png'))
window = TSD_env()
window.ui.show()
app.exec_(app.exec_())
