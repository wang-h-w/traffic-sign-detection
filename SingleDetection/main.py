"""
Author: Haowen Wang
Last Edit: 2021/11/14
"""

from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5 import uic
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap, QIcon
from utils import TSD_single_utils as Tool
from TSD_single_test import TSD_single_predict
import cv2

class TSD_single:

    def __init__(self):
        self.ui = uic.loadUi('ui/single_ui.ui')
        self.tool = Tool(self.ui)
        self.ui.btn_load.clicked.connect(self.loadGraphics)
        self.ui.btn_show.clicked.connect(self.detect)
        self.ui.btn_video.clicked.connect(self.video)
        self.loadPath = None
        self.timer = QTimer(self.ui)

    def loadGraphics(self):
        self.timer.stop()
        self.loadPath = self.tool.getFilePath()
        pixmap = self.tool.img2pix(self.loadPath)
        self.tool.disGraphics(pixmap)

    def detect(self):
        if self.loadPath is None:
            QMessageBox.critical(self.ui, 'Input error', 'Please load an image first!')
        elif self.loadPath == '':
            QMessageBox.critical(self.ui, 'Input error', "Please choose an image from the folder!")
        elif self.loadPath == 'video':
            QMessageBox.warning(self.ui, 'On-line detect warning', "On-line detect is already in use.\nResults are shown in \"Detect Result\" area!")
        else:
            figure = cv2.imread(self.loadPath)
            detect_img, className, probabilityValue = TSD_single_predict(figure)
            detect_img = cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB)
            x = detect_img.shape[1]
            y = detect_img.shape[0]
            frame = QImage(detect_img, x, y, QImage.Format_RGB888)
            pixmap = self.tool.img2pix(frame)
            self.tool.disGraphics(pixmap)
            self.tool.disResults(className, probabilityValue)

    def get_frame(self):
        _, frame = self.capture.read()
        detect, className, probabilityValue = TSD_single_predict(frame)
        image = QImage(detect, *self.dimensions, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(image)
        self.pixmapItem.setPixmap(pixmap)
        self.tool.disResults(className, probabilityValue)

    def video(self):
        self.loadPath = 'video'
        self.capture = cv2.VideoCapture(0)

        frameWidth = 320  # CAMERA RESOLUTION
        frameHeight = 240
        brightness = 180

        # SETUP THE VIDEO CAMERA
        self.capture.set(3, frameWidth)
        self.capture.set(4, frameHeight)
        self.capture.set(10, brightness)

        if self.capture.read()[1] is None:
            QMessageBox.warning(self.ui, 'Open camera error!', 'Please click ON-LINE DETECT button again!')
        else:
            self.dimensions = self.capture.read()[1].shape[1::-1]
            self.pixmapItem = self.tool.disVideo(self.dimensions)
            
            self.timer.setInterval(int(1000/30))
            self.timer.timeout.connect(self.get_frame)
            self.timer.start()

app = QApplication([])
app.setWindowIcon(QIcon('./figs/signal.png'))
window = TSD_single()
window.ui.show()
app.exec_(app.exec_())
