"""
Author: Haowen Wang
Last Edit: 2021/11/18
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
        self.ui.btn_load_t.clicked.connect(self.loadGraphics)
        self.ui.btn_show_t.clicked.connect(self.detect_cut)
        self.loadPath = None
        self.timer = QTimer(self.ui)
        self.poss_num = 5 # number of possible cuts
        self.ui.tab_operations.setCurrentIndex(0)

    def loadGraphics(self):
        self.timer.stop()
        self.loadPath = self.tool.getFilePath()
        pixmap = self.tool.img2pix(self.loadPath)
        self.tool.disGraphics(pixmap)

    def detect(self):
        check = self.tool.checkLoadPath(self.loadPath)
        if check:
            figure = cv2.imread(self.loadPath)
            detect_img, className, probabilityValue = TSD_single_predict(figure)
            detect_img = cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB)
            x = detect_img.shape[1]
            y = detect_img.shape[0]
            frame = QImage(detect_img, x, y, QImage.Format_RGB888)
            pixmap = self.tool.img2pix(frame)
            self.tool.disGraphics(pixmap)
            self.tool.disResults(className, probabilityValue)

    def detect_cut(self):
        check = self.tool.checkLoadPath(self.loadPath)
        if check:
            option = self.ui.comboBox_cut.currentText()
            figure = cv2.imread(self.loadPath)
            cut = self.tool.cutSign(figure, option, self.poss_num)
            detect_img_list = []
            className_list = []
            probabilityValue_list = []
            for i in range(self.poss_num):
                d, c, p = TSD_single_predict(cut[i])
                detect_img_list.append(d)
                className_list.append(c)
                probabilityValue_list.append(p)
            output_idx = probabilityValue_list.index(max(probabilityValue_list, key=abs))
            detect_img = detect_img_list[output_idx]
            className = className_list[output_idx]
            probabilityValue = probabilityValue_list[output_idx]
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
