"""
Author: Haowen Wang
Last Edit: 2021/11/14
"""

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

class TSD_env_utils:

    def __init__(self, ui):
        self.ui = ui

    def getFilePath(self):
        dlg = QFileDialog()
        fname = dlg.getOpenFileName(self.ui, 'Open File', '/Users/wang.h.w/Desktop/Courses/Computer Graphics/Course Design/EnvDetection/figs', 'Image files (*.jpg *.png *.jpeg)')
        filePath = fname[0]
        return filePath

    def saveFilePath(self):
        dlg = QFileDialog()
        fname = dlg.getSaveFileName(self.ui, 'Save Detected Picture', '/Users/wang.h.w/Desktop/Courses/Computer Graphics/Course Design/EnvDetection/figs', 'Image files (*.jpg *.png *.jpeg)')
        filePath = fname[0]
        return filePath

    def frame2img(self, frame):
        x = frame.shape[1]
        y = frame.shape[0]
        img = QImage(frame, x, y, QImage.Format_RGB888)
        return img

    def img2pix(self, filePath):
        pixmap = QPixmap(filePath)
        return pixmap

    def disGraphics(self, pixmap):
        imgShowItem = QGraphicsPixmapItem()
        imgShowItem.setPixmap(QPixmap(pixmap))
        self.ui.graphicsView_whole.scene_img = QGraphicsScene()
        self.ui.graphicsView_whole.scene_img.addItem(imgShowItem)
        self.ui.graphicsView_whole.setScene(self.ui.graphicsView_whole.scene_img)
        self.ui.graphicsView_whole.fitInView(QGraphicsPixmapItem(QPixmap(pixmap)), Qt.KeepAspectRatio)
    
    def disGraphicsCrop(self, pixmap):
        imgShowItem = QGraphicsPixmapItem()
        imgShowItem.setPixmap(QPixmap(pixmap))
        self.ui.graphicsView_crop.scene_img = QGraphicsScene()
        self.ui.graphicsView_crop.scene_img.addItem(imgShowItem)
        self.ui.graphicsView_crop.setScene(self.ui.graphicsView_crop.scene_img)
        self.ui.graphicsView_crop.fitInView(QGraphicsPixmapItem(QPixmap(pixmap)), Qt.KeepAspectRatio)

    def disResults(self, num, category):
        self.ui.label_number.setStyleSheet('color: forestgreen')
        self.ui.label_category.setStyleSheet('color: forestgreen')
        self.ui.label_number.setText(str(num))
        self.ui.label_category.setText(category)
    
