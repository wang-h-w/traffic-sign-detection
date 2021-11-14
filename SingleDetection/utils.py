"""
Author: Haowen Wang
Last Edit: 2021/11/14
"""

import PyQt5
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

class TSD_single_utils:

    def __init__(self, ui):
        self.ui = ui

    def getFilePath(self):
        dlg = QFileDialog()
        fname = dlg.getOpenFileName(self.ui, 'Open File', './SingleDetection/figs', 'Image files (*.jpg *.png *jpeg)')
        filePath = fname[0]
        return filePath

    def img2pix(self, filePath):
        self.ui.graphicsView.scene_img = QGraphicsScene()
        pixmap = QPixmap(filePath)
        return pixmap

    def disGraphics(self, pixmap):
        imgShowItem = QGraphicsPixmapItem()
        imgShowItem.setPixmap(QPixmap(pixmap))
        self.ui.graphicsView.scene_img.addItem(imgShowItem)
        self.ui.graphicsView.setScene(self.ui.graphicsView.scene_img)
        self.ui.graphicsView.fitInView(QGraphicsPixmapItem(QPixmap(pixmap)), Qt.KeepAspectRatio)

    def disVideo(self, dimensions):
        scene = QGraphicsScene(self.ui)
        pixmap = QPixmap(*dimensions)
        self.pixmapItem = scene.addPixmap(pixmap)
        self.ui.graphicsView.setScene(scene)
        self.ui.graphicsView.fitInView(QGraphicsPixmapItem(QPixmap(pixmap)))
        return self.pixmapItem
    
    def showFigure(self, filepath):
        pixmap = self.img2pix(filepath)
        self.ui.label_figure.setPixmap(pixmap)

    def disResults(self, className, probabilityValue):
        if probabilityValue >= 0.75:
            self.ui.label_category.setStyleSheet('color: forestgreen')
            self.ui.label_probability.setStyleSheet('color: forestgreen')
        else:
            self.ui.label_category.setStyleSheet('color: black')
            self.ui.label_probability.setStyleSheet('color: black')
        self.ui.label_category.setText(className)
        self.ui.label_probability.setText(str(probabilityValue))
    
