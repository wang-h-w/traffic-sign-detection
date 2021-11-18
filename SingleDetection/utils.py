"""
Author: Haowen Wang
Last Edit: 2021/11/18
"""

import PyQt5
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import cv2
import numpy as np

class TSD_single_utils:

    def __init__(self, ui):
        self.ui = ui

    def getFilePath(self):
        dlg = QFileDialog()
        fname = dlg.getOpenFileName(self.ui, 'Open File', './SingleDetection/figs', 'Image files (*.jpg *.png *jpeg)')
        filePath = fname[0]
        return filePath

    def checkLoadPath(self, loadPath):
        if loadPath is None:
            QMessageBox.critical(self.ui, 'Input error', 'Please load an image first!')
            return False
        elif loadPath == '':
            QMessageBox.critical(self.ui, 'Input error', "Please choose an image from the folder!")
            return False
        elif loadPath == 'video':
            QMessageBox.warning(self.ui, 'On-line detect warning', "On-line detect is already in use.\nResults are shown in \"Detect Result\" area!")
            return False
        else:
            return True
    
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
    
    def cutSign(self, figure, option, poss_num):
        flag = -1
        hsv = cv2.cvtColor(figure, cv2.COLOR_BGR2HSV)
        if option == 'cut red sign':
            low_hsv = np.array([0, 43, 46])
            high_hsv = np.array([10, 255, 255])
            img_range = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)
        elif option == 'cut blue sign':
            low_hsv = np.array([100, 43, 46])
            high_hsv = np.array([124, 255, 255])
            img_range = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)
        else:
            low_hsv = np.array([0, 43, 46])
            high_hsv = np.array([10, 255, 255])
            low_hsv2 = np.array([100, 43, 46])
            high_hsv2 = np.array([124, 255, 255])
            mask = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)
            mask2 = cv2.inRange(hsv, lowerb=low_hsv2, upperb=high_hsv2)
            img_range = cv2.bitwise_or(mask, mask2)
        blur = cv2.GaussianBlur(img_range, (5, 5), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        ker = np.ones((5, 5), np.uint8)
        close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, ker)
        contours, _ = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        img = [] # record 5 possible cuts
        for _ in range(poss_num):
            img.append(np.zeros((100, 100, 3), np.uint8))
        for j in contours:
            x, y, w, h = cv2.boundingRect(j)
            if 0.8 <= w / h <= 1.3:
                if w * h >= 200:
                    flag += 1
                    img[flag] = figure[y:y+h, x:x+w]
                    img[flag] = cv2.resize(img[flag], (100, 100))

        return img
