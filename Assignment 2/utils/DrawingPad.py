import sys

import numpy as np
import PIL.Image as Image
from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtGui import QImage, QPainter, QPen
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget


class DrawingPad(QWidget):
    def __init__(self):
        super().__init__()
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.setFixedSize(600, 200)
        self.image.fill(Qt.black)
        self.drawing = False
        self.brushSize = 10
        self.brushColor = Qt.white
        self.lastPoint = QPoint()
        
    def get_pil_image(self):
        """Convert the QImage to a PIL Image."""
        qimage = self.image.convertToFormat(QImage.Format.Format_RGB32)

        width = qimage.width()
        height = qimage.height()
        ptr = qimage.bits()
        ptr.setsize(height * width * 4)
        arr = np.array(ptr).reshape(height, width, 4)  # 32-bit QImage to array
        return (Image.fromarray(arr[..., :3]))
        
    def displayImage(self, imagePath):
        loadedImage = QImage(imagePath)
        if loadedImage.isNull():
            return

        # Resize if necessary and display the image
        self.image = loadedImage.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.update()  # Trigger a repaint

    def clearDrawing(self):
        """Clear the drawing area."""
        self.image.fill(Qt.black)
        self.update()
    
    def resizeEvent(self, event):
        if self.image.size() != self.size():
            newImage = QImage(self.size(), QImage.Format_RGB32)
            newImage.fill(Qt.black)
            painter = QPainter(newImage)
            painter.drawImage(QPoint(0, 0), self.image)
            self.image = newImage
        super(DrawingPad, self).resizeEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.pos()  # Position relative to the widget

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) and self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(self.brushColor, self.brushSize, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.lastPoint, event.pos())  # Draw line to the current position
            self.lastPoint = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def paintEvent(self, event):
        canvasPainter = QPainter(self)
        canvasPainter.drawImage(self.rect(), self.image, self.image.rect())
