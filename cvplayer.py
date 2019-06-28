import datetime
import sys

import cv2
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot


class VideoCapture(QtCore.QObject):
    gotFrame = pyqtSignal(np.ndarray)

    def __init__(self, source, parent=None):
        super().__init__(parent)
        self.cap = cv2.VideoCapture(source)

    @property
    def framerate(self):
        return self.cap.get(cv2.CAP_PROP_FPS)

    @property
    def height(self):
        return self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    @property
    def timestamp(self):
        msec = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        return datetime.timedelta(milliseconds=msec)
    
    @timestamp.setter
    def timestamp_setter(self, td):
        msec = td / timedelta(milliseconds=1)
        self.cap.set(cv2.CAP_PROP_POS_MSEC, msec)

    @property
    def width(self):
        return self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    def read(self):
        success, frame = self.cap.read()
        if not success:
            raise Exception()  # TODO: Figure out how error handling works
        self.gotFrame.emit(frame)


class VideoCaptureThread(QtCore.QThread):
    changePixmap = pyqtSignal(QtGui.QImage)

    def __init__(self, cap, parent=None):
        super().__init__(parent)
        self.cap = cap
        self.cap.gotFrame.connect(self.gotFrame)

    @pyqtSlot(np.ndarray)
    def gotFrame(self, frame):
        # https://stackoverflow.com/a/55468544/6622587
        rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgbImage.shape
        bytesPerLine = ch * w
        convertToQtFormat = QtGui.QImage(rgbImage.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
        p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
        self.changePixmap.emit(p)

    def run(self):
        while True:
            self.cap.read()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    @pyqtSlot(QtGui.QImage)
    def setImage(self, image):
        self.label.setPixmap(QtGui.QPixmap.fromImage(image))

    def initUI(self):
        self.setWindowTitle('HerringRunner')

        layout = QtWidgets.QVBoxLayout()
        self.label = QtWidgets.QLabel(self)
        self.label.resize(640, 480)
        layout.addWidget(self.label)
        self.setLayout(layout)

        # create a label
        cap = VideoCapture('./20170701145052891.avi')
        th = VideoCaptureThread(cap, parent=self)
        th.changePixmap.connect(self.setImage)
        th.start()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    player = MainWindow()
    player.resize(640, 480)
    player.show()
    sys.exit(app.exec_())
