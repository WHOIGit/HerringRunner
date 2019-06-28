import datetime
import sys

import cv2
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, pyqtProperty, pyqtSignal, pyqtSlot


class VideoCapture(QtCore.QObject):
    frameChanged = pyqtSignal(int)
    gotFrame = pyqtSignal(np.ndarray)
    timestampChanged = pyqtSignal(datetime.timedelta)

    def __init__(self, source, parent=None):
        super().__init__(parent)
        self.cap = cv2.VideoCapture(source)
    
    @pyqtProperty(int)
    def nframes(self):
        return self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    @pyqtProperty(int)
    def frame(self):
        return self.cap.get(cv2.CAP_PROP_POS_FRAMES)
    
    @frame.setter
    def frame(self, n):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        self.frameChanged.emit(n)

    @pyqtProperty(int)
    def framerate(self):
        return self.cap.get(cv2.CAP_PROP_FPS)

    @pyqtProperty(int)
    def height(self):
        return self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    @pyqtProperty(datetime.timedelta)
    def timestamp(self):
        msec = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        return datetime.timedelta(milliseconds=msec)
    
    @timestamp.setter
    def timestamp(self, td):
        msec = td / timedelta(milliseconds=1)
        self.cap.set(cv2.CAP_PROP_POS_MSEC, msec)
        self.timestampChanged.emit(td)

    @pyqtProperty(int)
    def width(self):
        return self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    def read(self):
        success, frame = self.cap.read()
        if not success:
            raise Exception()  # TODO: Figure out how error handling works
        self.frameChanged.emit(self.frame)
        self.gotFrame.emit(frame)
        self.timestampChanged.emit(self.timestamp)


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
        #p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
        self.changePixmap.emit(convertToQtFormat)

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

    @pyqtSlot(datetime.timedelta)
    def newPlaybackPosition(self, td):
        self.message.setText(str(td))

    def setPosition(self, frame):
        print(frame)
        self.cap.frame = frame

    def initUI(self):
        self.setWindowTitle('HerringRunner')

        self.cap = VideoCapture('./20170701145052891.avi', self)
        self.cap.frame = 100

        mainWidget = QtWidgets.QWidget(self)

        layout = QtWidgets.QVBoxLayout()

        self.message = QtWidgets.QLabel(self)
        self.message.setText('hello')
        layout.addWidget(self.message)

        self.positionSlider = QtWidgets.QSlider(Qt.Horizontal, self)
        self.positionSlider.setRange(0, self.cap.nframes)
        self.positionSlider.sliderMoved.connect(self.setPosition)
        layout.addWidget(self.positionSlider)

        self.label = QtWidgets.QLabel(self)
        self.label.resize(self.cap.width, self.cap.height)
        layout.addWidget(self.label)

        mainWidget.setLayout(layout)
        self.setCentralWidget(mainWidget)

        self.cap.frameChanged.connect(self.positionSlider.setValue)
        self.cap.timestampChanged.connect(self.newPlaybackPosition)

        th = VideoCaptureThread(self.cap, parent=self)
        th.changePixmap.connect(self.setImage)
        th.start()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    player = MainWindow()
    player.resize(640, 480)
    player.show()
    sys.exit(app.exec_())
