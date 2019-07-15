import datetime
import sys

import cv2
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, pyqtProperty, pyqtSignal


class VideoCapture(QtCore.QObject):
    frameChanged = pyqtSignal(int)
    gotFrame = pyqtSignal(np.ndarray)
    timestampChanged = pyqtSignal(datetime.timedelta)

    def __init__(self, source, parent=None):
        super().__init__(parent)
        self.source = source
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
        self.read()

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
        self.read()

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

def H(*args, **kwargs):
    kwargs['vertical'] = False
    return V(*args, **kwargs)

def V(*args, widget=None, vertical=True, margins=True):
    layout = QtWidgets.QVBoxLayout() if vertical else QtWidgets.QHBoxLayout()
    for child in args:
        layout.addWidget(child)

    if not margins:
        layout.setContentsMargins(0, 0, 0, 0)

    widget = widget if widget else QtWidgets.QWidget()
    widget.setLayout(layout)
    return widget

def F(*args, widget=None):
    layout = QtWidgets.QFormLayout()
    for child in args:
        layout.addRow(*child)

    widget = widget if widget else QtWidgets.QWidget()
    widget.setLayout(layout)
    return widget


PreviewModes = (
    'Background',
    'Frame',
    'Frame - Background',
    'Threshold',
    'Dilate',
    'Contours',
)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def gotFrame(self, frame):
        print('hey')
        # https://stackoverflow.com/a/55468544/6622587
        rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgbImage.shape
        bytesPerLine = ch * w
        convertToQtFormat = QtGui.QImage(rgbImage.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
        p = convertToQtFormat.scaled(w/2, h/2, Qt.KeepAspectRatio)
        self.frameViewer.setPixmap(QtGui.QPixmap.fromImage(p))

    def newPlaybackPosition(self, td):
        self.message.setText(str(td))

    def setPosition(self, frame):
        print(frame)
        self.cap.frame = frame

    def openVideoFile(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(filter="*.avi")
        if path == '':
            return  # no file selected

        self.cap = VideoCapture(path, self)
        self.cap.gotFrame.connect(self.gotFrame)
        self.cap.read()

    def openClickPointsFile(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(filter="*.cdb")
        if path == '':
            return  # no file selected
        # TODO: use this file somehow

    def computeBackground(self):
        pass

    def initUI(self):
        self.setWindowTitle('HerringRunner')

        self.loadVideoButton = QtWidgets.QPushButton('Load Video')
        self.loadVideoButton.clicked.connect(self.openVideoFile)

        self.loadClickPointsButton = QtWidgets.QPushButton('Load ClickPoints')
        self.loadClickPointsButton.clicked.connect(self.openClickPointsFile)

        self.messageLabel = QtWidgets.QLabel()
        self.messageLabel.setText('00:00:00')

        self.playbackSlider = QtWidgets.QSlider(Qt.Vertical)
        self.playbackSlider.setInvertedAppearance(True)

        # 0 is no blur, n > 0 is 2*(n - 1) + 1
        self.blurSlider = QtWidgets.QSlider(Qt.Horizontal)
        self.blurSlider.valueChanged.connect(lambda n: \
            self.blurSlider.setToolTip(str(0 if n == 0 else 2*(n - 1) + 1)))
        self.blurSlider.setRange(0, 50)
        self.blurSlider.setValue(13)

        self.bgWeightSlider = QtWidgets.QSlider(Qt.Horizontal)
        self.bgWeightSlider.valueChanged.connect(lambda n: \
            self.bgWeightSlider.setToolTip('%i%%' % n))
        self.bgWeightSlider.setRange(0, 100)
        self.bgWeightSlider.setValue(60)

        self.computeBgButton = QtWidgets.QPushButton('Compute Background')
        self.computeBgButton.clicked.connect(self.computeBackground)

        self.brightnessSlider = QtWidgets.QSlider(Qt.Horizontal)
        self.brightnessSlider.valueChanged.connect(lambda n: \
            self.brightnessSlider.setToolTip('off' if n == 0 else str(n)))
        self.brightnessSlider.setRange(0, 255)
        self.brightnessSlider.setValue(5)

        self.dilationSlider = QtWidgets.QSlider(Qt.Horizontal)
        self.dilationSlider.valueChanged.connect(lambda n: \
            self.dilationSlider.setToolTip('off' if n == 0 else str(n)))
        self.dilationSlider.setRange(0, 50)
        self.dilationSlider.setValue(2)

        self.detectionAreaSlider = QtWidgets.QSlider(Qt.Horizontal)
        self.detectionAreaSlider.valueChanged.connect(lambda n: \
            self.detectionAreaSlider.setToolTip('%i%%' % n))
        self.detectionAreaSlider.setRange(0, 100)
        self.detectionAreaSlider.setValue(21)

        self.detectionAreaLabel = QtWidgets.QLabel('0.00%')

        self.previewModeLabel = QtWidgets.QLabel()
        self.previewModeLabel.setAlignment(Qt.AlignHCenter)

        self.previewModeSlider = QtWidgets.QSlider(Qt.Horizontal)
        self.previewModeSlider.valueChanged.connect(lambda n: \
            self.previewModeLabel.setText('Displaying ' + PreviewModes[n]))
        self.previewModeSlider.setRange(0, len(PreviewModes) - 1)
        self.previewModeSlider.setValue(1)

        self.clickpointsCheckBox = QtWidgets.QCheckBox('Show ClickPoints')

        self.frameViewer = QtWidgets.QLabel()
        self.frameViewer.setFrameStyle(QtWidgets.QFrame.Box)
        self.frameViewer.setStyleSheet('border: 8px solid red')

        mainWidget = V(
            H(
                H(
                    self.playbackSlider,
                    self.frameViewer,
                ),
                V(
                    H(
                        self.loadVideoButton,
                        self.loadClickPointsButton,
                    ),
                    F(
                        ('Blur Radius', self.blurSlider),
                        ('Current Frame Weight', self.bgWeightSlider),
                        (self.computeBgButton,),
                        widget=QtWidgets.QGroupBox('Background')
                    ),
                    F(
                        ('Brightness Threshold', self.brightnessSlider),
                        ('Dilation', self.dilationSlider),
                        ('Detection Area Threshold', self.detectionAreaSlider),
                        ('Current detection area:', self.detectionAreaLabel),
                        widget=QtWidgets.QGroupBox('Detection')
                    ),
                    F(
                        (self.previewModeSlider,),
                        (self.previewModeLabel,),
                        (self.clickpointsCheckBox,),
                        widget=QtWidgets.QGroupBox('Preview')
                    )
                )
            ),
            self.messageLabel
        )
        self.setCentralWidget(mainWidget)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    player = MainWindow()
    player.resize(640, 480)
    player.show()
    sys.exit(app.exec_())
