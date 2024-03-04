import sys

from PyQt5 import QtCore
from PyQt5.QtWidgets import (QApplication, QComboBox, QGroupBox, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QSizePolicy,
                             QVBoxLayout, QWidget)

from Services.AugmentedRealityService import AugmentedRealityService
from Services.CalibrationService import CalibrationService
from Services.SIFTService import SIFTService
from Services.StereoDisparityService import StereoDisparityService
from Services.VGG19Service import VGG19Service
from Store.ImageStorer import ImageStorer


class MyWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.image_storer = ImageStorer()
        self.calibrationService = CalibrationService(image_storer=self.image_storer)
        self.steroDisparityService = StereoDisparityService(image_storer=self.image_storer)
        self.siftService = SIFTService(image_storer=self.image_storer)
        self.augmentedRealityService = AugmentedRealityService(image_storer=self.image_storer)
        self.vgg19Service = VGG19Service(image_storer=self.image_storer)
        
        self.setObjectName("MainWindow")
        self.setWindowTitle('cvdl-hw1')
        self.firstRow = QHBoxLayout()
        self.secondRow = QHBoxLayout()
        self.secondRow.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        
        self.layout = QVBoxLayout()
        self.layout.addLayout(self.firstRow)
        self.layout.addLayout(self.secondRow)
        self.setLayout(self.layout)

        self.load_image_ui()
        self.load_calibration_ui()
        self.load_augmented_reality_ui()
        self.load_stereo_disparity_map_ui()
        self.load_SIFT_ui()
        self.loadVGG19_ui()

    def load_image_ui(self):
        loadImageBox = QGroupBox(self)
        loadImageBox.setTitle("Load Image")

        loadFolderbutton = QPushButton(self)
        loadFolderbutton.setText("Load folder")
        loadFolderbutton.clicked.connect(self.image_storer.load_image_folder)

        loadImageLButton = QPushButton(self)
        loadImageLButton.setText("Load Image_L")
        loadImageLButton.clicked.connect(self.image_storer.set_imageL)

        loadImageRButton = QPushButton(self)
        loadImageRButton.setText("Load Image_R")
        loadImageRButton.clicked.connect(self.image_storer.set_imageR)
        
        vbox = QVBoxLayout()
        vbox.addWidget(loadFolderbutton)
        vbox.addWidget(loadImageLButton)
        vbox.addWidget(loadImageRButton)
        
        loadImageBox.setLayout(vbox)
        self.firstRow.addWidget(loadImageBox)
    
    def load_calibration_ui(self):
        loadCalibrationBox = QGroupBox(self)
        loadCalibrationBox .setTitle("1.Calibration")

        findCornerButton = QPushButton(self)
        findCornerButton.setText("1.1 Find Corners")
        findCornerButton.clicked.connect(self.calibrationService.find_corners)

        findInstrinsicButton = QPushButton(self)
        findInstrinsicButton.setText("1.2 Find Intrinsic")
        findInstrinsicButton.clicked.connect(self.calibrationService.find_intrinsic)

        findExtrinsicBox = QGroupBox()
        findExtrinsicBox.setTitle("1.3 Find extrinsic")
        extrinsicOptions = QComboBox()
        extrinsicOptions.addItems([str(i) for i in range(1, 16)])
        extrinsicOptions.setFixedWidth(extrinsicOptions.sizeHint().width())
        

        findExtrinsicButton = QPushButton(self)
        findExtrinsicButton.setText("1.3 Find extrinsic")
        findExtrinsicButton.clicked.connect(lambda: self.calibrationService.find_extrinsic(extrinsicOptions.currentText()))

        fvbox = QVBoxLayout()
        fvbox.addWidget(extrinsicOptions)
        fvbox.setAlignment(extrinsicOptions,QtCore.Qt.AlignmentFlag.AlignCenter)
        
        fvbox.addWidget(findExtrinsicButton)
        findExtrinsicBox.setLayout(fvbox)

        findDistrotionButton = QPushButton(self)
        findDistrotionButton.setText("1.4 Find distortion")
        findDistrotionButton.clicked.connect(self.calibrationService.find_distortion)

        showResultButton = QPushButton(self)
        showResultButton.setText("1.5 Show result")
        showResultButton.clicked.connect(self.calibrationService.show_result)

        vbox = QVBoxLayout()
        vbox.addWidget(findCornerButton)
        vbox.addWidget(findInstrinsicButton)
        vbox.addWidget(findExtrinsicBox)
        vbox.addWidget(findDistrotionButton)
        vbox.addWidget(showResultButton)
        
        loadCalibrationBox.setLayout(vbox)
        self.firstRow.addWidget(loadCalibrationBox)
    
    def load_augmented_reality_ui(self):
        augmentedRealityBox = QGroupBox(self)
        augmentedRealityBox.setTitle("2. Augmented Reality")

        wordInputField = QLineEdit(self)
        wordInputField.setMaxLength(6)
        showWordOnBoardButton = QPushButton(self)
        showWordOnBoardButton.setText("2.1 show words on board")
        showWordOnBoardButton.clicked.connect(lambda: self.augmentedRealityService.showWordsOnBoard(wordInputField.text(),mode="ONBOARD"))

        showWordVerticalButton = QPushButton(self)
        showWordVerticalButton.clicked.connect(lambda: self.augmentedRealityService.showWordsOnBoard(wordInputField.text(),mode="VERTICAL"))
        showWordVerticalButton.setText("2.2 show words vertical")

        vbox = QVBoxLayout()
        vbox.addWidget(wordInputField)
        vbox.addWidget(showWordOnBoardButton)
        vbox.addWidget(showWordVerticalButton)

        augmentedRealityBox.setLayout(vbox)
        self.firstRow.addWidget(augmentedRealityBox)

    def load_stereo_disparity_map_ui(self):
        stereoDisparityBox = QGroupBox(self)
        stereoDisparityBox.setTitle("3. Stereo disparity map")
        
        stereoDisparityButton = QPushButton(self)
        stereoDisparityButton.setText("3.1 stereo dispartiy map")
        stereoDisparityButton.clicked.connect(self.steroDisparityService.showStereoDisparityMap)

        vbox = QVBoxLayout()
        vbox.addWidget(stereoDisparityButton)

        stereoDisparityBox.setLayout(vbox)
        self.firstRow.addWidget(stereoDisparityBox)

    def load_SIFT_ui(self):
        SIFTBox = QGroupBox(self)
        SIFTBox.setTitle("4. SIFT")
        SIFTBox.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        loadImage1Btn = QPushButton(self)
        loadImage1Btn.clicked.connect(self.image_storer.set_sift_image1)
        loadImage2Btn = QPushButton(self)
        loadImage2Btn.clicked.connect(self.image_storer.set_sift_image2)
        keypointsBtn = QPushButton(self)
        matchedKeypointsBtn = QPushButton(self)

        loadImage1Btn.setText("Load Image 1")
        loadImage2Btn.setText("Load Image 2")
        keypointsBtn.setText("4.1 Keypoints")
        keypointsBtn.clicked.connect(self.siftService.showKeypoints)
        matchedKeypointsBtn.setText("4.2 Matched Keypoints")
        matchedKeypointsBtn.clicked.connect(self.siftService.findMatchedKeypoints)

        vbox = QVBoxLayout()
        vbox.addWidget(loadImage1Btn)
        vbox.addWidget(loadImage2Btn)
        vbox.addWidget(keypointsBtn)
        vbox.addWidget(matchedKeypointsBtn)

        SIFTBox.setLayout(vbox)
        self.secondRow.addWidget(SIFTBox,alignment=QtCore.Qt.AlignmentFlag.AlignTop)

    def loadVGG19_ui(self):
        vgg19box = QGroupBox(self)
        vgg19box.setTitle("5. VGG19")
        vgg19box.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        loadImgBtn = QPushButton(self)
        loadImgBtn.setText("Load Image")
        loadImgBtn.clicked.connect(self.image_storer.load_inference_image)

        showAugImgBtn = QPushButton(self)
        showAugImgBtn.setText("5.1 Show Augmented Images")
        showAugImgBtn.clicked.connect(self.vgg19Service.showAugmentedPictures)

        showModelSturctBtn = QPushButton(self)
        showModelSturctBtn.setText("5.2 Show Model Structure")
        showModelSturctBtn.clicked.connect(self.vgg19Service.showStructure)

        showAccAndLossBtn = QPushButton(self)
        showAccAndLossBtn.setText("5.3 Show Acc and Loss")
        showAccAndLossBtn.clicked.connect(self.vgg19Service.showAccuracyLoss)

        predictLabel = QLabel(self)
        predictLabel.setText("Predict =")

        predictImageBox = QLabel(self)
        predictImageBox.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        predictImageBox.setText("Inference Image")
        predictImageBox.setFixedSize(128,128)
        predictImageBox.setStyleSheet("""
            QLabel {
                border: 2px solid black;
                background-color: white;
            }
        """)

        inferenceBtn = QPushButton(self)
        inferenceBtn.setText("5.4 Inference")
        inferenceBtn.clicked.connect(lambda: self.vgg19Service.inferenceModel(predictLabel,predictImageBox))

        vbox = QVBoxLayout()
        vbox.addWidget(loadImgBtn)
        vbox.addWidget(showAugImgBtn)
        vbox.addWidget(showModelSturctBtn)
        vbox.addWidget(showAccAndLossBtn)
        vbox.addWidget(inferenceBtn)
        vbox.addWidget(predictLabel)
        vbox.addWidget(predictImageBox)

        vgg19box.setLayout(vbox)
        self.secondRow.addWidget(vgg19box)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = MyWidget()
    MainWindow.show()
    sys.exit(app.exec_())