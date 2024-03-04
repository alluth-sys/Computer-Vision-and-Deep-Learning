import sys

from PyQt5 import QtCore
from PyQt5.QtWidgets import (QApplication, QGroupBox, QHBoxLayout, QLabel,
                             QPushButton, QVBoxLayout, QWidget)

from Services.BackgroundSubtractionService import BackgroundSubtractionService
from Services.OpticalFlowService import OpticalFlowService
from Services.PCAService import PCAService
from Services.ResNet50Service import ResNet50Service
from Services.VGGService import VGG19Service
from Store.ImageVideoStorer import ImageVideoStorer
from utils.DrawingPad import DrawingPad


class CVDLMainWidget(QWidget):
    def __init__(self):
        super().__init__()
        
        self.image_video_storer = ImageVideoStorer()
        self.background_subtraction_service = BackgroundSubtractionService(self.image_video_storer)
        self.pca_service = PCAService(self.image_video_storer)
        self.optical_flow_service = OpticalFlowService(self.image_video_storer)
        self.mnist_vgg_service = VGG19Service(self.image_video_storer)
        self.drawing_pad = DrawingPad()
        self.resnet50_service = ResNet50Service()
        
        self.setObjectName("MainWindow")
        self.setWindowTitle('cvdl-hw2')
        
        self.loadImageBox = QVBoxLayout()
        self.ImageProcessingBox = QVBoxLayout()
        self.deepLearningBox = QVBoxLayout()
        self.layout = QHBoxLayout()
        
        self.layout.addLayout(self.loadImageBox)
        self.layout.addLayout(self.ImageProcessingBox)
        self.layout.addLayout(self.deepLearningBox)
        self.setLayout(self.layout)
        
        self.load_image_video_ui()
        self.background_subtraction_ui()
        self.optical_flow_ui()
        self.pca_ui()
        self.MNIST_VGG19_ui()
        self.ResNet50_ui()
        
    def load_image_video_ui(self):
        load_image_video_box = QGroupBox(self)
        load_image_button = QPushButton(self, text='Load Image')
        load_video_button = QPushButton(self, text='Load Video')
        load_video_button.clicked.connect(lambda: self.image_video_storer.store_video())
        load_image_button.clicked.connect(lambda: self.image_video_storer.store_image())
        
        vbox = QVBoxLayout()
        vbox.addWidget(load_image_button)
        vbox.addWidget(load_video_button)
        
        load_image_video_box.setLayout(vbox)
        self.loadImageBox.addWidget(load_image_video_box)
    
    def background_subtraction_ui(self):
        background_subtraction_box = QGroupBox(self, title='1.Background Subtraction')
        background_subtraction_button = QPushButton(self, text='1.Background Subtraction')
        background_subtraction_button.clicked.connect(self.background_subtraction_service.background_subtraction)
        
        vbox = QVBoxLayout()
        vbox.addWidget(background_subtraction_button)
        
        background_subtraction_box.setLayout(vbox)
        self.ImageProcessingBox.addWidget(background_subtraction_box)
    
    def optical_flow_ui(self):
        optical_flow_box = QGroupBox(self, title='2.Optical Flow')
        preprocessing_button = QPushButton(self, text='2.1 Preprocessing')
        preprocessing_button.clicked.connect(self.optical_flow_service.find_good_features_to_track)
        video_tracking_button = QPushButton(self, text='2.2 Video Tracking')
        video_tracking_button.clicked.connect(self.optical_flow_service.draw_trajectory)
        
        vbox = QVBoxLayout()
        vbox.addWidget(preprocessing_button)
        vbox.addWidget(video_tracking_button)
        optical_flow_box.setLayout(vbox)
        self.ImageProcessingBox.addWidget(optical_flow_box)
    
    def pca_ui(self):
        pca_box = QGroupBox(self, title='3.PCA')
        dimension_reduction_button = QPushButton(self, text='3.Dimension Reduction')
        dimension_reduction_button.clicked.connect(self.pca_service.apply)
        
        vbox = QVBoxLayout()
        vbox.addWidget(dimension_reduction_button)
        pca_box.setLayout(vbox)
        self.ImageProcessingBox.addWidget(pca_box)
    
    def MNIST_VGG19_ui(self):
        MINST_VGG19_box = QGroupBox(self, title='4.MNIST_VGG19')
        show_model_structure_button = QPushButton(self, text='1.Show Model Structure')
        show_model_structure_button.clicked.connect(self.mnist_vgg_service.showStructure)
        show_model_accuracy_button = QPushButton(self, text='2.Show Accuracy and Loss')
        predict_button = QPushButton(self, text='3.Predict')
        
        reset_button = QPushButton(self, text='4.Reset')
        result_label = QLabel('Predict = ')
        predict_button.clicked.connect(lambda: self.mnist_vgg_service.inferenceModel(
            inferenceImage=self.drawing_pad.get_pil_image(),
            resultField=result_label
        ))
        reset_result_text = lambda: result_label.setText('Predict = ')
        reset_button.clicked.connect(lambda: (self.drawing_pad.clearDrawing(), reset_result_text()))
        show_model_accuracy_button.clicked.connect(lambda: self.drawing_pad.displayImage('./Store/vgg19_accuracy_loss.png'))
        
        hbox = QHBoxLayout()
        buttonVBox = QVBoxLayout()
        buttonVBox.addWidget(show_model_structure_button)
        buttonVBox.addWidget(show_model_accuracy_button)
        buttonVBox.addWidget(predict_button)
        buttonVBox.addWidget(reset_button)
        buttonVBox.addWidget(result_label)
        
        drawPadBox = QVBoxLayout()
        drawPadBox.addWidget(self.drawing_pad)
        
        hbox.addLayout(buttonVBox)
        hbox.addLayout(drawPadBox)
        MINST_VGG19_box.setLayout(hbox)
        self.deepLearningBox.addWidget(MINST_VGG19_box)
    
    def ResNet50_ui(self):
        ResNet50_box = QGroupBox(self, title='5.ResNet50')
        
        image_label = QLabel()
        load_image_button = QPushButton(self, text='Load Image')
        load_image_button.clicked.connect(lambda: self.resnet50_service.load_image(image_label=image_label))
        
        show_image_button = QPushButton(self, text='5.1 Show Image')
        show_image_button.clicked.connect(lambda: self.resnet50_service.show_images())
        show_model_structure_button = QPushButton(self, text='5.2 Show Model Structure')
        show_model_structure_button.clicked.connect(lambda: self.resnet50_service.show_model_structure())
        show_comparison_button = QPushButton(self, text='5.3 Show Comparison')
        show_comparison_button.clicked.connect(lambda: self.resnet50_service.show_accuracy_figure())
        
        predict_label = QLabel('Predict: ')
        inference_button = QPushButton(self, text='5.4 Inference')
        inference_button.clicked.connect(lambda: self.resnet50_service.inference(pred_label=predict_label))
        
        hbox = QHBoxLayout()
        buttonVBox = QVBoxLayout()
        buttonVBox.addWidget(load_image_button)
        buttonVBox.addWidget(show_image_button)
        buttonVBox.addWidget(show_model_structure_button)
        buttonVBox.addWidget(show_comparison_button)
        buttonVBox.addWidget(inference_button)
        
        inferenceResultBox = QVBoxLayout()
        
        image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        image_label.setText("Inference Image")
        image_label.setFixedSize(224,224)
        image_label.setStyleSheet("""
            QLabel {
                background-color: gray;
            }
        """)
        
        inferenceResultBox.addWidget(image_label)
        inferenceResultBox.addWidget(predict_label)
        
        hbox.addLayout(buttonVBox)
        hbox.addLayout(inferenceResultBox)
        ResNet50_box.setLayout(hbox)
        self.deepLearningBox.addWidget(ResNet50_box)
            
if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = CVDLMainWidget()
    MainWindow.show()
    sys.exit(app.exec_())