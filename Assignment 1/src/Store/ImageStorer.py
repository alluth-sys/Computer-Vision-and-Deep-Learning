import os

import cv2
from PIL import Image
from PyQt5 import QtWidgets


class ImageStorer:

    def __init__(self):
        self.currentDir = ""
        self.imageLDir = ""
        self.imageRDir = ""
        self.SIFTImage1 = ""
        self.SIFTImage2 = ""
        self.inferenceImagePath = ""

    def load_image_folder(self):
        folderPath = QtWidgets.QFileDialog.getExistingDirectory()
        self.currentDir = folderPath
    
    def get_current_folder(self)->str:
        return self.currentDir
    
    def load_images(self)->list:
        images = []
        folder = self.currentDir

        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder,filename))
            if img is not None:
                images.append(img)
        
        return images
    
    def load_image_by_name(self,name:str):
        folder = self.currentDir
        filename = name + '.bmp'

        img = cv2.imread(os.path.join(folder,filename))

        return img

    def set_imageR(self):
        filePath,_ = QtWidgets.QFileDialog.getOpenFileName()
        self.imageRDir = filePath

    def set_imageL(self):
        filePath,_ = QtWidgets.QFileDialog.getOpenFileName()
        self.imageLDir = filePath

    def load_imageR(self):
        if self.imageRDir == "":
            self.imageRDir = "./Dataset_CvDl_Hw1/Q3_image/imR.png"

        return cv2.imread(self.imageRDir)
    
    def load_imageL(self):
        if self.imageLDir == "":
            self.imageLDir = "./Dataset_CvDl_Hw1/Q3_image/imL.png"
        return cv2.imread(self.imageLDir)
    
    def set_sift_image1(self):
        filePath,_ = QtWidgets.QFileDialog.getOpenFileName()
        self.SIFTImage1 = filePath

    def set_sift_image2(self):
        filePath,_ = QtWidgets.QFileDialog.getOpenFileName()
        self.SIFTImage2 = filePath
    
    def load_sift_image1(self):
        return cv2.imread(self.SIFTImage1)
    
    def load_sift_image2(self):
        return cv2.imread(self.SIFTImage2)
    
    def load_inference_image(self):
        filePath,_ = QtWidgets.QFileDialog.getOpenFileName()
        self.inferenceImagePath = filePath
    
    def get_inference_image(self):
        img = Image.open(self.inferenceImagePath)
        return img
    
    def get_inference_image_path(self):
        return self.inferenceImagePath

    def load_augmentation_images(self)->list[Image.Image]:
        images = []
        path = "./Dataset_CvDl_Hw1/Q5_image/Q5_1"

        for filename in os.listdir(path):
            img = Image.open(os.path.join(path,filename))
            images.append(img)
        return images
    
    def load_accuracy_loss_image(self):
        script_dir = os.path.dirname(__file__)
        image_path = os.path.join(script_dir, "accuracy_loss_chart.png")
        img = cv2.imread(image_path)
        return img
