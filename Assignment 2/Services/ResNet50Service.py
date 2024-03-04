
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from torchsummary import summary
from torchvision import datasets, transforms


class ResNet50Service():
    
    def __init__(self) -> None:
        self.image = None
        self.model = None
    
    def show_images(self):
        plt.ion()
        dataset_path = './inference_dataset'
        transform = transforms.Compose([
            transforms.Resize((224)),
            transforms.ToTensor()
        ])
        
        inference_dataset = datasets.ImageFolder(dataset_path, transform=transform)
        
        class_images = {}
        for img, label in inference_dataset:
            if label not in class_images:
                class_images[label] = img
                if len(class_images) == len(inference_dataset.classes):
                    break

        plt.figure(figsize=(10, 5))
        for i, (label, img) in enumerate(class_images.items()):
            ax = plt.subplot(1, len(class_images), i + 1)
            plt.imshow(img.permute(1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
            ax.set_title(f"Class: {inference_dataset.classes[label]}")
            plt.axis("off")
        plt.show()
        
    def show_model_structure(self):
        model = models.resnet50(pretrained=True)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        nr_filters = model.fc.in_features  #number of input features of last layer
        model.fc = nn.Sequential(
            nn.Linear(nr_filters, 1), 
            nn.Sigmoid()
        )
        
        model = model.to(device)
        
        summary(model, (3, 224, 224))
        
    def load_image(self, image_label: QtWidgets.QLabel):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.jpg *.png)")
        
        if fname:
            self.image = Image.open(fname)
            pixmap = QPixmap(fname)
            image_label.setPixmap(pixmap.scaled(224, 224, Qt.AspectRatioMode.KeepAspectRatio))
        
    def inference(self, pred_label: QtWidgets.QLabel):
        if self.image is None:
            return
        
        # Resize and transform the image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        image = transform(self.image).unsqueeze(0)
        
        if self.model is None:
            self.model = models.resnet50()
            nr_filters = self.model.fc.in_features
            self.model.fc = nn.Linear(nr_filters, 1)
            model_state = torch.load('./model/ResNet50_With_no_erasing_1.pth', map_location=torch.device('cpu'))
            self.model.load_state_dict(model_state)
            self.model.eval()

        # Run inference
        with torch.no_grad():
            output = self.model(image)
            predicted_class = 'Cat' if torch.sigmoid(output).item() < 0.5 else 'Dog'
            pred_label.setText(f'Prediction: {predicted_class}')
            
    def show_accuracy_figure(self):
        image_path = './Store/resnet_figure.png'
        img = Image.open(image_path)
        img_array = np.array(img)
        
        plt.imshow(img_array)
        plt.axis('off')
        plt.show()      