import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.models as models
from PIL.Image import Image
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QLabel
from torchsummary import summary
from torchvision import transforms

from Store.ImageVideoStorer import ImageVideoStorer


class VGG19Service:
    def __init__(self,image_storer:ImageVideoStorer) -> None:
        self.imageStorer = image_storer

    def showStructure(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.vgg19_bn(pretrained=False, num_classes=10).to(device)
        summary(model, input_size=(3, 32, 32))
        
    def inferenceModel(self, inferenceImage:Image, resultField:QLabel):
        plt.ion()
        transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
        image_transformed  = transform(inferenceImage).unsqueeze(0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.vgg19_bn(weights=None, num_classes=10)
        checkpoint = torch.load("./model/vgg19_bn_MNIST_1.pth", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            image_transformed  = image_transformed.to(device)
            outputs = model(image_transformed)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            probs = probabilities[0].cpu().numpy()

        classes = [str(i) for i in range(10)]
        x_pos = range(len(classes))
        plt.figure(figsize=(6, 8))
        plt.bar(x_pos, probs, align='center',width=0.5)
        plt.xticks(x_pos, classes, rotation=45)
        plt.title('Probability of each class')
        plt.ylabel('Probability')
        plt.xlabel("Class")

        plt.tight_layout(pad=3.0)
        plt.show()

        index_of_highest_prob = np.argmax(probs)
        class_with_highest_prob = classes[index_of_highest_prob]

        resultField.setText(f'Predict = {class_with_highest_prob}')
        



