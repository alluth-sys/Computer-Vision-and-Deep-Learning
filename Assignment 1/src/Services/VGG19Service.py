import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.models as models
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QLabel
from torchsummary import summary
from torchvision import transforms

from Store.ImageStorer import ImageStorer


class VGG19Service:
    def __init__(self,image_storer:ImageStorer) -> None:
        self.imageStorer = image_storer

    def showAugmentedPictures(self):
        images = self.imageStorer.load_augmentation_images()
        plt.ion()

        rand_horizontal_flip = transforms.RandomHorizontalFlip(p=1)
        perspective_transformer = transforms.RandomPerspective(distortion_scale=0.6, p=1.0)
        rotater = transforms.RandomRotation(degrees=(0, 180))
        affine_transfomer = transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3))
        posterizer = transforms.RandomPosterize(bits=2)

        transform_functions = [rand_horizontal_flip,perspective_transformer,rotater,affine_transfomer,posterizer]

        transformed_images = []

        for img in images:
            flipped_img = random.choice(transform_functions)(img)
            transformed_images.append(flipped_img)

        fig, axes = plt.subplots(3,3)
        for ax, img in zip(axes.ravel(), transformed_images):
            ax.imshow(img)  

        plt.tight_layout()

    def showStructure(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.vgg19_bn(pretrained=False, num_classes=10).to(device)
        summary(model, input_size=(3, 32, 32))

    def showAccuracyLoss(self):
        img = self.imageStorer.load_accuracy_loss_image()
        
        cv2.imshow('Accuracy and Loss', img)
        cv2.waitKey()
        cv2.destroyAllWindows()
        
    def inferenceModel(self, resultField:QLabel, imageBox: QLabel):
        plt.ion()
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        image = self.imageStorer.get_inference_image()
        image_transformed  = transform(image).unsqueeze(0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.vgg19_bn(weights=None, num_classes=10)
        checkpoint = torch.load("./vgg19_bn_cifar10.pth", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            image_transformed  = image_transformed.to(device)
            outputs = model(image_transformed)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            probs = probabilities[0].cpu().numpy()

        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
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
        pixmap = QPixmap(self.imageStorer.get_inference_image_path())
        scaled_pixmap = pixmap.scaled(128, 128, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        imageBox.setPixmap(scaled_pixmap)


