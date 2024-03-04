import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets


def load_data(use_erasing: bool):
    transform_list = [
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ]
    
    if use_erasing:
        transform_list.append(transforms.RandomErasing())

    transform = transforms.Compose(transform_list)

    dataset_path = './Dataset_Cvdl_Hw2_Q5/dataset'
    # Load dataset
    trainset = datasets.ImageFolder(os.path.join(dataset_path,'training_dataset'), transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=4)
    
    testset = datasets.ImageFolder(os.path.join(dataset_path,'validation_dataset'), transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=4)

    return trainloader, testloader

def make_train_step(model, optimizer, loss_fn):
    def train_step(x, y):
        # Make prediction
        yhat = model(x)
        # Enter train mode
        model.train()
        # Compute loss
        loss = loss_fn(yhat, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return loss
    return train_step

def calculate_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.unsqueeze(1).float().to(device)
            outputs = model(inputs)
            predicted = torch.round(torch.sigmoid(outputs))  # Apply sigmoid and round to get predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def buildModel(use_erasing: bool):
    trainloader, testloader = load_data(use_erasing)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=True)
    
    for params in model.parameters():
        params.requires_grad_ = False
        
    nr_filters = model.fc.in_features  # Number of input features of the last layer
    model.fc = nn.Linear(nr_filters, 1)
    
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.fc.parameters())
    train_step = make_train_step(model, optimizer, criterion)
    scheduler = ReduceLROnPlateau(optimizer, patience=3,verbose=True)

    print(f"==Training Starts== {'With' if use_erasing else 'Without'} Random Erasing")
    
    losses = []
    val_losses = []
    epoch_train_losses = []
    epoch_test_losses = []

    n_epochs = 30
    early_stopping_tolerance = 7
    early_stopping_threshold = 0.03
    
    best_model_wts = None  # Store the best model weights

    train_accuracies = []
    val_accuracies = []

    for epoch in range(n_epochs):
        epoch_loss = 0
        model.train()
        for data in trainloader:
            x_batch, y_batch = data
            x_batch = x_batch.to(device)
            y_batch = y_batch.unsqueeze(1).float()
            y_batch = y_batch.to(device)

            loss = train_step(x_batch, y_batch)
            epoch_loss += loss.item() / len(trainloader)
            losses.append(loss)
        
        epoch_train_losses.append(epoch_loss)
        train_accuracy = calculate_accuracy(model, trainloader, device)
        
        print(f'Epoch: {epoch+1}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')

        # Validation
        model.eval()
        cum_loss = 0
        for x_batch, y_batch in testloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.unsqueeze(1).float()
            y_batch = y_batch.to(device)

            yhat = model(x_batch)
            val_loss = criterion(yhat, y_batch)
            cum_loss += val_loss.item() / len(testloader)
            val_losses.append(val_loss.item())

        epoch_test_losses.append(cum_loss)
        val_accuracy = calculate_accuracy(model, testloader, device)
        scheduler.step(cum_loss)

        print(f'Epoch: {epoch+1}, Val Loss: {cum_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
        best_loss = min(epoch_test_losses)
        
        # Save the best model
        if cum_loss <= best_loss:
            best_model_wts = model.state_dict()

        # Early stopping
        if cum_loss > best_loss:
            early_stopping_counter +=1
        else:
            early_stopping_counter = 0

        if (early_stopping_counter == early_stopping_tolerance) or (best_loss <= early_stopping_threshold):
            print("\nTerminating: early stopping")
            break
        
        # Append accuracies
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

    # Save the best model's weights
    model.load_state_dict(best_model_wts)
    model_filename = f"ResNet50_With{'_erasing' if use_erasing else '_no_erasing'}_1.pth"
    torch.save(model.state_dict(), model_filename)

    return max(train_accuracies), max(val_accuracies)

if __name__ == '__main__':
    accuracy_with_erasing, val_accuracy_with_erasing = buildModel(use_erasing=True)
    accuracy_without_erasing, val_accuracy_without_erasing = buildModel(use_erasing=False)

    # Data to plot
    labels = ['Without Random Erasing', 'With Random Erasing']
    accuracies = [accuracy_without_erasing, accuracy_with_erasing]

    # Create bar chart
    plt.figure(figsize=(10, 6))  # Adjust the figure size to your preference
    plt.bar(labels, accuracies, color=['blue', 'blue'])

    # Add the text for the labels, title, and axes ticks
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Comparison')

    # Adding the text on top of each bar
    for i, v in enumerate(accuracies):
        plt.text(i, v + 1, str(v), color='black', ha='center')

    plt.tight_layout()
    plt.show()
