import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

from EarlyStopping import EarlyStopping


def load_data():
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=4)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                            shuffle=True, num_workers=4)

    return trainloader, testloader


def train(epoch, model, optimizer, trainloader, device, criterion, train_losses, train_accuracies):
    model.train()
    total_loss = 0
    correct = 0
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        pred = outputs.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(trainloader)
    accuracy = 100. * correct / len(trainloader.dataset)
    train_losses.append(avg_loss)
    train_accuracies.append(accuracy)
    print(f"Train Epoch: {epoch}, Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

# Validation function
def validate(epoch, model, testloader, device, criterion, val_losses, val_accuracies):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

    avg_loss = val_loss / len(testloader)
    accuracy = 100. * correct / len(testloader.dataset)
    val_losses.append(avg_loss)
    val_accuracies.append(accuracy)
    print(f"Val Epoch: {epoch}, Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

def buildModel():
    trainloader, testloader = load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.vgg19_bn(weights=None,num_classes=10)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,weight_decay=5e-4)
    # onPlateauLrScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.1,patience=3,verbose=True)
    
    early_stopping = EarlyStopping(patience=7, verbose=True)

    # Lists to store accuracy and loss values
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    print("==Training Starts==")
    num_epochs = 60
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0, verbose=True)
    for epoch in range(1, num_epochs + 1):
        train(epoch, model, optimizer, trainloader, device, criterion,train_losses, train_accuracies)
        validate(epoch, model, testloader, device, criterion,val_losses, val_accuracies)
        scheduler.step() 

        early_stopping(val_losses[-1])
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Save the trained model and metrics
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }, 'vgg19_bn_cifar10_useCosine.pth')

    # Plot training and validation loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')  
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')  
    plt.ylabel('Accuracy(%)')  
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    buildModel()