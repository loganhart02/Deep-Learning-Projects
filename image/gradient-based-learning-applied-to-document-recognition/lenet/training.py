import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from lenet import Lenet5
from dataset import AnimalClassificationDataset, get_data_loaders, get_transforms

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"



def get_loss_optimizer(model, lr, momentum=0.9):
    mse = nn.CrossEntropyLoss()# nn.MSELoss()
    sgd = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    return mse, sgd


def get_accuracy(output, target):
    preds = torch.argmax(output, dim=1) # get highest prob pred
    correct = (preds == target).sum().item()
    
    acc = correct / len(target)
    return acc


def train_animal_classifier(
    train_csv: str, 
    test_csv: str, 
    img_dir: str, 
    epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    momentum: float = 0.9,
    device = "cuda",
    model_weights = None
):
    """fine-tune on my dataset"""
    writer = SummaryWriter()
    img_transforms = get_transforms()
    train_ds = AnimalClassificationDataset(train_csv, img_dir, transform=img_transforms)
    test_ds = AnimalClassificationDataset(test_csv, img_dir, transform=img_transforms)
    
    train_dl, test_dl = get_data_loaders(train_ds, test_ds, batch_size=batch_size)
    
    net = Lenet5(img_channels=1, num_classes=5, activation="relu")
    net.load_state_dict(torch.load(model_weights))
    net = net.to(device)
    
    criterion, optimizer = get_loss_optimizer(net, learning_rate, momentum)
    
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        running_acc = 0.0
        
        for i, (inputs, labels) in enumerate(train_dl):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            running_acc += get_accuracy(outputs, labels)
        
        epoch_loss = running_loss / len(train_dl)
        epoch_acc = running_acc / len(train_dl)
        
        writer.add_scalar('Training Loss', epoch_loss, epoch)
        writer.add_scalar('Training Accuracy', epoch_acc, epoch)
        
        
        # eval model
        net.eval()
        test_loss = 0.0
        test_acc = 0.0
        best_test_acc = 0.0
        
        with torch.no_grad():
            for inputs, labels in test_dl:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                test_acc += get_accuracy(outputs, labels)
        
        test_loss /= len(test_dl)
        test_acc /= len(test_dl)
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(net.state_dict(), "best_model.pth")
            print(f'Epoch {epoch+1}: New best model saved with accuracy: {test_acc:.4f}')
        
        writer.add_scalar('Test Loss', test_loss, epoch)
        writer.add_scalar('Test Accuracy', test_acc, epoch)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}, '
              f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
    
    writer.close()
    
    
def train_cifar10(
    epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    momentum: float = 0.9,
    device = "cuda"
):
    """I am training base model on this and finetuning on my dataset since my dataset is too small to train from scratch"""
    import torchvision
    transform  = get_transforms()
    writer = SummaryWriter()

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    train_dl = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    test_dl = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    net = Lenet5(img_channels=1, num_classes=len(classes), activation="relu")
    net = net.to(device)
    
    criterion, optimizer = get_loss_optimizer(net, learning_rate, momentum)
    
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        running_acc = 0.0
        
        for i, (inputs, labels) in enumerate(train_dl):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            running_acc += get_accuracy(outputs, labels)
        
        epoch_loss = running_loss / len(train_dl)
        epoch_acc = running_acc / len(train_dl)
        
        writer.add_scalar('Training Loss', epoch_loss, epoch)
        writer.add_scalar('Training Accuracy', epoch_acc, epoch)
        
        
        # eval model
        net.eval()
        test_loss = 0.0
        test_acc = 0.0
        best_test_acc = 0.0
        
        with torch.no_grad():
            for inputs, labels in test_dl:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                test_acc += get_accuracy(outputs, labels)
        
        test_loss /= len(test_dl)
        test_acc /= len(test_dl)
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(net.state_dict(), "models/cifar10_best_model.pth")
            print(f'Epoch {epoch+1}: New best model saved with accuracy: {test_acc:.4f}')
        
        writer.add_scalar('Test Loss', test_loss, epoch)
        writer.add_scalar('Test Accuracy', test_acc, epoch)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}, '
              f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
    
    writer.close()

    
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=str, required=True)
    parser.add_argument("--test-csv", type=str, required=True)
    parser.add_argument("--img-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train-cifar10", type=bool, default=False, help="train cifar10 instead of animal classifier. use to debug/check model")
    
    args = parser.parse_args()
    
    if args.train_cifar10:
        train_cifar10(
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.momentum,
            args.device
        )
    else:
        train_animal_classifier(
            args.train_csv, 
            args.test_csv, 
            args.img_dir, 
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.momentum,
            args.device
        )
    


                
    
    