import torch
import torch.nn as nn
import torch.optim as optim


def get_loss_optimizer(model, lr, momentum=0.9, weight_decay=0.0005):
    f_loss = nn.CrossEntropyLoss()
    sgd = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    return f_loss, sgd


def get_accuracy(output, target):
    preds = torch.argmax(output, dim=1)  # get the highest probability prediction for each item in the batch
    target_indices = torch.argmax(target, dim=1)  # convert one-hot encoded target to class indices
    correct = (preds == target_indices).sum().item()
    acc = correct / len(target)
    return acc






