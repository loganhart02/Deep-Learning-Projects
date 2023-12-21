import torch


def get_accuracy(output, target):
    preds = torch.argmax(output, dim=1) # get highest prob pred
    correct = (preds == target).sum().item()
    
    acc = correct / len(target)
    return acc