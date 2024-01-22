import torch


def get_accuracy(output, target):
    preds = torch.argmax(output, dim=1)  # get the highest probability prediction for each item in the batch
    try:
        target_indices = torch.argmax(target, dim=1)  # convert one-hot encoded target to class indices
        correct = (preds == target_indices).sum().item()
        acc = correct / len(target)
    except:
        correct = (preds == target).sum().item()
        acc = correct / len(target)
    return acc
