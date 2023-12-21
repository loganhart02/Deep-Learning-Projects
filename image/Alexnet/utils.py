import torch.nn as nn
import torch.optim as optim


def get_loss_optimizer(model, lr, momentum=0.9, weight_decay=0.0005):
    f_loss = nn.CrossEntropyLoss()
    sgd = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    return f_loss, sgd
