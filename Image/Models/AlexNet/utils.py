import torch.nn as nn
import torch.optim as optim


def get_loss_optimizer(model, lr, device, momentum=0.9, weight_decay=0.0005):
    f_loss = nn.CrossEntropyLoss().to(device)
    sgd = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(sgd, step_size=30, gamma=0.1)
    return f_loss, sgd, scheduler
