import torch.nn as nn
import torch.optim as optim

def get_loss_optimizer(model, lr, momentum=0.9):
    mse = nn.CrossEntropyLoss() # nn.MSELoss()
    sgd = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    return mse, sgd
