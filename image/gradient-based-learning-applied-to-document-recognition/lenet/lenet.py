import torch
import torch.nn as nn
import torch.nn.functional as F


class Lenet5(nn.Module):
    "input is 32x32 pixel image can use 28x28"
    def __init__(self, img_channels: int = 1, num_classes: int = 5, activation: str = "tanh"):
        super(Lenet5, self).__init__()
        self.num_classes = num_classes
        self.in_channels = img_channels
        
        self.conv1 = nn.Conv2d(self.in_channels, 6, kernel_size=5, stride=1, padding=0, bias=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5 ,stride=1, padding=0, bias=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16*53*53, 120) # change the 53 if you plan on changing from 224x224 img size
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.num_classes)
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
                self.activation = nn.ReLU()
        else:
            raise ValueError("pick either tanh or relu for activation")
        
    def forward(self, x):
        """
        Notes:
             we flatten because conv and pooling layers output a 3D tensor and linear layers expect 1D input
             flattening converts the high-dim output into one long vector. it doesn't change the data just how it is represented
        """
        x = self.activation(self.maxpool1(self.conv1(x)))
        x = self.activation(self.maxpool2(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten everything but batch
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        # x = F.softmax(x, dim=1) # uncomment if you want probability scores
        return x
        
        
        