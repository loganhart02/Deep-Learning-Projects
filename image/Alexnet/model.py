import torch
import torch.nn as nn



class AlexNet(nn.Module):
    """
        - Introduces using relu activation
        - uses local response normalization
              they use k=2, n=5, a=10^-4, B=0.75:: n is the layer size
        - normalization is applied after relu in certain layers
        - uses softmax to create prob. distribution
        
    """
    def __init__(self, img_channels: int=3, num_classes: int = 1000, dropout: float = 0.5):
        super(AlexNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(img_channels, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, k=2, alpha=1e-4, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, k=2, alpha=1e-4, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, stride=1), #
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, stride=1),#
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 192, kernel_size=3, stride=1),#
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # flatten everything but batch
        x = self.classifier(x)
        return x
        
        
        

        
        
        
        