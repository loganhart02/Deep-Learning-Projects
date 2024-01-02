import torch
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

from dataset import get_transforms


class Lenet5(nn.Module):
    def __init__(self, img_channels: int = 1, num_classes: int = 5, activation: str = "tanh"):
        super(Lenet5, self).__init__()
        self.num_classes = num_classes
        self.in_channels = img_channels
        
        self.conv1 = nn.Conv2d(self.in_channels, 6, kernel_size=5, stride=1, padding=0, bias=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5 ,stride=1, padding=0, bias=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16*53*53, 120) # 53 is output size of conv for images of 224x224 size
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
        return x
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    
def predict_torch(model, image, return_probabilities=False):
    model.eval()
    with torch.no_grad():  # Turn off gradients for inference
        logits = model(x)
        if return_probabilities:
            probabilities = F.softmax(logits, dim=1)
            return probabilities
        else:
            return torch.argmax(logits, dim=1)  
        
        

def predict(image_path, model, device, return_probabilities=False):
    """
    Predict the class for a given image using the Lenet5 model.

    Parameters:
    image_path (str): Path to the image file.
    model (torch.nn.Module): The Lenet5 model.
    device (torch.device): The device to run the model on (e.g., 'cpu' or 'cuda').

    Returns:
    int: Predicted class label.
    """
    # Define the transformation. This should match your training data preprocessing
    transform = get_transforms()

    # Load and transform the image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add a batch dimension

    # Move the image and model to the same device
    image = image.to(device)
    model = model.to(device)

    # Set the model to evaluation mode and make a prediction
    model.eval()
    with torch.no_grad():
        logits = model(image)
        if return_probabilities:
            probabilities = F.softmax(logits, dim=1)
            return probabilities
        else:
            return torch.argmax(logits, dim=1)  

