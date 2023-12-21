import os
import torch
import pandas as pd
from torchvision.io import read_image
from torchvision.transforms import v2, Lambda
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt



def get_transforms():
        """These are all the transforms they use in the alexnet paper"""
        return v2.Compose(
            [
                v2.Resize(256),
                v2.CenterCrop(224),
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # ColorJitter added here works similar to PCA augment in paper
                v2.RandomHorizontalFlip(p=0.5),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                ),
            ]
        )


class AlexNetDataset(Dataset):
    """
    Image classification dataset based on alexnet paper. It just includes the transforms they use
    the rest is a normal pytorch image dataset class
    """
    def __init__(self):
        self.img_transforms = get_transforms()
    