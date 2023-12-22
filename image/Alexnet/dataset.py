import torch
import pandas as pd
from torchvision.io import read_image
from torchvision.transforms import v2, Lambda
from torch.utils.data import Dataset
from PIL import Image



def get_transforms():
        """These are all the transforms they use in the alexnet paper"""
        return v2.Compose(
            [
                v2.Resize(256, antialias=True),
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


class ImageNetDataset(Dataset):
    """
    csv file should have two columns: image_path, label with sep="|"
    image_path should be entire path to image
    """
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file, sep="|")
        self.img_transforms = get_transforms()
        self.label_transform = Lambda(lambda y: torch.zeros(
            1000, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
        
        self._create_label_map()
        
    def _create_label_map(self):
        self.label_map = {}
        for i, label in enumerate(self.data.label.unique()):
            self.label_map[label] = i
        self.num_to_label = {value: key for key, value in self.label_map.items()}
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0] # image is first column
        img = read_image(img_path)
        label = self.label_map[self.data.iloc[idx, 1]] # label is second column
        img = self.img_transforms(img)

        label = self.label_transform(label)
        return img, label

    