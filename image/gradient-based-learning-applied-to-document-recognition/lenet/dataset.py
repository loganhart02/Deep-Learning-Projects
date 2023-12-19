import os
import torch
from glob import glob
import pandas as pd
from torchvision.io import read_image
from torchvision.transforms import v2, Lambda
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt



class AnimalClassificationDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, target_transform=None):
        self.csv_file = pd.read_csv(csv_file, sep="|")
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        
        self.class_map = {
            "cat": 0,
            "dog": 1,
            "sheep": 2,
            "cow": 3,
            "horse": 4
        }
        
        self.label_transform = Lambda(lambda y: torch.zeros(
             10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
        
    def __len__(self):
        return len(self.csv_file)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.csv_file.iloc[idx, 0])
        img = read_image(img_path)
        label = self.class_map[self.csv_file.iloc[idx, 1]]
        if self.transform:
            img = self.transform(img)
            
        if self.target_transform:
            label = self.target_transform(label)
        return img, torch.tensor(label)


def get_data_loaders(train_d: Dataset, test_d: Dataset, batch_size: int):
    train_loader = DataLoader(train_d, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_d, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader



def check_data_loader(loader):
    # Display image and label.
    train_features, train_labels = next(iter(loader))
    img = train_features[0].squeeze()
    label = train_labels[0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")
    
    

def get_transforms():
    transforms = v2.Compose([
        v2.Grayscale(num_output_channels=1),
        v2.Resize((224, 224)),  
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToTensor(),
        v2.ToDtype(torch.float32, scale=True),  
        v2.Normalize(mean=[0.485], std=[0.229]), 
       #  v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) if you change to rgb img use this
    ])
    return transforms
