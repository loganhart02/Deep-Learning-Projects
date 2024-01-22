import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset


class ImageNetDataset(Dataset):
    """
    csv file should have two columns: image_path, label with sep="|"
    image_path should be entire path to image
    """
    def __init__(self, csv_file, transforms):
        self.data = pd.read_csv(csv_file, sep="|")
        self.img_transforms = transforms
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
        return img, label
    