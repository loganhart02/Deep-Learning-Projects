import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2


def transforms(test_set: bool = False, normalize_transforms: v2.Normalize = None):
    transforms = [
        v2.RandomResizedCrop(224),
        v2.RandomHorizontalFlip(),
        v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
    if normalize_transforms:
        transforms.append(normalize_transforms)
    return v2.Compose(transforms)
    


