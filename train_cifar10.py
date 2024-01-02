import os
import torch
from model import Vgg16
from torchvision import transforms as v2
from trainer import Trainer, get_loss_optim
from dataset import transforms





batch_size = 256
lr = 1e-2
epochs = 100
num_workers = 8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


vgg16 = Vgg16(img_channels=3, num_classes=10).to(device)

l, o = get_loss_optim(vgg16, device, lr=lr)

img_normalize = v2.Normalize((0.5,), (1.0))
img_transforms = transforms(normalize_transforms=img_normalize)


trainer =  Trainer(
    vgg16, 
    None,
    None,
    None,
    batch_size=batch_size,
    epochs=epochs,
    loss_func=l,
    optimizer=o,
    device=device,
    num_workers=num_workers,
    img_transforms=img_transforms,
    test_transforms=img_transforms,
    model_dir="/media/logan/models/vgg16/cifar10",
    tensorboard_dir="/media/logan/models/vgg16/cifar10",  
    log_interval=25
)

trainer.train_cifar10(path_to_store_data="/media/logan/datasets/image/cifar10", download=True)