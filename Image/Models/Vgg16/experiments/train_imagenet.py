import os
import torch
import torch
from model import Vgg19
from torchvision import transforms as v2
from g_trainer import Trainer, get_loss_optim
from dataset import transforms

from dataset import ImageNetDataset


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


batch_size = 256
lr = 1e-2
epochs = 300
num_workers = 2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


vgg16 = Vgg19(img_channels=3, num_classes=1000).to(device)

l, o = get_loss_optim(vgg16, device, lr=lr)

img_normalize = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
img_transforms = transforms(normalize_transforms=img_normalize)

train_set = ImageNetDataset("/home/logan/projects/paper-implementations/data_engine/train.csv", transforms=img_transforms)
test_set = ImageNetDataset("/home/logan/projects/paper-implementations/data_engine/eval.csv", transforms=img_transforms)

trainer =  Trainer(
    vgg16, 
    batch_size=batch_size,
    epochs=epochs,
    loss_func=l,
    optimizer=o,
    device=device,
    num_workers=num_workers,
    img_transforms=img_transforms,
    test_transforms=img_transforms,
    model_dir="/home/logan/models/vgg16/imagenet",
    tensorboard_dir="/home/logan/models/vgg16/imagenet",  
    log_interval=50
)


train_dl = torch.utils.data.DataLoader(
    train_set, 
    batch_size=trainer.batch_size, 
    shuffle=True, 
    num_workers=num_workers
)
test_dl = torch.utils.data.DataLoader(
    test_set, 
    batch_size=trainer.batch_size // 2, 
    shuffle=False, 
    num_workers=num_workers
)

trainer.generic_training_loop(train_dl, test_dl)