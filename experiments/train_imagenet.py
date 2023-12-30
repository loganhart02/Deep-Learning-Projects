import os
import torch

from Alexnet.utils import get_loss_optimizer
from Alexnet.model import AlexNet
from Alexnet.dataset import ImageNetDataset
from Alexnet.trainer import ImageClassificationTrainer


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

batch_size = 1024
epochs = 100
lr = 0.01
momentum = 0.9
weight_decay = 0.0005
num_workers = 8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_set = ImageNetDataset("/home/logan/projects/paper-implementations/image/Alexnet/data_engine/train.csv")
test_set = ImageNetDataset("/home/logan/projects/paper-implementations/image/Alexnet/data_engine/eval.csv", test_dataset=True)

model = AlexNet(num_classes=1000)
model.to(device)

loss_func, optimizer, schedular = get_loss_optimizer(model, lr, device, momentum, weight_decay)


trainer = ImageClassificationTrainer(
        model, 
        train_set,
        test_set,
        "/media/logan/m.2/datasets/image/imagenet-dataset/2012/test/test", # test directory
        batch_size, 
        epochs, 
        loss_func, 
        optimizer, 
        device,
        num_workers,
        lr_schedular=schedular,
        tensorboard_dir="/home/logan/projects/paper-implementations/image/Alexnet/experiments/imagenet/runs",
        model_dir="/home/logan/projects/paper-implementations/image/Alexnet/experiments/imagenet/models"
    )

trainer.fit()