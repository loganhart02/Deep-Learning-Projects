import os
import torch

from utils import get_loss_optimizer
from model import AlexNet
from dataset import ImageNetDataset
from trainer import ImageClassificationTrainer


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

batch_size = 1024
epochs = 100
lr = 0.01
momentum = 0.9
weight_decay = 0.0005
num_workers = 8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_set = ImageNetDataset("/media/logan/m.2/datasets/image/imagenet-dataset/2010/train.csv")
test_set = ImageNetDataset("/media/logan/m.2/datasets/image/imagenet-dataset/2010/eval.csv", test_dataset=True)

model = AlexNet(num_classes=49)
model.to(device)

loss_func, optimizer = get_loss_optimizer(model, lr, momentum, weight_decay)


trainer = ImageClassificationTrainer(
        model, 
        train_set,
        test_set,
        "/media/logan/m.2/datasets/image/imagenet-dataset/2010/test/test", # test directory
        batch_size, 
        epochs, 
        loss_func, 
        optimizer, 
        device,
        num_workers,
        tensorboard_dir="/home/logan/projects/paper-implementations/image/Alexnet/imagenet_experiments/runs",
        model_dir="/home/logan/projects/paper-implementations/image/Alexnet/imagenet_experiments/weights"
    )

trainer.fit()