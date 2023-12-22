import os
import torch
from image.generic_trainers import PopularTrainers
from image.Alexnet.utils import get_loss_optimizer
from image.Alexnet.model import AlexNet
from image.Alexnet.dataset import get_transforms

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


if __name__=="__main__":
    # hyperparams
    batch_size = 1024
    epochs = 100
    lr = 0.01
    momentum = 0.9
    weight_decay = 0.0005
    num_workers = 8
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # get model
    model = AlexNet(num_classes=10)
    model.to(device)
    
    # get loss and optimizer
    loss_func, optimizer = get_loss_optimizer(model, lr, momentum, weight_decay)
    
    
    # get trainer
    trainer = PopularTrainers(
        model, 
        get_transforms(), 
        batch_size, 
        epochs, 
        loss_func, 
        optimizer, 
        device,
        num_workers
    )
    
    # train
    trainer.train_cifar10()

    