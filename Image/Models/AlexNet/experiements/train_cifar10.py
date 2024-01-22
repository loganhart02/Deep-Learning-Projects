import os
import torch
from Image.generic_trainers import PopularTrainers
from Image.Alexnet.utils import get_loss_optimizer
from Image.Alexnet.model import AlexNet
from Image.Alexnet.dataset import get_transforms

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


if __name__=="__main__":
    # hyperparams
    batch_size = 1024
    epochs = 500
    lr = 0.01
    momentum = 0.9
    weight_decay = 0.0005
    num_workers = 8
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # get model
    model = AlexNet(num_classes=10)
    model.to(device)
    
    # get loss and optimizer
    loss_func, optimizer, schedular = get_loss_optimizer(model, lr, device, momentum, weight_decay)
    
    
    # get trainer
    trainer = PopularTrainers(
        model, 
        get_transforms(), 
        get_transforms(test_dataset=True),
        batch_size, 
        epochs, 
        loss_func, 
        optimizer, 
        device,
        schedular,
        num_workers,
        tensorboard_dir="/home/logan/projects/paper-implementations/image/Alexnet/experiments/cifar10/runs/",
        model_dir="/home/logan/projects/paper-implementations/image/Alexnet/experiments/cifar10/models",
    )
    
    # train
    trainer.train_cifar10(path_to_store_data="/media/logan/m.2/datasets/image/cifar-10")
