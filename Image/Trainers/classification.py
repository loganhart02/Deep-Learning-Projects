import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter



def get_loss_optim(model, device, lr=1e-2, momentum=0.9, weight_decay=5e-4):
    "same as in the vgg16 paper default values are what they use in paper"
    losd_func = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    return losd_func, optimizer


def get_accuracy(output, target):
    preds = torch.argmax(output, dim=1)  # get the highest probability prediction for each item in the batch
    try:
        target_indices = torch.argmax(target, dim=1)  # convert one-hot encoded target to class indices
        correct = (preds == target_indices).sum().item()
        acc = correct / len(target)
    except:
        correct = (preds == target).sum().item()
        acc = correct / len(target)
    return acc


class ImageClassificationTrainer:    
    def __init__(
        self, 
        model, 
        img_transforms, 
        test_transforms,
        batch_size, 
        epochs, 
        loss_func, 
        optimizer, 
        device,
        lr_scheduler=None,
        num_workers=2,
        tensorboard_dir="./runs",
        model_dir="./models",
        log_interval=100,  # Log every 10 steps by default
    ):
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.transform = img_transforms
        self.test_transform = test_transforms
        self.lr_scheduler = lr_scheduler
        self.num_workers = num_workers
        self.writer = SummaryWriter(tensorboard_dir)
        self.model_dir = model_dir
        self.log_interval = log_interval
        
        
    def generic_training_loop(self, train_dl, test_dl):
        """generic training loop for all datasets needs improvement"""
        best_test_acc = 0.0
        global_step = 0  # Initialize a counter for global steps

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            running_acc = 0.0
            
            for i, (inputs, labels) in enumerate(train_dl):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.loss_func(outputs, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                running_acc += get_accuracy(outputs, labels)

                global_step += 1  # Increment the global step counter

                # Print current step in epoch and global step count
                if global_step % self.log_interval == 0:
                    print(f"Epoch {epoch+1}/{self.epochs} -- Step {i+1}/{len(train_dl)} -- Global Step {global_step}\nLoss: {loss.item():.4f}\nAccuracy: {get_accuracy(outputs, labels):.4f}")
            
            epoch_loss = running_loss / len(train_dl)
            epoch_acc = running_acc / len(train_dl)
            
            self.writer.add_scalar('Training Loss', epoch_loss, epoch)
            self.writer.add_scalar('Training Accuracy', epoch_acc, epoch)
            
            
            # eval model
            self.model.eval()
            test_loss = 0.0
            test_acc = 0.0
            with torch.no_grad():
                for x, (inputs, labels) in enumerate(test_dl):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.loss_func(outputs, labels)
                    test_loss += loss.item()
                    test_acc += get_accuracy(outputs, labels)
                    
                    global_step += 1  # Increment the global step counter

                    if global_step % self.log_interval == 0:
                        print(f"Epoch {epoch+1}/{self.epochs} -- Step {x+1}/{len(test_dl)} -- Global Step {global_step}\nLoss: {loss.item():.4f}\nAccuracy: {get_accuracy(outputs, labels):.4f}")
            
            
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            test_loss /= len(test_dl)
            test_acc /= len(test_dl)
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save(self.model.state_dict(), f"{self.model_dir}/best_model_{best_test_acc:.4f}.pth")
                print(f'New best model saved with accuracy: {test_acc:.4f}')
            
            self.writer.add_scalar('Test Loss', test_loss, epoch)
            self.writer.add_scalar('Test Accuracy', test_acc, epoch)

            print(f'Epoch {epoch+1}/{self.epochs}\nTrain Loss: {epoch_loss:.4f} -- Train Accuracy: {epoch_acc:.4f} -- '
                f'Test Loss: {test_loss:.4f} -- Test Accuracy: {test_acc:.4f}')
        
        self.writer.close()

        
    def train_cifar10(self, path_to_store_data="./data"):
        trainset = torchvision.datasets.CIFAR10(
            root=path_to_store_data, 
            train=True,
            download=True, 
            transform=self.transform
        )
        testset = torchvision.datasets.CIFAR10(
            root=path_to_store_data, 
            train=False,
            download=True, 
            transform=self.test_transform
        )
        train_dl = torch.utils.data.DataLoader(
            trainset, 
            batch_size=self.batch_size,
            shuffle=True, 
            num_workers=self.num_workers
        )
        test_dl = torch.utils.data.DataLoader(
            testset, 
            batch_size=self.batch_size,
            shuffle=False, 
            num_workers=self.num_workers
        )

        classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        print(f"there are {len(classes)} classes in cifar10")
        print(f"classes are {classes}")
        self.generic_training_loop(train_dl, test_dl)