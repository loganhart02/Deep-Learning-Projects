"""File contains trainers for mnist, imagenet, cifar10, cifar100"""
import os
import random
import torch
import torchvision
from torchvision.transforms import v2
from PIL import Image, ImageDraw
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


class Trainer:    
    def __init__(
        self, 
        model,
        trainset,
        evalset,
        test_directory,
        batch_size, 
        epochs, 
        loss_func, 
        optimizer, 
        device, 
        num_workers=2,
        lr_schedular=None,
        img_transforms=None,
        test_transforms=None,
        tensorboard_dir="./runs",
        model_dir="./models",
        log_interval=100,  # Log every 10 steps by default
    ):
        self.model = model
        self.bs = batch_size
        self.trainset = trainset
        self.testset = evalset
        self.test_directory = test_directory
        self.epochs = epochs
        self.device = device
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.transform = img_transforms
        self.test_transform = test_transforms
        self.lr_scheduler = lr_schedular
        self.num_workers = num_workers
        self.writer = SummaryWriter(tensorboard_dir)
        self.model_dir = model_dir
        self.log_interval = log_interval
        
    
    def _get_dataloaders(self):
        self.train_dl = torch.utils.data.DataLoader(
            self.trainset, 
            batch_size=self.bs,
            shuffle=True, 
            num_workers=self.num_workers
        )
        self.test_dl = torch.utils.data.DataLoader(
            self.testset, 
            batch_size=self.bs,
            shuffle=False, 
            num_workers=self.num_workers
        )
        
    def train_one_step(self):
        self.model.train()
        running_loss = 0.0
        running_acc = 0.0
        for i, (inputs, labels) in enumerate(self.train_dl):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            outputs = self.model(inputs)
            loss = self.loss_func(outputs, labels)
            
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            running_acc += get_accuracy(outputs, labels)
            
            if (i + 1) % self.log_interval == 0:
                print(
                f"""\nStep {i + 1} out of {len(self.trainset) // self.bs}\nLoss: {running_loss / (i + 1):.4f}\nAccuracy: {running_acc / (i + 1):.4f}""")
        return running_loss, running_acc
    
    def eval_one_step(self):
        self.model.eval()
        test_loss = 0.0
        test_acc = 0.0
        with torch.no_grad():
            for inputs, labels in self.test_dl:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_func(outputs, labels)
                test_loss += loss.item()
                test_acc += get_accuracy(outputs, labels)
        return test_loss, test_acc
    
    def train_one_epoch(self):
        train_loss, train_acc = self.train_one_step()
        train_loss /= len(self.train_dl)
        train_acc /= len(self.train_dl)
        return train_loss, train_acc
    
    def eval_one_epoch(self, best_test_acc: float = None):
        if best_test_acc is None:
            best_test_acc = 0.0
        test_loss, test_acc = self.eval_one_step()
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(self.model.state_dict(), f"{self.model_dir}/best_model_{best_test_acc:.4f}.pth")
            print(f'New best model saved with accuracy: {test_acc:.4f}')
        return test_loss, test_acc, best_test_acc
    
    def run_test(self, num_images: int = 5):
        transform = v2.Compose([
            v2.Resize(256, antialias=True), # Adjust according to your model's requirements
            v2.ToTensor(),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.model.eval()
        image_files = os.listdir(self.test_dir)
        random.shuffle(image_files)
        image_files = image_files[:num_images]
                
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(self.test_dir, image_file)
            image = Image.open(image_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device=self.device)

            with torch.no_grad():
                output = self.model(input_tensor)
                prediction = output.argmax(dim=1).item()

            # Add image with label to TensorBoard
            label = self.trainset.num_to_label[prediction]
            self.add_image_with_text(f'Image_{i}', input_tensor.squeeze(0), label, 0)
            
    def fit(self, run_test: bool = True):
        best_test_acc = 0.0
        self._get_dataloaders()
        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_one_epoch()
            test_loss, test_acc, best_test_acc = self.eval_one_epoch(best_test_acc)
            if run_test:
                self.run_test()
            if self.lr_schedular is not None:
                self.lr_schedular.step()
            
            self.writer.add_scalar('Training Loss', train_loss, epoch)
            self.writer.add_scalar('Training Accuracy', train_acc, epoch)
            self.writer.add_scalar('Test Loss', test_loss, epoch)
            self.writer.add_scalar('Test Accuracy', test_acc, epoch)
            
            print(f'Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, '
                  f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
        self.writer.close()
            
    def _add_image_with_text(self, tag, image_tensor, label, index):
        image = v2.ToPILImage()(image_tensor).convert("RGB")

        # Estimate text size for default font
        font_size = 20  # Approximate font size for default font
        text_height = font_size  # Approximate height of text

        # Create a new image with extra space for text
        new_image = Image.new('RGB', (image.width, image.height + text_height), (255, 255, 255))
        new_image.paste(image, (0, 0))

        # Draw text
        draw = ImageDraw.Draw(new_image)
        text_position = (5, image.height + 5)  # Position text below the image
        draw.text(text_position, label, (0, 0, 0))

        # Convert back to tensor and write to TensorBoard
        new_image_tensor = v2.ToTensor()(new_image)
        self.writer.add_image(tag, new_image_tensor, index)
        
    def train_cifar10(self, path_to_store_data="./data", download=True):
        self.trainset = torchvision.datasets.CIFAR10(
            root=path_to_store_data, 
            train=True,
            download=download, 
            transform=self.transform
        )
        self.testset = torchvision.datasets.CIFAR10(
            root=path_to_store_data, 
            train=False,
            download=download, 
            transform=self.test_transform
        )
        self.train_dl = torch.utils.data.DataLoader(
            self.trainset, 
            batch_size=self.bs,
            shuffle=True, 
            num_workers=self.num_workers
        )
        self.test_dl = torch.utils.data.DataLoader(
            self.testset, 
            batch_size=self.bs,
            shuffle=False, 
            num_workers=self.num_workers,
        )

        classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        print(f"there are {len(classes)} classes in cifar10")
        print(f"classes are {classes}")
        self.fit(run_test=False)
