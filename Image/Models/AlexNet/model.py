import torch
import onnx
import torch.nn as nn
from torchvision.transforms import v2



def get_transforms(test_dataset=False):
        """These are all the transforms they use in the alexnet paper"""
        if test_dataset:
            return v2.Compose(
                [
                    v2.Resize(256, antialias=True),
                    v2.CenterCrop(224),
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(
                        mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            return v2.Compose(
                [
                    v2.Resize(256, antialias=True),
                    v2.CenterCrop(224),
                    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # ColorJitter added here works similar to PCA augment in paper
                    v2.RandomHorizontalFlip(p=0.5),
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(
                        mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225]
                    ),
                ]
            )


class AlexNet(nn.Module):
    """
        - Introduces using relu activation
        - uses local response normalization
              they use k=2, n=5, a=10^-4, B=0.75:: n is the layer size
        - normalization is applied after relu in certain layers
        - uses softmax to create prob. distribution
    """
    def __init__(self, img_channels: int=3, num_classes: int = 1000, dropout: float = 0.5, pretrained: bool = False, weights_path: str = None):
        super(AlexNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(img_channels, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, k=2, alpha=1e-4, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, k=2, alpha=1e-4, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, stride=1), #
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, stride=1),#
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 192, kernel_size=3, stride=1),#
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )
        
        if pretrained is True:
            assert weights_path is not None, "weights_path must be provided if pretrained is True"
            self.load_pretrained(path_to_weights=weights_path)
            
        
    def load_pretrained(self, path_to_weights: str):
        self.load_state_dict(torch.load(path_to_weights))
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # flatten everything but batch
        x = self.classifier(x)
        return x
        
        
        
class ModelExporter:
    def __init__(self, model, output_path):
        self.model = model
        self.input_shape = (1, 3, 224, 224)
        self.output_path = output_path
        
    def export(self):
        self.model.eval()
        dummy_input = torch.randn(self.input_shape)
        onnx_program = torch.onnx.dynamo_export(self.model, dummy_input)
        onnx_program.save(self.output_path)
        
        print("Exported model to ", self.output_path)
        
        print("Testing exported model...")
        try:
            onnx_model = onnx.load(self.output_path)
            onnx.checker.check_model(onnx_model)
            print("Exported model passed testing")
        except:
            print("Exported model failed testing")
        
        
        

        
        
        
        