import os
import torch
import onnxruntime
import gradio as gr
from torchvision.transforms import v2

from model import AlexNet


if __name__ == "__main__":
    current_directory = os.getcwd()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels = [
        "airplane", 
        "automobile", 
        "bird", 
        "cat", 
        "deer", 
        "dog", 
        "frog", 
        "horse", 
        "ship", 
        "truck"
    ]

    
    if device == torch.device("cpu"):
        model_path = os.path.join(current_directory, "experiments", "cifar10", "models",  "alexnet_cifar10.onnx")
        alexnet = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        
        def predict(inp):
            global alexnet
            def to_numpy(tensor):
                return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
            
            preprocess = v2.Compose([
                v2.Resize([224, 224]),            # Resize the image (change according to your requirements)
                v2.ToTensor(),             # Convert the image to a Tensor
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize (use the same values as during training)
            ])
            
            img = preprocess(inp).unsqueeze(0)

            ort_inputs = {alexnet.get_inputs()[0].name: to_numpy(img)}
            ort_outs = alexnet.run(None, ort_inputs)
            img_out_y = ort_outs[0]
            
            probs = torch.nn.functional.softmax(torch.from_numpy(img_out_y), dim=1)
            
            confidences = {labels[i]: float(probs[0][i]) for i in range(len(labels))}
            
            return confidences
            

    elif device == torch.device("cuda"):
        model_path = os.path.join(current_directory, "experiments", "cifar10", "models",  "best_model_0.7351.pth")
        
        alexnet = AlexNet(
            num_classes=10,
            pretrained=True,
            weights_path=model_path
        )
        
        def predict(inp):
            global alexnet
            # Ensure the model is in evaluation mode
            alexnet.eval()

            # Preprocess the input image
            preprocess = v2.Compose([
                v2.Resize([224, 224]),            # Resize the image (change according to your requirements)
                v2.ToTensor(),             # Convert the image to a Tensor
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize (use the same values as during training)
            ])
            inp = preprocess(inp).unsqueeze(0)

            # Check if GPU is available and move the model and input to GPU if it is
            alexnet = alexnet.to("cuda")
            inp = inp.to("cuda")

            with torch.no_grad():
                # Forward pass
                outputs = alexnet(inp)

                # Apply softmax to get probabilities
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

                # Convert probabilities to readable format
                confidences = {labels[i]: float(probabilities[i]) for i in range(len(labels))}

            return confidences
        
        
    gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil"),
        outputs=gr.Label(num_top_classes=10),
        examples=[
            os.path.join(current_directory, "demo/cat.jpg"),
            os.path.join(current_directory, "demo/dog.jpg"),
        ]
    ).launch(share=True)
        
    

