import torch
import torch.onnx
import onnx
from lenet import Lenet5


def convert_to_onnx(
    pretrained_weights_path: str, 
    input: tuple = (1, 1, 224, 224),
    output_path: str = "lenet.onnx"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_model = Lenet5()
    torch_model.load_state_dict(torch.load(pretrained_weights_path))
    torch_model.to(device)
    torch_model.eval()
    torch_input = torch.randn(input[0], input[1], input[2], input[3], device=device, requires_grad=True)
    onnx_program = torch.onnx.dynamo_export(torch_model, torch_input)
    onnx_program.save(output_path)
    print(f"Saved ONNX model to {output_path}")
    print("Checking ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model checked successfully. Model is ready for use!")
