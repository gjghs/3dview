import torch
import torchvision
from model.encoder import Encoder

if __name__ == '__main__':

    # model = torchvision.models.resnet18()
    # model.eval()
    # dummy_input = torch.randn(1, 3, 224, 224)
    # torch.onnx.export(model, dummy_input, "resnet18.onnx")

    model = Encoder()
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input, "encoder.onnx")