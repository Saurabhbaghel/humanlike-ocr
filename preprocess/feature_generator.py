import torch
from torch import nn
from torchvision import transforms

class FeatureExtractor(nn.Module):
    def __init__(self, img_size:int = 112) -> None:
        super().__init__()
        model_ = torch.hub.load("pytorch/vision:v0.10.0", "mobilenet_v2", pretrained=True)
        self.model = torch.nn.Sequential(*(list(model_.children)))
        self.preprocess = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # to convert to Grayscale 
            transforms.Resize(img_size), # resizing to 112
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def forward(self, img):
        self.model.eval()
        input_tensor = self.preprocess(img)
        input_batch = input_tensor.unsqueeze(0) # create a minibatch as expected by the model
        with torch.no_grad():
            output = self.model(input_batch)