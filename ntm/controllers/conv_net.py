import torch
from torch import nn
from torch.nn import Parameter
import numpy as np
from torchvision.models.resnet import resnet50
from .base import BaseController

class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, out_channels: int, kernel_size: tuple = (3, 3)):
        """Basic Conv Block with convolution layer, etc.

        Args:
            input_channels (int): _description_
            out_channels (int): _description_
            kernel_size (tuple, optional): _description_. Defaults to (3, 3).
        """
        super().__init__()
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        self.conv = nn.Conv2d(self.input_channels, self.out_channels, self.kernel_size)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(self.kernel_size)
        self.dropout = nn.Dropout2d()
        self.batchnorm = nn.BatchNorm2d(self.out_channels)
    
    def forward(self, x, training:bool = True):
        conv_res = self.conv(x)
        activated = self.relu(conv_res)
        pooled = self.maxpool(activated)
        if training:
            pooled = self.dropout(pooled)
        y = self.batchnorm(pooled)
        return y
    
    def __repr__(self):
        return f"ConvBlock({self.input_channels, self.out_channels, self.kernel_size})"


class ConvNetController(BaseController):
    def __init__(self, num_inputs:int, num_layers:int) -> None:
        super().__init__()
        self.num_inputs = num_inputs
        # self.num_outputs = num_outputs
        self.num_layers = num_layers
        # self.batch_size = batch_size
        self.device_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.layer = [
        #     nn.Conv1d(1, 1, 1, device=self.device_),
        #     nn.LeakyReLU()
        # ]
        # self.layers = [
        #     nn.Linear(80, 100),
        #     nn.ReLU(),
        #     nn.Linear(100, 200),
        #     nn.ReLU(),
        #     nn.Linear(200, 300),
        #     nn.ReLU(),
        #     nn.Linear(300, 200),
        #     nn.ReLU(),
        #     nn.Linear(200, 80),
        #     nn.ReLU()
        # ]
        self.model_ = nn.Sequential(
            # nn.Linear(25, 100),
            # nn.Linear(100, 200),
            # nn.Linear(200, 400)
            
            ConvBlock(self.num_inputs, 32, (2,2)),
            # ConvBlock(32, 32, (2,2)),
            # ConvBlock(32, 64, (2,2))
            ConvBlock(32, 64, (2, 2)),
            ConvBlock(64, 128, (2, 2)),
            ConvBlock(128, 256, (2, 2))
        )
        self.flatten = nn.Flatten()
        # self.fc_ = nn.Linear(64, 44, device=self.device_)
        # model_ = resnet50(pretrained=True)
        # self.feature_extractor = nn.Sequential(*list(model_.children())[:-1]).to(self.device_)
        
        # for layer_ in self.layers[::2]:
        #     nn.init.kaiming_normal_(layer_.weight)

        # self.feedforward = nn.ModuleList(
        #     self.layers #* self.num_layers
            # )
            # nn.LazyLinear(out_features=20, device=self.device_),
            # nn.Linear(20, self.num_outputs, device=self.device_)
            


    def forward(self, x, training:bool=True):
        # if x.ndim != 3:
        #     x = x.unsqueeze(1)
            # raise AssertionError(f"dimension of the input is {x.ndim} and shape is {x.size()}. It should be a 3d tensor.")
        # for layer in self.feature_extractor:
        #     x = layer(x)
        
        # x1 = ConvBlock(3, 32)(x)
        # print(x1.shape)
        # x2 = ConvBlock(32, 32)(x1)
        # print(x2.shape)
        # y = ConvBlock(32, 64)(x2)
        # print(y.shape)
        
        # x = x.squeeze() if x.ndim == 4 else x

        # print(x.size())
        y = self.model_(x)

        # outp = self.fc_(y)
        return self.flatten(y)


    def __name__(self):
        return "ConvNetController"
