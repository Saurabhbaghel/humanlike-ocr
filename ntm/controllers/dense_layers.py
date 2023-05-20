import torch
from torch import nn
from torch.nn import Parameter
import numpy as np
from torchvision.models.resnet import resnet50


class LinearNetController(nn.Module):
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
            nn.Linear(32, 100),
            nn.Linear(100, 200),
            nn.Linear(200, 400)
            
            # ConvBlock(self.num_inputs, 32, (2,2)),
            # ConvBlock(32, 32, (2,2)),
            # ConvBlock(32, 64, (2,2))
            # ConvBlock(32, 64),
            # ConvBlock(64, 128),
            # ConvBlock(128, 256)
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
        
        y = self.model_(x)

        # outp = self.fc_(y)
        return self.flatten(y)

    def __name__(self):
        return "LinearNetController"