import torch
from torch import nn
from torch.nn import Parameter
import numpy as np
from torchvision.models.resnet import resnet50

class LSTMController(nn.Module):
    def __init__(self, num_inputs:int, num_outputs:int, num_layers:int) -> None:
        """initalizes the lstm controller

        Args:
            num_inputs (int): Number of inputs
            num_outputs (int): 
            num_layers (int): 
        """        
        super().__init__()
        
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_layers = num_layers
        self.device_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        self.lstm = nn.LSTM(
            input_size = num_inputs,
            hidden_size = num_outputs,
            num_layers = num_layers
        )
        
        # the hidden state is a learned parameter
        self.lstm_h_bias = Parameter(torch.randn(self.num_layers, 1, self.num_outputs) * 0.05)
        self.lstm_c_bias = Parameter(torch.randn(self.num_layers, 1, self.num_outputs) * 0.05)
        
        self.reset_parameters()
        
    def create_new_state(self, batch_size):
        # Dimension: (num_layers * num_directions, batch, hidden_size)
        lstm_h = self.lstm_h_bias.clone().repeat(1, batch_size, 1).to(self.device_)
        lstm_c = self.lstm_c_bias.clone().repeat(1, batch_size, 1).to(self.device_)
        return lstm_h, lstm_c
    
    def reset_parameters(self):
        for p in self.lstm.parameters():
            if p.dim() == 1:
                nn.init.constant_(p, 0)
            else:
                stdev = 5 / (np.sqrt(self.num_inputs + self.num_outputs))
                nn.init.uniform_(p, -stdev, stdev)
    
    def size(self):
        return self.num_inputs, self.num_outputs
    
    def forward(self, x, prev_state):
        x = x.unsqueeze(0) 
        outp, state = self.lstm(x, prev_state) 
        return outp.squeeze(0), state
    
class FeedforwardController(nn.Module):
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
        model_ = resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(model_.children())[:-1]).to(self.device_)
        
        # for layer_ in self.layers[::2]:
        #     nn.init.kaiming_normal_(layer_.weight)

        # self.feedforward = nn.ModuleList(
        #     self.layers #* self.num_layers
            # )
            # nn.LazyLinear(out_features=20, device=self.device_),
            # nn.Linear(20, self.num_outputs, device=self.device_)
            


    def forward(self, x):
        # if x.ndim != 3:
        #     x = x.unsqueeze(1)
            # raise AssertionError(f"dimension of the input is {x.ndim} and shape is {x.size()}. It should be a 3d tensor.")
        # for layer in self.feature_extractor:
        #     x = layer(x)
        outp = self.feature_extractor
        return outp
