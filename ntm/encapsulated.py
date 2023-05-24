import torch
from torch import nn
from .ntm import NTM
from ntm.controllers import LSTMController,LinearNetController, ConvNetController
from .head  import NTMReadHead, NTMWriteHead
from .memory import NTMMemory

class EncapsulatedNTM(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_outputs,
        controller_size,
        controller_layers,
        num_heads,
        N,
        M,
        controller_=ConvNetController
        ) -> None:
        super().__init__()
        
        # save the args
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_heads = num_heads
        self.controller_size = controller_size
        self.controller_layers = controller_layers
        self.N = N
        self.M = M
        
        # create the NTM components
        memory = NTMMemory(N, M)
        # TODO check the controller according to the name
        controller = controller_(num_inputs, controller_layers) #LSTMController(num_inputs + M*num_heads, controller_size, controller_layers)
        heads = nn.ModuleList([])
        for i in range(num_heads):
            heads += [
                NTMReadHead(memory, controller_size),
                NTMWriteHead(memory, controller_size)
            ]
        self.ntm = NTM(num_inputs, num_outputs, controller, memory, heads)
        self.memory = memory
        
    def init_sequence(self, batch_size):
        self.batch_size = batch_size
        self.memory.reset(batch_size)
        self.previous_state = self.ntm.create_new_state(batch_size)
        
    def forward(self, x=None):
        if x is None:
            x = torch.zeros(self.batch_size, self.num_inputs, dtype=torch.float32)
        x = x.type(torch.float)
        o, self.previous_state = self.ntm(x, self.previous_state)
        return o, self.previous_state
    
    def calculate_num_params(self):
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)
        return num_params