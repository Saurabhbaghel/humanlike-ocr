import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from .memory import NTMMemory

def _split_cols(mat, lengths):
    '''splitting a 2D matrix to variable length columns'''
    assert mat.size()[1] == sum(lengths), "Lengths must be summed to num columns"
    l = np.cumsum([0] + lengths)
    results = []
    for s, e in zip(l[:-1], l[1:]):
        results += [mat[:, s:e]]
    return results

class NTMHeadBase(nn.Module):
    def __init__(self, memory:NTMMemory, controller_size:int) -> None:
        super().__init__()
        self.memory = memory
        self.N, self.M = memory.size()
        self.controller_size = controller_size
        
    def create_new_state(self, batch_size):
        raise NotImplementedError
    
    def register_parameters(self):
        raise NotImplementedError
    
    def is_read_head(self):
        return NotImplementedError
    
    def _address_memory(self, k, beta, g, s, gamma, w_prev):
        """_summary_

        Args:
            k (_type_): key vector at time t
            beta (_type_): key strength at time t
            g (_type_): interpolation gate at time t
            s (_type_): shift weighting at time t
            gamma (_type_): sharpening factor at time t
            w_prev (_type_): weight vector at at time t-1

        Returns:
            _type_: _description_
        """
        k = k.clone()
        beta = F.softplus(beta)
        g = torch.sigmoid(g)
        s = torch.softmax(s, dim=1)
        gamma = 1 + F.softplus(gamma)
        
        w = self.memory.address(k, beta, g, s, gamma, w_prev)
        
        return w
    
class NTMReadHead(NTMHeadBase):
    def __init__(self, memory:NTMMemory, controller_size:int) -> None:
        super().__init__(memory, controller_size)
        
        self.read_lengths = [self.M, 1, 1, 3, 1] # corresponding to k, beta, g, s, gamma sizes
        self.fc_read = nn.Linear(controller_size, sum(self.read_lengths))
        self.reset_parameters()
        
    def create_new_state(self, batch_size):
        # the state holds the previous time step address weightings
        return torch.zeros(batch_size, self.N)
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc_read.weight, gain=1.4)
        nn.init.normal_(self.fc_read.bias, std=0.01)
        
    def is_read_head(self):
        return True
    
    def forward(self, embeddings, w_prev):
        """_summary_

        Args:
            embeddings (_type_): input embeddings 
            w_prev (_type_): the weight vector at prev time

        Returns:
            _type_: _description_
        """
        
        o_ = self.fc_read(embeddings)
        o = o_.clone()
        k, beta, g, s, gamma = _split_cols(o, self.read_lengths)
        
        # Read from memory
        w = self._address_memory(k, beta, g, s, gamma, w_prev)
        r = self.memory.read(w)
        
        return r, w
    
class NTMWriteHead(NTMHeadBase):
    def __init__(self, memory:NTMMemory, controller_size:int) -> None:
        super().__init__(memory, controller_size)
        
        self.write_lengths  = [self.M, 1, 1, 3, 1, self.M, self.M]
        self.fc_write = nn.Linear(controller_size, sum(self.write_lengths))
        self.reset_parameters()
        
    def create_new_state(self, batch_size):
        return torch.zeros(batch_size, self.N)
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc_write.weight, gain=1.4)
        nn.init.normal_(self.fc_write.bias, std=0.01)
        
    def is_read_head(self):
        return False
    
    def forward(self, embeddings, w_prev):
        o = self.fc_write(embeddings)
        k, beta, g, s, gamma, e, a = _split_cols(o, self.write_lengths)
        
        # e should be in [0, 1] 
        e = torch.sigmoid(e)
        
        # write to memory
        w = self._address_memory(k, beta, g, s, gamma, w_prev)
        self.memory.write(w, e, a)
        
        return w
        