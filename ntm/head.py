import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from .memory import NTMMemory

class SimpleRegressor(nn.Module):
    def __init__(self, input_size, output_size, activation=None):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()
        self.activation = activation

    def forward(self, x, ):
        y = self.linear(x)
        if self.activation:
            return self.activation(y)
        return y


def _split_cols(mat, lengths):
    '''splitting a 2D matrix to variable length columns'''
    k_len, beta_len, g_len, s_len, gamma_len, e_len, a_len = lengths
    # regressor for "k"
    k_reg = SimpleRegressor(mat.size()[1], k_len)

    # regressor for "beta"
    beta_reg = SimpleRegressor(mat.size()[1], beta_len)

    # regressor for "g"
    g_reg = SimpleRegressor(mat.size()[1], g_len)

    # regressor for "s"
    s_reg = SimpleRegressor(mat.size()[1], s_len)
    
    # regressor for "gamma"
    gamma_reg = SimpleRegressor(mat.size()[1], gamma_len)

    if e_len > 0:
        # "regressor for e"
        e_reg = SimpleRegressor(mat.size()[1], e_len)
    if a_len > 0:
        # regressor for "a"
        a_reg = SimpleRegressor(mat.size()[1], a_len)
    

    k = k_reg(mat)
    beta = F.softplus(beta_reg(mat))
    g = torch.sigmoid(g_reg(mat))
    s = torch.softmax(s_reg(mat), dim=1)
    gamma = 1 + F.softplus(gamma_reg(mat))

    if e_len > 0:
        e = torch.sigmoid(e_reg(mat))
        a = a_reg(mat) 
        return [k, beta, g, s, gamma, e, a]
    return [k, beta, g, s, gamma] #results

class NTMHeadBase(nn.Module):
    def __init__(self, memory:NTMMemory, controller_size:int) -> None:
        super().__init__()
        self.memory = memory
        self.N, self.M = memory.size()
        self.controller_size = controller_size
        self.device_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        
        self.read_lengths = [self.M, 1, 1, 3, 1] # corresponding to k, beta, g, s, gamma, e, a sizes
        self.fc_read = nn.Linear(controller_size, sum(self.read_lengths))
        
        k_len, beta_len, g_len, s_len, gamma_len = self.read_lengths
        # regressor for "k"
        self.k_reg = SimpleRegressor(controller_size, k_len)

        # regressor for "beta"
        self.beta_reg = SimpleRegressor(controller_size, beta_len)

        # regressor for "g"
        self.g_reg = SimpleRegressor(controller_size, g_len)

        # regressor for "s"
        self.s_reg = SimpleRegressor(controller_size, s_len)
        
        # regressor for "gamma"
        self.gamma_reg = SimpleRegressor(controller_size, gamma_len)

        self.reset_parameters()
        
    def create_new_state(self, batch_size):
        # the state holds the previous time step address weightings
        return torch.zeros(batch_size, self.N).to(self.device_)
    
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
        
        # o_ = self.fc_read(embeddings)
        # o = o_.clone()
        # k, beta, g, s, gamma = _split_cols(embeddings.clone(), self.read_lengths)
        
        mat = embeddings.clone()

        k = self.k_reg(mat)
        beta = F.softplus(self.beta_reg(mat))
        g = torch.sigmoid(self.g_reg(mat))
        s = torch.softmax(self.s_reg(mat), dim=1)
        gamma = 1 + F.softplus(self.gamma_reg(mat))

        # Read from memory
        w = self.memory.address(k, beta, g, s, gamma, w_prev) #self._address_memory(k, beta, g, s, gamma, w_prev)
        r_ = self.memory.read(w)
        r = r_.clone()
        return r, w
    
class NTMWriteHead(NTMHeadBase):
    def __init__(self, memory:NTMMemory, controller_size:int) -> None:
        super().__init__(memory, controller_size)
        
        self.write_lengths  = [self.M, 1, 1, 3, 1, self.M, self.M]
        self.fc_write = nn.Linear(controller_size, sum(self.write_lengths))
        k_len, beta_len, g_len, s_len, gamma_len, e_len, a_len = self.write_lengths
        
        # regressor for "k"
        self.k_reg = SimpleRegressor(controller_size, k_len)

        # regressor for "beta"
        self.beta_reg = SimpleRegressor(controller_size, beta_len)

        # regressor for "g"
        self.g_reg = SimpleRegressor(controller_size, g_len)

        # regressor for "s"
        self.s_reg = SimpleRegressor(controller_size, s_len)
        
        # regressor for "gamma"
        self.gamma_reg = SimpleRegressor(controller_size, gamma_len)

        # "regressor for e"
        self.e_reg = SimpleRegressor(controller_size, e_len)
        
        # regressor for "a"
        self.a_reg = SimpleRegressor(controller_size, a_len)
        self.reset_parameters()
        
    def create_new_state(self, batch_size):
        return torch.zeros(batch_size, self.N).to(self.device_)
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc_write.weight, gain=1.4)
        nn.init.normal_(self.fc_write.bias, std=0.01)
        
    def is_read_head(self):
        return False
    
    def forward(self, embeddings, w_prev):
        # o = self.fc_write(embeddings)
        # k, beta, g, s, gamma, e, a = _split_cols(embeddings.clone(), self.write_lengths)
        mat = embeddings.clone()

        k = self.k_reg(mat)
        beta = F.softplus(self.beta_reg(mat))
        g = torch.sigmoid(self.g_reg(mat))
        s = torch.softmax(self.s_reg(mat), dim=1)
        gamma = 1 + F.softplus(self.gamma_reg(mat))

        e = torch.sigmoid(self.e_reg(mat))
        a = self.a_reg(mat) 

        # e should be in [0, 1] 
        # e = torch.sigmoid(e)
        
        # write to memory
        w = self.memory.address(k, beta, g, s, gamma, w_prev) #self._address_memory(k, beta, g, s, gamma, w_prev)
        self.memory.write(w, e, a)
        
        return w
        