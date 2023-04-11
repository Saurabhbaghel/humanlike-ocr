import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

def _convolve(w, s):
    assert s.size(0) == 3
    t = torch.cat([w[-1:], w, w[:1]])
    c = F.conv1d(t.view(1, 1, -1), s.view(1, 1, -1)).view(-1)
    return c

class NTMMemory(nn.Module):
    def __init__(self, N:int, M:int) -> None:
        """initializes the NTM memory

        Args:
            N (int): number of rows
            M (int): number of columns
        """        
        super().__init__()
        
        self.N = N
        self.M = M
        self.device_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        self.register_buffer("mem_bias", torch.Tensor(N, M))
        
        # initialize memory bias
        stdev = 1/(np.sqrt(N+M))
        nn.init.uniform_(self.mem_bias, -stdev, stdev)
        
    def reset(self, batch_size:int):
        """Initialize memory from bias, for start-of-sequence. 

        Args:
            batch_size (int): _description_
        """
        self.batch_size = batch_size
        self.memory = self.mem_bias.clone().repeat(batch_size, 1, 1)
        
    def size(self):
        return self.N, self.M
    
    def read(self, w_:torch.Tensor):
        """_summary_

        Args:
            w_ (torch.Tensor): _description_
        """
        w = w_.clone()
        return torch.matmul(w.unsqueeze(1), self.memory).squeeze(1)
    
    def write(self, w, e, a):
        self.prev_mem = self.memory
        self.memory = torch.Tensor(self.batch_size, self.N, self.M)
        erase = torch.matmul(w.unsqueeze(-1), e.unsqueeze(1))
        add = torch.matmul(w.unsqueeze(-1), a.unsqueeze(1))
        self.memory = self.prev_mem * (1-erase) + add
        
    def address(self, k, beta, g, s, gamma, w_prev):
        """_summary_

        Args:
            k (_type_): key vector at time t
            beta (_type_): key strength at time t
            g (_type_): interpolation gate at time t
            s (_type_): shift weighting at time t
            gamma (_type_): sharpening factor at time t
            w_prev (_type_): weight vector at at time t-1

        Returns:
            w (_type_): weight at time t
        """
        # content focus
        wc = self._similarity(k, beta).to(self.device_)
        
        # location focus
        wg = self._interpolate(w_prev, wc, g).to(self.device_)
        w_hat = self._shift(wg, s).to(self.device_)    # convolutional shift
        w = self._sharpen(w_hat, gamma).to(self.device_)
        return w
    
    def _similarity(self, k, beta):
        k = k.view(self.batch_size, 1, -1).to(self.device_)
        w = F.softmax(beta * F.cosine_similarity(self.memory + 1e-16, k+1e-16, dim=-1), dim=1).to(self.device_)
        return w
    
    def _interpolate(self, w_prev, wc, g):
        return g * wc + (1-g) * w_prev
    
    def _shift(self, wg, s):
        result = torch.zeros(wg.size())
        for b in range(self.batch_size):
            result[b] = _convolve(wg[b], s[b])
        return result
    
    def _sharpen(self, w_hat, gamma):
        w = w_hat ** gamma
        w = torch.div(w, torch.sum(w, dim=1).view(-1, 1) + 1e-16)
        return w         