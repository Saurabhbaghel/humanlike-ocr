import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

def _update_usage_weights(wu_t_1, wr_t, ww_t, gamma):
    """
    updating the weights of the previous usage weights wu_t_1
    wu_t_1 : the previous usage weights
    wr_t : read weight which is wc at time t
    ww_t : the write weight at time t
    gamma : decay parameter
    """
    return gamma*wu_t_1 + wr_t + ww_t

def _m(arr, n):
    """
    to get the nth smallest element in arr
    """
    # flatten the arr first to find the nth smallest element aomng the whole vector
    return torch.kthvalue(arr.flatten(), n, -1)[0]

def _least_used_weights(wu_t, n):
    """
    wu_t : the usage weights
    n : to find the nth smallest element of the vector = number of reads to memory 
    """
    wlu_t = torch.where(wu_t > _m(wu_t, n), 0, 1)
    return wlu_t
