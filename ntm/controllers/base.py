import torch.nn as nn

class BaseController(nn.Module):
    def __init__(self):
        super().__init__()

    def __name__(self):
        return NotImplementedError

    # def extra_repr(self):
    #     return f"{self.__name__()}"