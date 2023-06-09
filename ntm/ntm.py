import torch
from torch import nn
import torch.nn.functional as F
from typing import Any

class NTM(nn.Module):
    def __init__(self, num_inputs:int, num_outputs:int, controller:Any, memory:Any, heads:list[Any]) -> None:
        super().__init__()
        
        self.num_inputs  = num_inputs
        self.num_outputs  = num_outputs
        self.controller = controller
        self.memory = memory
        self.heads = heads
        self.device_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state: tuple = ()

        self.N, self.M = memory.size()
        # _, self.controller_size = controller.size()    ######
        
        self.num_read_heads = 0
        self.init_r = []
        for head in heads:
            if head.is_read_head():
                init_r_bias = torch.randn(1, self.M) * 0.01
                self.register_buffer("read{}_bias".format(self.num_read_heads), init_r_bias.data)
                self.init_r += [init_r_bias]
                self.num_read_heads += 1
                
        assert self.num_read_heads > 0, "heads list must contain at least a single read head"
        
        # initialise a fully connected layer to produce the actual output:
        # [controller_output; previous_reads] -> output
        # self.fc = nn.Linear(self.controller_size + self.num_read_heads * self.M, num_outputs)
        self.fc = nn.LazyLinear(num_outputs, device=self.device_)
        # self.reset_parameters()
        
    def create_new_state(self, batch_size):
        """
        Creates a new state for the NTM.
        Initializes the Read state, Controller state, state of the heads.
        Do not confuse the Read state with the state of Read head

        Args:
            batch_size (int): size of the batch

        Returns:
            tuple[list[torch.Tensor], tuple[torch.Tensor], list[Tensor]]: Returns the newly created states of the above mentioned components
        """
        init_r  = [r.clone().repeat(batch_size, 1).to(self.device_) for r in self.init_r]
        # controller_state = self.controller.create_new_state(batch_size)
        heads_state = [head.create_new_state(batch_size) for head in self.heads]
        # return init_r, controller_state, heads_state
        return init_r, heads_state
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight, gain=1)
        nn.init.normal_(self.fc.bias, std=0.01)
        
    def forward(self, x, prev_state, training=True):
        """ 
        

        Args:
            x (_type_): _description_
            prev_state (_type_): _description_

        Returns:
            _type_: _description_
        """
        # unpack previous states
        # prev_reads, prev_controller_state, prev_heads_states = prev_state
        prev_reads, prev_heads_states = prev_state

        # use the controller to get an embeddings
        # inp = torch.cat([x] + prev_reads, dim=1)
        inp = x
        # print(inp)
        # print(inp.size())
        # controller_outp, controller_state = self.controller(inp, prev_controller_state) # prev_controller_state = float32
        controller_outp = self.controller(inp, training).squeeze() # this controller is FCN
        # print(controller_outp)
        # Read/Write from the list of heads
        reads = []
        heads_states = []
        for head, prev_head_state in zip(self.heads, prev_heads_states):
            if head.is_read_head():
                r, head_state = head(controller_outp, prev_head_state)
                reads += [r]
            else:
                head_state = head(controller_outp, prev_head_state)
            heads_states += [head_state]
            
        # Generate Output 
        inp2 = torch.cat([controller_outp] + reads, dim=1)
        # print(len(reads))
        # inp2 = torch.cat(reads, dim=1)
        # inp2 = controller_outp
        o =  self.fc(inp2) #torch.sigmoid(self.fc(inp2))
        
        # Pack the current state
        # self.state = (reads, controller_state, heads_states)
        self.state = (reads, heads_states)
        
        return F.softmax(o), self.state
    
    def __repr__(self):
        return f"""Controller: {self.controller}, Num of Inputs: {self.num_inputs}, Num of Outputs: {self.num_outputs}, Memory: {self.memory}"""
        