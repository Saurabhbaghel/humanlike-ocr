import torch
from torchmetrics.classification import MulticlassAccuracy, MulticlassAveragePrecision, MulticlassF1Score
from ntm.encapsulated import EncapsulatedNTM
from ntm.controllers import LinearNetController, LSTMController, ConvNetController

class configuration:
    def __init__(self) -> None:
        pass
    
    def constants(self, 
                  num_inputs=1, 
                  num_outputs=44, 
                  controller_size=3136, 
                  controller_layers=1, 
                  batch_size=2,
                  num_classes=44,
                  num_epochs = 50,                  
                  num_heads=4, N=10, M=10,
                  controller_=ConvNetController
                  ):
        """
        Initializes the constants

        Args:
            num_inputs (int, optional): _description_. Defaults to 1.
            num_outputs (int, optional): _description_. Defaults to 44.
            controller_size (int, optional): _description_. Defaults to 3136.
            controller_layers (int, optional): _description_. Defaults to 1.
            batch_size (int, optional): _description_. Defaults to 2.
            num_classes (int, optional): _description_. Defaults to 44.
            num_heads (int, optional): _description_. Defaults to 4.
            N (int, optional): _description_. Defaults to 10.
            M (int, optional): _description_. Defaults to 10.
        """
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.controller_size = controller_size #1024
        self.controller_layers = controller_layers
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.num_heads = num_heads
        self.N, self.M = N, M
        self.controller_ = controller_
        

    
    def paths(self, printed:bool, train:bool):
        """returns the paths of the csv of labels and the directory 
           having the images

        Args:
            printed (bool): True if paths for printed dataset is required
            train (bool): True if paths for the training dataset is required

        Returns:
            tuple[str]: path of the annotation csv and image dir 
        """        
        
        
        if printed:
            annotation_printed = "data/atoms_4300.csv"
            printed_img_dir = "data/atoms_4300"
            return annotation_printed, printed_img_dir

        
        else:
            if train:
                
                # handwritten train dir
                hw_img_train_dir = "data/atoms_training_handwritten"

                # annotations file for training
                annotation_hw_training = "data/atoms_train_handwritten.csv"

                return annotation_hw_training, hw_img_train_dir
            
            else:        
                # handwritten evaluation dir
                hw_img_val_dir = "data/atoms_evaluation_handwritten"

                # annotations file for validation set
                annotation_hw_validation = "data/atoms_eval_handwritten.csv"

                return annotation_hw_validation, hw_img_val_dir
      

    def net(self):
        """
        to return the encapsulated NTM

        Returns:
            Any: encapsulated NTM
        """
        # defining the network
        return EncapsulatedNTM(
            self.num_inputs,
            self.num_outputs,
            self.controller_size,
            self.controller_layers,
            self.num_heads,
            self.N,
            self.M,
            self.controller_
        )

    def loss_optimizer_metric(self, learning_rate=0.005):
        """
        Initializes the loss, optimizer and metric fn.

        Returns:
            _type_: _description_
        """
        # loss
        # loss_fn = torch.nn.BCELoss()
        loss_fn = torch.nn.CrossEntropyLoss()

        # optimizer
        optimizer = torch.optim.Adam(self.net().parameters(), lr=learning_rate)

        # metric
        # metric = AveragePrecision(task="multiclass", num_classes=num_outputs)
        metric = MulticlassAccuracy(num_classes=self.num_classes)
        
        return loss_fn, optimizer, metric