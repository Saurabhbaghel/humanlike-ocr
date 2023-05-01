import os, sys
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
import cv2 as cv
from ntm.ntm import NTM
from ntm.controller import LSTMController
from ntm.head import NTMWriteHead, NTMReadHead
from ntm.memory import NTMMemory
from ntm.encapsulated import EncapsulatedNTM
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.io.image import read_image
import torchvision.transforms as transforms
from torchmetrics.classification import MulticlassAccuracy
from sklearn.model_selection import train_test_split
import pandas as pd


path_csv_labels, user_batch_size, user_img_dir = sys.argv[1], sys.argv[2], sys.argv[3]


torch.autograd.set_detect_anomaly(True)
device_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

N, M = 120, 120

num_inputs = 240
num_outputs = 37
num_layers = 4

controller_size = 37
num_heads = 2

# num_inputs + M * num_heads

# getting the features index
features_= pd.read_csv("/content/humanlike-ocr/data/final_80_features.csv")
feats = features_.columns[:-1]
feats_idx = np.array(feats).astype(np.int32)


# Dataset Preparation
class AtomicCharsDataset(Dataset):
    def __init__(
        self, 
        annotations_file:pd.DataFrame,
        img_dir:str, 
        target_transform:None
        ) -> None:
        self.image_labels = pd.read_csv(annotations_file).reset_index(drop=True)
        self.img_dir = img_dir
        self.transforms = transforms.Compose([
            transforms.Resize(40),
            transforms.CenterCrop(40),
            transforms.ConvertImageDtype(torch.float),
            # transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.target_transform = target_transform
        self.feature_extractor_ = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor_.children())[:-1]).to(device_)
        
    def __len__(self):
        return len(self.image_labels)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.image_labels.iloc[index, 0])
        image = read_image(img_path)
        image = self.transforms(image).to(device_)
        c, h, w = image.size()
        slice_width = h//3
        head, body, tail = image[:, :slice_width, :], image[:, slice_width: 2*slice_width, :], image[:, 2*slice_width: 3*slice_width]
        stacked_ = torch.stack([head, body, tail], dim=0)
        
        # getting the features of the three parts of the image
        features = self.feature_extractor(stacked_).squeeze()[:, feats_idx] # getting the top 80 features only
        
        # now the features which are stacked are to concatenated to get an array of the features
        feat_list = [features[i, :] for i in range(features.size()[0])]
        feats_catted = torch.cat(feat_list, dim=0)
        label = self.image_labels.iloc[index, 1]
        if self.target_transform:
            label = self.target_transform(label)
        return feats_catted, label
    

# reading the label csv
if path_csv_labels is None or not isinstance(path_csv_labels, str):
    raise ValueError(f"{path_csv_labels} is not a path to the label csv.")
else:
    data = pd.read_csv(path_csv_labels) # old csv with fewer images  pd.read_csv("/content/humanlike-ocr/data/annotations_atomic_char_5iter.csv")

# splitting the data into training and validation
training_data, val_data = train_test_split(data, test_size=0.3, train_size=0.7, random_state=4340, shuffle=True) 

# assert isinstance(training_csv_path, str)
# assert isinstance(val_csv_path, str)

training_csv_path = os.path.join("/content/humanlike-ocr/data",f"training_{len(training_data)}.csv")
val_csv_path = os.path.join("/content/humanlike-ocr/data", f"val_{len(val_data)}.csv")


if not os.path.exists(training_csv_path):
    training_data.to_csv(training_csv_path, index=False)
if not os.path.exists(val_csv_path):
    val_data.to_csv(val_csv_path, index=False)
    
# names of the files/folders
img_dir = user_img_dir
ann_train = training_csv_path
ann_val = val_csv_path

# creating training dataset and validation dataset
train_dataset = AtomicCharsDataset(ann_train, img_dir, None)
val_dataset = AtomicCharsDataset(ann_val, img_dir, None)

BATCH_SIZE = int(user_batch_size)

assert len(train_dataset) % BATCH_SIZE == 0, "batch size should divide length of the train dataset."
assert len(val_dataset) % BATCH_SIZE == 0, "batch size should divide length of the val dataset."


# making the dataloader for both set of points
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


# Preparing the model for training
controller = LSTMController(num_inputs + M*num_heads, num_outputs, num_layers)
controller.to(device_)
memory = NTMMemory(N, M)
memory.to(device_)
read_head = NTMReadHead(memory, controller_size)
read_head.to(device_)
write_head = NTMWriteHead(memory, controller_size)
write_head.to(device_)

heads = nn.ModuleList([])

for i in range(num_heads):
    heads += [
        read_head,
        write_head
    ]
    
# instantiating the model  
ntmcell = NTM(num_inputs, num_outputs, controller, memory, heads)


ntmcell.to(device_)

 
# defining the loss function
loss_fn = torch.nn.CrossEntropyLoss()

# defining the optimizer 
optimizer = torch.optim.Adam(ntmcell.parameters(), lr=0.005) 
 
# defining the metric
acc = MulticlassAccuracy(37).to(device_) 
 
# training loop for one epoch
def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.0
    last_loss = 0.0
    
    for  i, data in enumerate(train_dataloader):
        inputs, labels = data[0].to(device_), data[1].to(device_)
        
        # zero gradients for every batch
        optimizer.zero_grad()
        
        #resetting the memory for new epoch
        memory.reset(BATCH_SIZE)
        prev_state = ntmcell.create_new_state(BATCH_SIZE)
        
        # make predictions for this batch
        outputs, _ = ntmcell(inputs, prev_state)
        outputs = outputs.type(torch.float)
        # pred_label = torch.argmax(outputs,dim=1)
        # compute the loss and its gradients
        accuracy = acc(outputs, labels)
        labels = torch.nn.functional.one_hot(labels, num_classes=37)
        labels = labels.type(torch.float)
        loss = loss_fn(outputs, labels)
        loss.backward()
        
        parameters = list(filter(lambda p: p.grad is not None, ntmcell.parameters()))
        for p in parameters:
            p.grad.data.clamp_(-10, 10)
        # adjust the learning rate
        optimizer.step()
        
        # gathering data and report
        running_loss += loss.item()
        if i % 10 == 9:
            last_loss = running_loss / 10
            print("batch {} loss: {} accuracy: {}".format(i+1, last_loss, accuracy))
            tb_x = epoch_index * len(train_dataloader) + i + 1
            tb_writer.add_scalar("Loss/train", last_loss, tb_x)
            running_loss = 0.0
            
    return last_loss

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter("/content/humanlike-ocr/runs/atom_trainer_{}".format(timestamp))

epoch_number = 0

EPOCHS = 100

best_vloss = 1_000_000.

# training loop for all the epochs
for epoch in range(EPOCHS):
    print("EPOCH {}".format(epoch_number + 1))
    

    
    ntmcell.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)
    
    ntmcell.train(False)
    
    with torch.no_grad():
        vprev_state = ntmcell.create_new_state(BATCH_SIZE)
        running_vloss = 0.0
        for i, vdata in enumerate(val_dataloader):
            vinputs, vlabels = vdata[0].to(device_), vdata[1].to(device_)
            voutputs, vprev_state = ntmcell(vinputs, vprev_state)
            voutputs = voutputs.type(torch.float)
            vlabels = torch.nn.functional.one_hot(vlabels, num_classes=37)
            vlabels = vlabels.type(torch.float)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss
    
    avg_vloss = running_vloss / (i + 1)
    print("LOSS train {} valid {}".format(avg_loss, avg_vloss))
    
    writer.add_scalars(
        "Training vs. Validation Loss", 
        {"Training": avg_loss, "Validation": avg_vloss},
        epoch_number + 1
        )
    writer.flush()
    
    # track best performance and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = "model_{}_{}".format(timestamp, epoch_number)
        torch.save(ntmcell.state_dict(), model_path)
        
    epoch_number += 1
 
 
 
 
 
 