import numpy as np
import os
import struct
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

dataset_name = "data_ogbn-products"
dataset_name = "data_Flickr"
dataset_name = "data_Reddit"

out = torch.load(dataset_name + '/out/out')
y_class = torch.load(dataset_name + '/y/y_class')
y_error = torch.ones_like(y_class)



cnt = out.max(-1)[1].eq(y_class).sum().item()


for i in tqdm(range(y_class.shape[0])):
    if out[i].max(-1)[1].eq(y_class[i]).item():
        y_error[i] = 0
print(y_error)
torch.save(y_error, dataset_name + '/y/y_error')
print(y_error.sum() / y_error.shape[0])

