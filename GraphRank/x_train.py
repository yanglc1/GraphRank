import numpy as np
import os
import struct
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

class MLP(torch.nn.Module):
    def __init__(self, n_feature, n_out):
        super(MLP,self).__init__()
        
        self.linear1=torch.nn.Linear(n_feature,256)
        self.relu=torch.nn.ReLU()

        self.linear2=torch.nn.Linear(256,128)
        self.relu2=torch.nn.ReLU()

        self.linear3=torch.nn.Linear(128,64)
        self.relu3=torch.nn.ReLU()

        self.linear4=torch.nn.Linear(64,n_out)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)

        x = self.linear2(x)
        x = self.relu2(x)

        x = self.linear3(x)
        x = self.relu3(x)

        x = self.linear4(x)
        return x

class myDataset(Dataset):
    def __init__(self, out, y_error):
        self.out = out
        self.y_error = y_error

    def __getitem__(self, index):
        return self.out[index], self.y_error[index]

    def __len__(self):
        return self.out.shape[0]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


dataset_name = "data_ogbn-products"
dataset_name = "data_Flickr"
dataset_name = "data_Reddit"

x = torch.load(dataset_name + '/data_x/x')
y_class = torch.load(dataset_name + '/y/y_class') 
split_masks = torch.load(dataset_name + '/split/split_ori')

if dataset_name == "data_ogbn-products":
    n_class = 47
elif dataset_name == "data_Flickr":
    n_class = 7
elif dataset_name == "data_Reddit":
    n_class = 41
    # temp = split_masks['train']
    # split_masks['train'] = split_masks['test']
    # split_masks['valid'] = split_masks['valid']
    # split_masks['test'] = temp
    
batch_size = 16


x = x.to(torch.float32)
y_one_hot = F.one_hot(y_class, n_class)
y_one_hot = y_one_hot.float()

dataset_train = myDataset(x[split_masks['train']], y_one_hot[split_masks['train']])
dataset_valid = myDataset(x[split_masks['valid']], y_one_hot[split_masks['valid']])
dataset_test = myDataset(x[split_masks['test']], y_one_hot[split_masks['test']])
dataset_all = myDataset(x, y_one_hot)

data_load_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
data_load_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True)
data_load_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
data_load_all = DataLoader(dataset_all, batch_size=batch_size, shuffle=False)

model = MLP(x.shape[1], y_one_hot.shape[1]) 
model.to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters()) 

epoch = 0
max_acc = 0
test_acc = 0
stop = 0
x_out = []

while(1):
    print("开始训练: ", epoch)
    model.train()  
    loss = 0.0
    for data, y in data_load_train:
        data = data.to(device)
        y = y.to(device)

        outs = model(data)
        
        optimizer.zero_grad()
        loss = loss_function(outs, y)
        loss.backward() 
        optimizer.step()
    epoch += 1

    if epoch % 5  == 0:
        model.eval()
        print("\n###########start test############\n")

        cnt = 0
        for data, y in tqdm(data_load_valid):
            data = data.to(device)
            y = y.to(device)

            outs = model(data)
            cnt += outs.max(-1)[1].eq(y.max(-1)[1]).sum().item()
            acc = cnt/split_masks['valid'].sum().item()
            # cnt += torch.ceil(outs[:, 0] - 0.5).eq(y[:, 0]).sum().item()
        print("valid acc : {}".format(acc))
        

        if max_acc < acc:
            max_acc = acc
            stop = 0
            
            x_out = []
            cnt = 0
            for data, y in tqdm(data_load_all):
                data = data.to(device)
                y = y.to(device)
                outs = model(data)
                x_out.extend(outs.tolist())
                
     
            cnt += torch.tensor(x_out)[split_masks['test']].max(-1)[1].eq(y_one_hot[split_masks['test']].max(-1)[1]).sum().item()
            test_acc = cnt/split_masks['test'].sum().item()
            print("test acc : {}".format(test_acc))
        else :
            stop += 1

        if stop > 5:
            break

torch.save(torch.tensor(x_out, dtype=torch.float32), dataset_name + '/data_x/x_out')
print(max_acc, test_acc)