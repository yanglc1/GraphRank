import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn import Sequential


class MLP(torch.nn.Module):
    def __init__(self, n_feature, n_out):
        super(MLP,self).__init__()
        self.model = Sequential(
            nn.Linear(n_feature,256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Linear(64,16),
            nn.ReLU(),
            nn.Linear(16,n_out)
        )
       
    def forward(self, x):
        x = self.model(x)
        return x

class myDataset(Dataset):
    def __init__(self, out, y_error):
        self.out = out
        self.y_error = y_error

    def __getitem__(self, index):
        return self.out[index], self.y_error[index]

    def __len__(self):
        return self.out.shape[0]


def test_input_MLP(out, y_error, split_masks, n, train_mask):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    batch_size = 32
    out = out.to(torch.float32)

    y = F.one_hot(y_error, 2)
    y = y.float()

    train_num = split_masks[train_mask].sum()
    test_num = split_masks["test"].sum()
    
    dataset_train = myDataset(out[split_masks[train_mask]], y[split_masks[train_mask]])
    dataset_test = myDataset(out[split_masks['test']], y[split_masks['test']])

    data_load_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    data_load_test = DataLoader(dataset_test, batch_size=4096, shuffle=False)

    model = MLP(out.shape[1], y.shape[1]) 
    model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters()) 


    for epoch in range(1, 21):
        loss_sum = 0
        # print("开始训练: ", epoch)
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

            loss_sum += loss

        if epoch % 5  == 0:
            model.eval()
            # print("\n###########start test############\n")

            cnt = 0
            for data, y in data_load_train:
                data = data.to(device)
                y = y.to(device)

                outs = model(data)
                
                cnt += outs.max(-1)[1].eq(y.max(-1)[1]).sum().item()
                
                outs = torch.sigmoid(outs)
                cnt += torch.ceil(outs[:, 0] - 0.5).eq(y[:, 0]).sum().item()
            print("train acc : {}".format(cnt/train_num)) 

            cnt = 0
            for data, y in data_load_test:
                data = data.to(device)
                y = y.to(device)

                outs = model(data)

                cnt += outs.max(-1)[1].eq(y.max(-1)[1]).sum().item()
                
                outs = torch.sigmoid(outs)
                cnt += torch.ceil(outs[:, 0] - 0.5).eq(y[:, 0]).sum().item()
            print("test acc : {}".format(cnt/test_num)) 
       
   
    model.eval()
    #计算 test input priority
    print("\n test input \n")

    priority_list = []
    y_test_input = []
    for data, y in data_load_test:
        data = data.to(device)
        y = y.to(device)

        outs = model(data)
        outs = F.softmax(outs, dim=1)
       
        priority_list.extend(outs[ : , 1].tolist())
        y_test_input.extend(y[ : , 1].tolist())

    
    step_num = 1

    budget_list = []
    for i in range(step_num-1):
        budget_list.append(int(n/step_num))
    budget_list.append(n - int(n/step_num) * (step_num-1) )
    
    
    priority_list_index = sorted(zip(priority_list, range(len(priority_list))))
    priority_list_index.sort(key = lambda x : x[0], reverse=True)
    priority_index = [x[1] for x in priority_list_index]

    
    y_test_input = np.array(y_test_input)
    priority_index = np.array(priority_index, dtype=np.int)

    cnt_re = [0]
    for step in tqdm(range(step_num)):
        for i in range(budget_list[step]):
            cnt_re.append(cnt_re[-1] + y_test_input[priority_index[i]])

    ATRC = 0
    for i in range(1, len(cnt_re)):
        ATRC += cnt_re[i] / i
    print("ATRC : {}".format( ATRC / (len(cnt_re) -1) ))
        
    print('MLP : test input prioritization : ', end='')

