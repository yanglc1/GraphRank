import numpy as np
import torch

import torch.nn.functional as F

from tqdm import tqdm


from Learn2Rank.XGboost import test_input_xgboost
from Learn2Rank.Linear import test_input_linear
from Learn2Rank.MLP import test_input_MLP
from Learn2Rank.ResNet import test_input_ResNet

from data_process import process_HE, process_con, process_PCA, process_ICA, process_soft_mean


def make_y_error(out, y_class, save_path):
    y_error = torch.zeros(size=[y_class.shape[0]], dtype=torch.int64)
    for i in tqdm(range(y_class.shape[0])):
        if out[i].max(-1)[1].item() != y_class[i]:
            y_error[i] = 1
        else :
            y_error[i] = 0
    return y_error
  

def make_test(select):
    datasets = ["data_ogbn-products", "data_Reddit", "data_Flickr"]
    models = ["/EnGCN", "/SIGN", "/GraphSAGE", "/ClusterGCN"]
         
    dataset_name = datasets[select[0]]
    model_name = models[select[1]]
        
    print("dataset : {}, model : {}".format(dataset_name, model_name))
    
    out = torch.load(dataset_name + model_name +'/out/out')   

    for i in tqdm(range(10)):      # get probabilistic output attributes
        out_mean += torch.load(dataset_name + model_name +'/drop_x_edge/out_{}'.format(i))
    out_mean /= 10
    out_unc = F.softmax(out_mean, dim=1)
    
    logpx = F.log_softmax(out_mean, dim=1).tolist()
    logpx = np.array(logpx)
    uncertainty = -np.sum(np.multiply(logpx, np.exp(logpx)), axis = 1)
    uncertainty = torch.tensor(uncertainty).unsqueeze(dim=1)

    x_out = torch.load(dataset_name + '/data_x/x_out')     # get graph node attributes
    
    edge_index = torch.load(dataset_name + '/edge/edge_index')       # 2*n,   [[0,0,0,1,1,1,2 ], [23,24,25,152,123,152,1234 ……]]  edge(0,23)(0,24)……(2,1234)
    split_masks = torch.load(dataset_name + '/split/split_masks')    

    deg = torch.zeros(size=(split_masks.shape[0], 1), dtype=torch.int64)    # get graph structure attributes
    for i in tqdm(range(edge_index.shape[1])):
        deg[edge_index[0][i]] += 1
    
    y_class = torch.load(dataset_name + '/y/y_class')
    y_error = make_y_error(out, y_class)          # 分类错误的样本置1，正确的样本置0，二分类  

    if dataset_name == "data_Reddit" or dataset_name == "Flickr":
        temp = split_masks['train']
        split_masks['train'] = split_masks['test']
        split_masks['valid'] = split_masks['valid']
        split_masks['test'] = temp
    
    T = {}
    for i in tqdm(range(edge_index.shape[1])):
        start = edge_index[0][i]
        end = edge_index[1][i]
        
        if start not in T.keys():
            T[start] = []
        T[start].append((end))
    
    
    # T = torch.load("/data/ylc/" + dataset_name +'/edge/T')

    HE = process_HE(out)
    x_HE = process_HE(x_out)
   

    out = F.softmax(out, dim=1)
    x_out = F.softmax(x_out, dim=1)
    DeepGini = 1 - torch.sum(torch.pow(out, 2), dim=1)
    DeepGini = DeepGini.unsqueeze(dim=1)

    out_last = torch.zeros(size=(out.shape[0],0), dtype=torch.float32)

    out_last = torch.cat((out_last, out), dim=1)
    out_last = torch.cat((out_last, x_out), dim=1)
    out_last = torch.cat((out_last, HE), dim=1)
    out_last = torch.cat((out_last, x_HE), dim=1)
    out_last = torch.cat((out_last, deg), dim=1)
    out_last = torch.cat((out_last, uncertainty), dim=1) 
    out_last = torch.cat((out_last, DeepGini), dim=1)

    out_agg = torch.zeros_like(out_last, dtype=torch.float32)   # Attributes Enhancement
    for i in tqdm(range(len(T))):
        out_nei = out_last[T[i]]
        out_nei_mean = torch.mean(out_nei, dim=0)
        out_agg[i] = out_nei_mean
    out_last = torch.cat((out_last, out_agg), dim=1)


    test_input_xgboost(out_last, y_error, split_masks, n, "valid", "test", "/data/ylc/" + dataset_name + model_name)

    return
    
   
   
   
if __name__ == "__main__":

    for select in [[0,0],[0,1],[0,2],[0,3], [1,0],[1,1],[1,2],[1,3], [2,0],[2,1],[2,2],[2,3]]:
        make_test(select)
        print()

   

    