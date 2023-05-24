import torch
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from sklearn.decomposition import PCA, FastICA

def process_HE(out, save_path):
    logpx = F.log_softmax(out, dim=1).tolist()
    logpx = np.array(logpx)
    out_HE = torch.tensor(-np.sum(np.multiply(logpx, np.exp(logpx)), axis = 1)).unsqueeze(1)
    return out_HE


def process_con(out, save_path):
    out_con = F.softmax(out, dim=1).tolist()
    out_con = torch.tensor(1 - np.max(out_con, axis=1)).unsqueeze(1)
    return out_con
   

# def process_PCA(out, n_components, save_path):
#     out = out.tolist()
#     pca = PCA(n_components=n_components)
#     out_pca = pca.fit_transform(out)
#     out_pca = torch.tensor(out_pca, dtype=torch.float32)
#     torch.save(out_pca, save_path)

# def process_ICA(out, n_components, save_path):
#     out = out.tolist()
#     ica = FastICA(n_components=n_components)
#     out_ica = ica.fit_transform(out)
#     out_ica = torch.tensor(out_ica, dtype=torch.float32)
#     torch.save(out_ica, save_path)
    
# def process_soft_mean(outs, save_path):
#     for i in tqdm(range(len(outs))):
#         outs[i] = F.softmax(outs[i], dim=1).tolist()
#     outs = np.array(outs)
#     out_mean = torch.tensor(np.mean(outs, axis=0))
#     torch.save(out_mean, save_path)