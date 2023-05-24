import numpy as np
import torch


from tqdm import tqdm
from xgboost import XGBClassifier
import copy 

def test_input_xgboost(out, y_error, split_masks, n, train_mask, test_mask, save_path):
    y_error = y_error.view(y_error.shape[0],1)    #torch.Size([2449029, 1])
    y_error = y_error.float()

    cnt_list = []

    print("xgboost train : {}".format(n))

    step_num = 10
    train_M = copy.deepcopy(split_masks[train_mask])
    test_M = copy.deepcopy(split_masks[test_mask])
    test_failure_mask = torch.zeros_like(split_masks[test_mask])

    budget_list = []
    for i in range(step_num-1):
        budget_list.append(int(n/step_num))
    budget_list.append(n - int(n/step_num) * (step_num-1) )
    
    cnt_re = [0]
    cnt_list = []
    for step in tqdm(range(step_num)):
        xgb_train_x = np.array(torch.Tensor.tolist(out[train_M]))     
        xgb_train_y = np.array(torch.Tensor.tolist(y_error[train_M]))
        
        xgb_test_x = torch.Tensor.tolist(out[test_M])
        xgb_test_y = torch.Tensor.tolist(y_error[test_M])
        
        xgb_model = XGBClassifier(learning_rate=0.003, max_delta_step = 0, min_child_weight = 7, max_depth = 7, colsample_bytree=0.5, objective='rank:pairwise')
        xgb_model.fit(xgb_train_x, xgb_train_y)
        
        xgb_predict_y = xgb_model.predict_proba(xgb_test_x)[:,1:]
        
        priority_list_index = sorted(zip(xgb_predict_y, range(len(xgb_predict_y))))
        priority_list_index.sort(key = lambda x : x[0], reverse=True)
        priority_index = [x[1] for x in priority_list_index]
        
        xgb_test_y = np.array(xgb_test_y)
        priority_index = np.array(priority_index, dtype=np.int)

        for i in range(budget_list[step]):
            cnt_re.append(cnt_re[-1] + xgb_test_y[priority_index[i], 0])
            
        cnt_list.append( (int)(xgb_test_y[priority_index[0:budget_list[step]], 0].sum()) )
        
        test_M_index = torch.arange(test_M.shape[0], dtype=torch.int32)
        test_M_index = test_M_index[test_M]
        test_M_index = test_M_index[priority_index[0:budget_list[step]]]
        
        train_M[test_M_index] = 1
        test_M[test_M_index] = 0
        
        test_failure_mask[test_M_index] = 1
    torch.save(test_failure_mask, save_path + "/select_mask_noagg_gini")
    torch.save(cnt_re, save_path + "/cnt_re_noagg_gini")
    print(test_failure_mask.sum())

    print('xgb : test input prioritization')
    
    ATRC = 0
    for i in range(1, len(cnt_re)):
        ATRC += cnt_re[i] / i
    print("ATRC : {}".format( ATRC / (len(cnt_re) -1) ))

    return cnt_re
