# re-designed adjustment

import numpy as np
import torch
import torch.nn as nn

from tqdm.auto import tqdm
from sklearn.linear_model import LinearRegression


def adj(train_Gamma, train_random, v_list, train_cluster, test_Gamma, test_fixed, device=torch.device('cpu')) : 

    N, K = train_random.shape
    m = len(v_list)

    train_v_list = [torch.zeros_like(train_Gamma) for _ in range(K)]

    for i in range(m) : 
        for k in range(K) : 
            train_v_list[k][train_cluster[i]] = v_list[i][:,k].repeat(len(train_cluster[i]), 1)

    w_list = [LinearRegression(fit_intercept=False) for _ in range(K)]
    
    for k in range(K) : 
        w_list[k].fit(X = train_Gamma.detach().cpu(), y = train_v_list[k].detach().cpu())
    
    nu_list = [torch.as_tensor(w.coef_).T.to(device) for w in w_list]
    test_adjusted = torch.stack([
        torch.sum(test_Gamma * (test_Gamma @ nu), dim=1) 
        for nu in nu_list
    ]).T

    return test_fixed, test_fixed + test_adjusted