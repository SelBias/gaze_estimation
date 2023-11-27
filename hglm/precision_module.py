import torch
import numpy as np
import torch.nn as nn

class precision_module(nn.Module) :
    def __init__(self, p=503, K=2, init_log_lamb = 0, device = torch.device('cpu'), dtype=torch.float) :
        super(precision_module, self).__init__()
        self.p = p
        self.K = K
        self.device = device
        self.dtype=dtype

        self.L_wo_diag = nn.Parameter((torch.rand(K, int(p * (p-1) / 2), device = device, dtype=dtype) * .2/p - .1/p) * np.exp(-0.5 * init_log_lamb))
        self.L_log_diag = nn.Parameter(torch.zeros(K, p, device=device, dtype=dtype) - 0.5 * init_log_lamb)

    def recover_L(self) : 
        L = torch.cat([torch.diag(torch.exp(self.L_log_diag[k])).unsqueeze(0) for k in range(self.K)], dim=0)
        
        tril_indices = torch.tril_indices(row=self.p, col=self.p, offset=-1)
        L[:, tril_indices[0], tril_indices[1]] = self.L_wo_diag.data
        return L
    
    def recover_LT(self) : 
        return self.recover_L().mT
    
    def recover_precision(self) : 
        L = self.recover_L()
        return L @ L.mT
    
    def MME_initialize(self, v_list, eps = 1e-6) : 
        m = len(v_list)
        sample_variance = sum([torch.bmm(v_i.T.unsqueeze(2), v_i.T.unsqueeze(1)) for v_i in v_list]) / (m-1) + torch.eye(self.p, device=self.device).repeat(self.K,1,1) * eps

        sample_var_cholesky = torch.linalg.cholesky(sample_variance)
        sample_precision = torch.cholesky_inverse(sample_var_cholesky)
        sample_prec_cholesky = torch.linalg.cholesky(sample_precision)
        
        tril_indices = torch.tril_indices(row=self.p, col=self.p, offset=-1)
        self.L_wo_diag.data = sample_prec_cholesky[:, tril_indices[0], tril_indices[1]]
        for k in range(self.K) : 
            self.L_log_diag.data[k] = torch.log(torch.diag(sample_prec_cholesky[k]) + eps)

        return None



class large_precision_module(nn.Module) :
    def __init__(self, p=503, K=2, device = torch.device('cpu')) :
        super(large_precision_module, self).__init__()

        self.p = p
        self.K = K
        self.Kp = K * p
        self.device = device

        self.L_wo_diag = nn.Parameter((torch.rand(int(K*p * (K*p-1) / 2), device = device) * .2/(K*p) - .1/(K*p)))
        self.L_log_diag = nn.Parameter(torch.zeros(K*p, device=device))

    def recover_L(self) : 
        L = torch.diag(torch.exp(self.L_log_diag))
        
        tril_indices = torch.tril_indices(row=self.Kp, col=self.Kp, offset=-1)
        L[tril_indices[0], tril_indices[1]] = self.L_wo_diag.data

        return L
    
    def recover_LT(self) : 
        return self.recover_L().mT

    def recover_precision(self) : 
        L = self.recover_L()
        return L @ L.mT
        
    def MME_initialize(self, v_list, eps = 1e-6) : 
        m = len(v_list)
        sample_variance = sum([v_i.T.flatten().unsqueeze(1) @ v_i.T.flatten().unsqueeze(0) for v_i in v_list]) / (m-1) + torch.eye(self.Kp, device=self.device) * eps

        sample_var_cholesky = torch.linalg.cholesky(sample_variance)
        sample_precision = torch.cholesky_inverse(sample_var_cholesky)
        sample_prec_cholesky = torch.linalg.cholesky(sample_precision)
        
        tril_indices = torch.tril_indices(row=self.Kp, col=self.Kp, offset=-1)
        self.L_wo_diag.data = sample_prec_cholesky[tril_indices[0], tril_indices[1]]
        self.L_log_diag.data = torch.log(torch.diag(sample_prec_cholesky) + eps)




