import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class covariance_module(nn.Module) :
    def __init__(self, p=503, K=2, init_log_lamb = 0, device = torch.device('cpu')) :
        super(covariance_module, self).__init__()
        '''
        Log-Cholesky parametrization of Covariance
        Sigma[k] = L[k] @ L[k].T
        '''
        self.p = p
        self.K = K
        self.device = device

        self.L_wo_diag = nn.Parameter((torch.rand(K, int(p * (p-1) / 2), device = device) * (.2/p) - .1/p) * np.exp(0.5 * init_log_lamb))
        self.L_log_diag = nn.Parameter(torch.zeros(K, p, device=device) + 0.5 * init_log_lamb)

    def recover_L(self) : 
        L = torch.cat([torch.diag(torch.exp(self.L_log_diag[k])).unsqueeze(0) for k in range(self.K)], dim=0)
        
        tril_indices = torch.tril_indices(row=self.p, col=self.p, offset=-1)
        L[:, tril_indices[0], tril_indices[1]] = self.L_wo_diag

        return L
    
    def inv_L(self) : 
        return torch.linalg.solve_triangular(self.recover_L(), torch.eye(self.p, device=self.device), upper=False)

    def recover_Sigma(self) : 
        L = self.recover_L()
        return torch.bmm(L, L.transpose(1,2))
    
    def MME_initialize(self, v_list, eps = 1e-6) : 
        m = len(v_list)
        sample_L = torch.linalg.cholesky(
            sum([torch.bmm(v_i.T.unsqueeze(2), v_i.T.unsqueeze(1)) for v_i in v_list]) / (m-1) + 
            torch.eye(self.p, device=self.device).repeat(self.K,1,1) * eps
            )
        
        tril_indices = torch.tril_indices(row=self.p, col=self.p, offset=-1)
        self.L_wo_diag.data = sample_L[:, tril_indices[0], tril_indices[1]]
        for k in range(self.K) : 
            self.L_log_diag.data[k] = torch.log(torch.diag(sample_L[k]) + eps)
    
        return None
    



class large_covariance_module(nn.Module) :
    def __init__(self, p=503, K=2, init_log_lamb=0, device = torch.device('cpu')) :
        super(large_covariance_module, self).__init__()

        self.p = p
        self.K = K
        self.Kp = K * p
        self.device = device

        self.L_wo_diag = nn.Parameter((torch.rand(int(K*p * (K*p-1) / 2), device = device) * .2/(K*p) - .1/(K*p)) * np.exp(0.5 * init_log_lamb))
        self.L_log_diag = nn.Parameter(torch.zeros(K*p, device=device) + 0.5 * init_log_lamb)

    def recover_L(self) : 
        L = torch.diag(torch.exp(self.L_log_diag))
        
        tril_indices = torch.tril_indices(row=self.Kp, col=self.Kp, offset=-1)
        L[tril_indices[0], tril_indices[1]] = self.L_wo_diag

        return L
    
    def inv_L(self) : 
        return torch.linalg.solve_triangular(self.recover_L(), torch.eye(self.Kp, device=self.device), upper=False)

    def recover_Sigma(self) : 
        L = self.recover_L()
        return L @ L.T
        
    def MME_initialize(self, v_list, eps = 1e-6) : 

        m = len(v_list)
        # self.L_log_diag.data = torch.log(sum([torch.pow(v_i,2) for v_i in v_list]) / (m-1) + eps).T
        sample_L
        sample_L = torch.linalg.cholesky(
            sum([v_i.T.flatten().unsqueeze(1) @ v_i.T.flatten().unsqueeze(0) for v_i in v_list]) / (m-1) + 
            torch.eye(self.Kp, device=self.device) * eps
        )
        
        tril_indices = torch.tril_indices(row=self.Kp, col=self.Kp, offset=-1)
        self.L_wo_diag.data = sample_L[:, tril_indices[0], tril_indices[1]]
        self.L_log_diag.data = torch.diag(sample_L)
        return None
    