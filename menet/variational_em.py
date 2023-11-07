import numpy as np
import torch
import torch.nn as nn

def EM_update(y_list, Gamma_list, beta, sigma_sq, Sigma_v, n_list, use_woodbury = False, device=torch.device('cpu')) : 
    '''
    Update v_list, sigma_sq and Sigma_v according to variational SGD + EM algorithm. 

    NAME            SIZE        TYPE                INFO    
    y_list          [m]         tuple               
     - y_i          [n_i x K]   torch tensor        response for i-th subject
    Gamma_list      [m]         tuple
     - Gamma_i      [n_i x p]   torch tensor        design matrix for i-th subject
    beta            [p x K]     torch tensor        coefficient
    v_list          [m]         tuple
     - v_i          [p x K]     torch tensor        random slope for i-th subject
    sigma_sq        [K]         torch tensor        estimated variance of e
    Sigma_v         [K x p x p] torch tensor        estimated variance matrix of random slope v
    n_list          [m]         tuple           
     - n_i                      scalar              number of observations for i-th subject
    '''
    K = sigma_sq.shape[0]
    m = len(y_list)

    f_hat_list = [Gamma_i @ beta for Gamma_i in Gamma_list]

    if use_woodbury : 
        inv_V_list = [
            torch.eye(n_list[i], device=device).repeat(K,1,1) / sigma_sq.view(-1,1,1) - torch.bmm(
                Gamma_list[i].repeat(K,1,1), 
                torch.bmm(
                    torch.linalg.inv(
                        torch.linalg.inv(Sigma_v) + 
                        torch.bmm(Gamma_list[i].T.repeat(K,1,1), Gamma_list[i].repeat(K,1,1)) / sigma_sq.view(-1,1,1)
                    ), 
                    Gamma_list[i].T.repeat(K,1,1)
                )
            ) / sigma_sq.pow(2).view(-1,1,1) for i in range(m)]
    else : 
        inv_V_list = [torch.linalg.inv(
            torch.eye(n_list[i], device=device).repeat(K,1,1) * sigma_sq.view(-1,1,1) + 
            torch.bmm(Gamma_list[i].repeat(K,1,1), torch.bmm(Sigma_v, Gamma_list[i].T.repeat(K,1,1)))
        ) for i in range(m)]

    new_v_list = [
        torch.bmm(
            torch.bmm(
                torch.bmm(
                    Sigma_v, 
                    Gamma_list[i].T.repeat(K,1,1)
                ), inv_V_list[i]
            ), (y_list[i] - Gamma_list[i] @ beta).T.unsqueeze(2)
        ).squeeze(2).T for i in range(m)
    ]
    
    new_eps_list = [
        y_list[i] - f_hat_list[i] - Gamma_list[i] @ new_v_list[i]
        for i in range(m)
    ]

    new_sigma_sq = sum([
        torch.sum(torch.pow(new_eps_list[i], 2), dim=0) + 
        sigma_sq * (n_list[i] - sigma_sq * torch.sum(torch.diagonal(inv_V_list[i], offset=0, dim1=1, dim2=2), dim=1))
        for i in range(m)
    ]) / sum(n_list)

    new_Sigma_v = sum([
        torch.bmm(new_v_list[i].T.unsqueeze(2), new_v_list[i].T.unsqueeze(1)) + Sigma_v - 
        torch.bmm(
            Sigma_v, 
            torch.bmm(
                Gamma_list[i].T.repeat(K,1,1), 
                torch.bmm(
                    inv_V_list[i], 
                    torch.bmm(
                        Gamma_list[i].repeat(K,1,1), 
                        Sigma_v
                    )
                )
            )
        ) for i in range(m)
    ]) / m

    return new_v_list, new_sigma_sq, new_Sigma_v



def multivariate_njll_i(y_i, Gamma_i, beta, v_i, sigma_sq, Sigma_v) : 
    '''
    Compute 2 x negative joint log-likelihood for i-th subject

    NAME            SIZE        TYPE                INFO    
    y_i             [n_i x K]   torch tensor        response
    Gamma_i         [n_i x p]   torch tensor        design matrix for i-th subject
    beta            [p x K]     torch tensor        fixed effects
    v_i             [p x K]     torch tensor        estimated random slopes for i-th subject
    sigma_sq        [K]         torch tensor        estimated variance of e
    Sigma_v         [K x p x p] torch tensor        estimated variance matrix of random slope v
    '''
    n_i = len(y_i)

    term_1 = torch.sum(torch.pow(y_i - Gamma_i @ (beta + v_i), 2) / sigma_sq, dim=0)
    term_2 = n_i * torch.log(2 * torch.pi * sigma_sq)
    term_3 = torch.sum(v_i.T * torch.linalg.solve(Sigma_v, v_i.T), dim=1)
    term_4 = torch.linalg.slogdet(2 * torch.pi * Sigma_v)[1]

    return torch.sum(term_1 + term_2 + term_3 + term_4)
