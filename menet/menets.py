
import os
import copy
import time
import torch
import random
import numpy as np

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm.auto import tqdm
from sklearn.linear_model import LinearRegression


from util import make_reproducibility, TensorDataset, convert_to_xyz, mae
from menet.variational_em import EM_update, multivariate_njll_i

def MeNets(
        train_ids, train_images, train_hps, train_gazes, 
        test_ids, test_images, test_hps, test_gazes, 
        network, hidden_features=500, K=2, 
        MAXITER=320000, snapshot=300, batch_size=1000, 
        base_lr=0.1, weight_decay=0.0005, momentum=0.9, power=1.0, max_iter=2, patience=1, 
        test_unseen=False, large_test=False, 
        device=torch.device('cpu'), SEED=None, experiment_name = 1): 
    '''
    Python implementation of MeNets. 
    Hyperparameters are selected according to the original MeNets code (implemented via MATLAB and Caffe). 

    Network architecture : ResNet-18

    batch_size = 1000
    MAXITER : 320000

    Caffe's default optimizer : SGD
    base_lr = 0.1
    weight_decay = 0.0005
    momentum = 0.9
    power = 1.0
    lr scheduler type : 'poly', that is,  lr_i = base_lr * (1-i/MAXITER)*power
    '''

    torch.cuda.empty_cache()
    if SEED is not None : 
        make_reproducibility(SEED)

    train_gazes_cuda = train_gazes.to(device)
    test_gazes_cuda = test_gazes.to(device)
    
    train_N = len(train_gazes)
    train_ids_unique = np.unique(train_ids)
    train_cluster = [np.where(train_ids == id)[0] for id in np.unique(train_ids_unique)]
    train_n_list = [len(cluster) for cluster in train_cluster]
    train_m = len(train_cluster)
    

    test_N = len(test_gazes)
    if test_unseen : 
        test_ids_unique = np.unique(test_ids)
    else : 
        test_ids_unique = train_ids_unique
    test_cluster = [np.where(test_ids == idx)[0] for idx in test_ids_unique]
    test_n_list = [len(cluster) for cluster in test_cluster]
    test_m = len(test_cluster)


    v_list = [torch.zeros(503, 2, device=device) for _ in range(train_m)]
    sigma_sq = torch.ones(2, device=device)
    Sigma_v = torch.eye(503, device=device).repeat(2,1,1)


    prediction_fixed = np.zeros((max_iter, test_N, 3))
    prediction = np.zeros((max_iter, test_N, 3))
    prediction_adjusted = np.zeros((max_iter, test_N, 3))

    
    train_nn_loss_list = []
    train_loss_list = np.zeros((2, max_iter))
    test_loss_list  = np.zeros((3, max_iter))

    v_list_list     = np.zeros((max_iter, train_m, 503, 2))
    beta_list       = np.zeros((max_iter, 503, 2))
    w_list          = np.zeros((max_iter, 503, 2))

    # Early stopping crietria
    best_njll = 1e8
    best_model = None
    best_v_list = copy.deepcopy(v_list)
    best_sigma_sq = copy.deepcopy(sigma_sq)
    best_Sigma_v = copy.deepcopy(Sigma_v)

    update_count = 0
    best_index=0
    update_stop = False

    # Main part
    print(f"EM algorithm starts")
    # Initialize fixed parts of responses
    train_y_fixed = train_gazes
    for iter in range(max_iter) : 
        if update_stop : 
            break

        model = network(hidden_features=hidden_features, out_features=K).to(device)
        opt = optim.SGD(model.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer=opt, lr_lambda = lambda i: (1-i/MAXITER) ** power)

        train_dataset = TensorDataset(train_images, train_hps, train_y_fixed)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Training : SGD step
        num_iter = 0
        exceed_MAXITER = False
        for epoch in tqdm(range(10000)) : 
            if exceed_MAXITER : 
                break

            model.train()
            for _, (image, hp, y) in enumerate(train_loader) :
                image = image.to(device)
                hp = hp.to(device)
                y = y.to(device)

                opt.zero_grad()
                train_loss = F.mse_loss(y, model(image, hp))
                train_loss.backward()
                train_nn_loss_list.append(train_loss.item())
                opt.step()
                scheduler.step()

                num_iter += 1
                if num_iter >= MAXITER : 
                    exceed_MAXITER=True
                    break

            if epoch % snapshot == 0 : 
                current_lr = opt.param_groups[0]['lr']
                print(f'Current learning rate : {current_lr}')
                print(f'Last batch mse loss : {train_loss.item()}')

        # Training : M-step
        model.eval()

        with torch.no_grad() : 

            train_y_list = [train_gazes[cluster].to(device) for cluster in train_cluster]
            train_Gamma_list = [
                model.get_feature_map(train_images[cluster].to(device), train_hps[cluster].to(device)).detach()
                for cluster in train_cluster
            ]
            beta = model.fc2.weight.data.T.detach()

            v_list, sigma_sq, Sigma_v = EM_update(train_y_list, train_Gamma_list, beta, sigma_sq, Sigma_v, train_n_list, False, device)

            train_njll = sum([multivariate_njll_i(train_y_list[i], train_Gamma_list[i], beta, v_list[i], sigma_sq, Sigma_v) for i in range(train_m)]).item() / train_N
            print(f'{iter}-th iter train negative (log) joint likelihood : {train_njll}')
            
        # Early stopping
        if train_njll < best_njll : 
            best_njll = train_njll
            best_model = copy.deepcopy(model)
            best_v_list = copy.deepcopy(v_list)
            best_sigma_sq = copy.deepcopy(sigma_sq)
            best_Sigma_v = copy.deepcopy(Sigma_v)
            best_index = 0
            update_count = 0
        
        else : 
            update_count += 1

        if update_count == patience : 
            update_stop = True
            print(f'Variational EM algorihtm stopped at {iter - patience}-th iteration')

        

        # Evaluation step
        with torch.no_grad() : 
            train_Gamma = torch.zeros(train_N, 503, device=device)
            
            train_fixed = torch.zeros(train_N, 2, device=device)
            train_random = torch.zeros(train_N, 2, device=device)
            for i in range(train_m) : 
                train_Gamma[train_cluster[i]]  = model.get_feature_map(train_images[train_cluster[i]].to(device), train_hps[train_cluster[i]].to(device)).detach()
                train_fixed[train_cluster[i]]  = model.fc2(train_Gamma[train_cluster[i]])
                train_random[train_cluster[i]] = train_Gamma[train_cluster[i]] @ v_list[i]

            # Evaluation 1 : random effect adjustment
            w = LinearRegression(fit_intercept=False)
            w.fit(X=train_Gamma.detach().cpu(), y=train_random.detach().cpu())
            w_beta = torch.as_tensor(w.coef_).T.to(device)
            
            # Evaluation 2 : train mae
            train_mae = mae(train_fixed + train_random, train_gazes_cuda, is_3d=False, deg=False).item()
            print(f'{iter}-th iteration train MAE : {train_mae}')


            # Evaluation 3 : test 
            test_Gamma = torch.zeros(test_N, 503, device=device)
            test_fixed = torch.zeros(test_N, 2, device=device)
            test_random = torch.zeros_like(test_fixed)
            for cluster in test_cluster : 
                test_Gamma[cluster] = model.get_feature_map(test_images[cluster].to(device), test_hps[cluster].to(device)).detach()
                test_fixed[cluster] = model.fc2(test_Gamma[cluster])

            if test_unseen is False : 
                for i in range(train_m) : 
                    test_random[test_cluster[i]] = test_Gamma[test_cluster[i]] @ v_list[i]

            test_adjusted = test_Gamma @ w_beta

            test_mae_fixed = mae(test_fixed, test_gazes_cuda, is_3d=False, deg=False).item()
            test_mae  = mae(test_fixed + test_random, test_gazes_cuda, is_3d=False, deg=False).item()
            test_mae_adjusted  = mae(test_fixed + test_adjusted, test_gazes_cuda, is_3d=False, deg=False).item()

            print(f'{iter}-iter train MAE, NJLL : {train_loss_list[1,best_index]:.4f} deg, {train_loss_list[0,best_index]:.4f}')
            print(f'{iter}-iter test MAE (fixed) : {test_loss_list[0,best_index]:.4f} deg')
            print(f'{iter}-iter test MAE : {test_loss_list[1,best_index]:.4f} deg')
            print(f'{iter}-iter test MAE (adjusted) : {test_loss_list[2,best_index]:.4f} deg')

        # Save
        with torch.no_grad() : 
            prediction_fixed[iter] = convert_to_xyz(test_fixed, deg=False).detach().cpu().numpy()
            prediction[iter] = convert_to_xyz(test_fixed + test_random, deg=False).detach().cpu().numpy()
            prediction_adjusted[iter] = convert_to_xyz(test_fixed + test_adjusted, deg=False).detach().cpu().numpy()

            train_loss_list[0, iter] = train_njll
            train_loss_list[1, iter] = train_mae

            test_loss_list[0, iter] = test_mae_fixed
            test_loss_list[1, iter] = test_mae
            test_loss_list[2, iter] = test_mae_adjusted

            v_list_list[iter]   = torch.cat([v_i.unsqueeze(0).detach() for v_i in v_list]).cpu().numpy()
            beta_list[iter]     = model.fc2.weight.T.detach().cpu().numpy()
            w_list[iter]        = w_beta.cpu().numpy()

    # Best model results
    
    print(f'Selected model train MAE, NJLL : {train_loss_list[1,best_index]:.4f} deg, {train_loss_list[0,best_index]:.4f}')
    print(f'Selected model test MAE (fixed) : {test_loss_list[0,best_index]:.4f} deg')
    print(f'Selected model test MAE : {test_loss_list[1,best_index]:.4f} deg')
    print(f'Selected model test MAE (adjusted) : {test_loss_list[2,best_index]:.4f} deg')

    
    os.makedirs('./menet', exist_ok=True)
    torch.save(model.state_dict(), f'./menet/{experiment_name}_trained_model.pt')
    torch.save(best_model.state_dict(), f'./menet/{experiment_name}_selected_model.pt')

    np.save(f'./menet/{experiment_name}_pred_fixed', prediction_fixed)
    np.save(f'./menet/{experiment_name}_pred', prediction)
    np.save(f'./menet/{experiment_name}_pred_adjusted', prediction_adjusted)
    np.save(f'./menet/{experiment_name}_train_loss', train_loss_list)
    np.save(f'./menet/{experiment_name}_test_loss', test_loss_list)
    np.save(f'./menet/{experiment_name}_v_list', v_list_list)
    np.save(f'./menet/{experiment_name}_beta', beta_list)
    np.save(f'./menet/{experiment_name}_w', w_list)
    
    return model, train_loss_list





