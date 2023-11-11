import time
import copy
import torch
import random
import torchvision
import numpy as np

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm.auto import tqdm

from util import make_reproducibility, TensorDataset, convert_to_xyz, mae

def gazenet(
    train_ids, train_images, train_hps, train_gazes, 
    test_ids, test_images, test_hps, test_gazes, 
    network, hidden_features=4096, K=2, 
    init_lr=1e-3, weight_decay=0, batch_size=256, max_iter=15000, betas=(0.9, 0.95), 
    device=torch.device('cpu'), experiment_name=1, SEED=None, snapshot = 300, verbose=False, large_test=False) : 
    
    torch.cuda.empty_cache()
    if SEED is not None : 
        make_reproducibility(SEED)
    
    train_ids_unique = np.unique(train_ids)
    # train_gazes_cuda = train_gazes.to(device)
    # train_cluster = [np.where(train_ids == idx)[0] for idx in train_ids_unique]

    train_dataset = TensorDataset(train_images, train_hps, train_gazes)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_images = test_images / 255.0
    test_y_cuda = test_gazes.to(device)
    test_ids_unique = np.unique(test_ids)
    test_cluster = [np.where(test_ids == idx)[0] for idx in test_ids_unique]
    # test_N = len(test_gazes)


    model=network(hidden_features=hidden_features, out_features=K).to(device)
    optimizer = optim.Adam(model.parameters(), lr=init_lr, betas=betas)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.1)

    # Ready for save
    train_loss_list = []
    # prediction = np.zeros((test_N, 3))
    train_eval_list = np.zeros((2,500))
    test_eval_list = np.zeros((2,500))

    num_iter = 0
    exceed_max_iter = False
    print('Main train starts')
    train_start = time.time()
    for epoch in tqdm(range(5000)) : 
        if exceed_max_iter : 
            break

        model.train()
        for _, (image, hp, y) in enumerate(train_loader) :
            image = image.to(device)
            hp = hp.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            train_loss = F.mse_loss(y, model(image, hp))
            train_loss.backward()
            train_loss_list.append(train_loss.item())
            optimizer.step()
            scheduler.step()
            num_iter += 1
            if num_iter >= max_iter : 
                exceed_max_iter=True
                break

        if epoch % snapshot == 0 : 
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Current learning rate : {current_lr}')
            print(f'Last batch mse loss : {train_loss.item()}')

            # Evaluation
            model.eval()
            with torch.no_grad() : 
                # train_y_hat = torch.zeros_like(train_y_cuda)
                # for cluster in train_cluster : 
                #     train_y_hat[cluster] = model(train_images[cluster].to(device), train_hps[cluster].to(device))

                # train_mse = F.mse_loss(train_y_cuda, train_y_hat).item()
                # train_mae = mae(train_y_cuda, train_y_hat, is_3d=False, deg=deg).item()
                
                # train_eval_list[0,epoch // snapshot] = train_mae
                # train_eval_list[1,epoch // snapshot] = train_mse

                # print(f'{epoch}-th epoch train MAE, MSE  : {train_mae:4f} deg, {train_mse:4f}')

                if large_test : 
                    test_y_hat = torch.zeros_like(test_y_cuda)
                    for cluster in test_cluster : 
                        test_y_hat[cluster] = model(test_images[cluster].to(device), test_hps[cluster].to(device))
                else : 
                    test_y_hat = model(test_images.to(device), test_hps.to(device))

                
                test_mse = F.mse_loss(test_y_cuda, test_y_hat).item()
                test_mae = mae(test_y_cuda, test_y_hat, is_3d=False, deg=False).item()
                test_eval_list[0,epoch // snapshot] = test_mse
                test_eval_list[1,epoch // snapshot] = test_mae

                print(f'{epoch}-th iteration test MAE, MSE : {test_mae:4f} deg, {test_mse:4f}')
    prediction = convert_to_xyz(test_y_hat, deg=False).cpu().numpy()
        

    train_end = time.time()
    print(f'Main train spends {(train_end - train_start):.4f} sec')

    np.save(f'./Prediction/{model.model_name}_{experiment_name}_pred', prediction)
    np.save(f'./Prediction/{model.model_name}_{experiment_name}_train_loss', train_loss_list)
    np.save(f'./Prediction/{model.model_name}_{experiment_name}_test_loss', test_eval_list)

    torch.save(model.state_dict(), f'./Model/{model.model_name}_{experiment_name}_trained_model.pt')

    return train_loss_list
