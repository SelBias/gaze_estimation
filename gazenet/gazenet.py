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
    device=torch.device('cpu'), experiment_name=1, SEED=None, verbose=False, large_test=False) : 
    
    torch.cuda.empty_cache()
    if SEED is not None : 
        make_reproducibility(SEED)
    
    train_dataset = TensorDataset(train_images, train_hps, train_gazes)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_y_cuda = test_gazes.to(device)
    test_ids_unique = np.unique(test_ids)
    test_cluster = [np.where(test_ids == idx)[0] for idx in test_ids_unique]


    model=network(hidden_features=hidden_features, out_features=K).to(device)
    optimizer = optim.Adam(model.parameters(), lr=init_lr, weight_decay=weight_decay, betas=betas)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.1)

    train_loss_list = []

    num_iter = 0
    exceed_max_iter = False
    print('Main train starts')
    train_start = time.time()
    for _ in tqdm(range(5000)) : 
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
            if verbose : 
                print(f'Batch MSE loss : {train_loss.item():.4f}')
            optimizer.step()
            scheduler.step()
            num_iter += 1
            if num_iter >= max_iter : 
                exceed_max_iter=True
                break

    # Evaluation (only once)
    model.eval()
    with torch.no_grad() : 

        if large_test : 
            test_y_hat = torch.zeros_like(test_y_cuda)
            for cluster in test_cluster : 
                test_y_hat[cluster] = model(test_images[cluster].to(device), test_hps[cluster].to(device)).detach()
        else : 
            test_y_hat = model(test_images.to(device), test_hps.to(device)).detach()

        
        test_mse = F.mse_loss(test_y_cuda, test_y_hat).item()
        test_mae = mae(test_y_cuda, test_y_hat, is_3d=False, deg=False).item()

        print(f'GazeNet+ test MAE, MSE : {test_mae:4f} deg, {test_mse:4f}')
    prediction = convert_to_xyz(test_y_hat, deg=False).cpu().numpy()
    
    train_end = time.time()
    print(f'Main train spends {(train_end - train_start):.4f} sec')

    np.save(f'./prediction/GazeNet_{experiment_name}_pred', prediction)
    np.save(f'./results/GazeNet_{experiment_name}_train_loss', np.array(train_loss_list))

    torch.save(model.state_dict(), f'./results/GazeNet_{experiment_name}_trained_model.pt')
    return train_loss_list
