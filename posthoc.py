

import os
import time
import copy
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm.auto import tqdm
from util import make_reproducibility, TensorDataset, convert_to_xyz, mae, convert_to_spherical

class fitrlinear(nn.Module) :
    def __init__(self, X, y, device) :
        super(fitrlinear, self).__init__()

        self.X = X # X does not include the intercept term
        self.y = y
        self.DEVICE=device

        self.fc = nn.Linear(in_features=X.shape[1], out_features=y.shape[1], bias=True)

        self.fc.weight.data = torch.zeros_like(self.fc.weight.data)
        self.fc.bias.data = torch.mean(y, dim=0)

        self.Epsilon = (torch.quantile(y, 0.75, dim=0) - torch.quantile(y, 0.25, dim=0)) / 13.49
        self.Lambda = 1 / len(y)
        self.Learner = 'leastsquares'
        self.Regularization = 'ridge'

        self.BatchSize = 10
        self.LearnRate  = 1 / torch.sqrt(torch.max(torch.sum(X ** 2, dim=0))).item()
        self.OptimizeLearnRate = True
        self.BatchLimit = 100000
        # self.BetaTolerance = 1e-4

        self.opt = optim.SGD(self.fc.parameters(), lr=self.LearnRate, weight_decay=self.Lambda)
        self.dataset = TensorDataset(X,y)
        self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.BatchSize, shuffle=True)
        self.train_loss_list = []

    def fit(self, max_epoch = 1000, SEED = None) :
        if SEED is not None :
            make_reproducibility(SEED)

        update_stop=False
        exceed_max_iter = False
        old_train_loss = 1e8
        num_iter = 0

        for _ in tqdm(range(max_epoch)) :
            if update_stop or exceed_max_iter :
                break

            for _, (X, y) in enumerate(self.loader) :
                # beta_old = self.fc.weight.data.detach()
                self.opt.zero_grad()
                train_loss = F.mse_loss(y.to(self.DEVICE), self.fc(X.to(self.DEVICE))) * 0.5
                train_loss.backward()
                self.train_loss_list.append(train_loss.item())
                self.opt.step()

                if old_train_loss < train_loss.item() :
                    self.opt.param_groups[0]['lr'] *= 0.5
                old_train_loss = train_loss.item()

                num_iter += 1
                if num_iter >= self.BatchLimit :
                    exceed_max_iter=True
                    break

    def predict(self, X_test) :
        return self.fc(X_test.to(self.DEVICE))