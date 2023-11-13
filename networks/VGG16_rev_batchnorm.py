import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

class VGG16(nn.Module):
    def __init__(self, hidden_features=4096, out_features=2, dtype=torch.float):
        super(VGG16, self).__init__()

        self.hidden_features=hidden_features
        self.out_features=out_features
        self.p = hidden_features + 1
        self.model_name = "VGG-16"
        
        self.convNet = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),  # input channel is one-dimensional
            nn.ReLU(inplace=True), 
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=2, stride=1),  # stride 2->1

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=2, stride=1), # stride 2-> 1

            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 
            
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), 
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 

            nn.ReLU(inplace=True), 
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), 
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 

            nn.ReLU(inplace=True), 
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 
            nn.ReLU(inplace=True), 

            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )

        # MLP
        self.FC = nn.Sequential(
            nn.Linear(512*4*7, hidden_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_features+2, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5)
        )

        self.fc2 = nn.Linear(hidden_features + 1, out_features, bias=False)


    def get_feature_map(self, image, head_pose) : 
        feature = self.convNet(image)
        feature = torch.flatten(feature, start_dim=1)
        feature = self.FC(feature)
        feature = self.fc1(torch.cat([feature, head_pose], dim=1))
        return torch.cat([torch.ones_like(feature[:,0]).unsqueeze(1), feature], dim=1)
      

    def forward(self, image, head_pose):
        return self.fc2(self.get_feature_map(image, head_pose))
