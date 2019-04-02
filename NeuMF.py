import os
import re
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class GMF(nn.Module):
    def __init__(self):
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        user_feature = x[0]
        item_feature = x[1]
        return self.sigmoid((user_feature * item_feature).sum(dim=1))

class MLP(nn.Module):
    def __init__(self, user_feature_num, item_feature_num, hidden_1=128, hidden_2=32, hidden_3=8, hidden_4=1):
        self.feature_num = user_feature_num + item_feature_num
        self.linear_1 = nn.Linear(self.feature_num, hidden_1)
        self.linear_2 = nn.Linear(hidden_1, hidden_2)
        self.linear_3 = nn.Linear(hidden_2, hidden_3)
        self.linear_4 = nn.Linear(hidden_3, hidden_4)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        features = torch.cat((x[0], x[1]),dim=1)
        layer_1 = self.relu(self.linear_1(features))
        layer_2 = self.relu(self.linear_2(layer_1))
        layer_3 = self.relu(self.linear_3(layer_2))
        output = self.sigmoid(self.linear_4(layer_3))
        return output

class NeuMF(nn.Module):
    def __init__(self, stack_dim=2, output_dim=1):
        self.linear = nn.Linear(stack_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        cat = torch.cat((x,y), dim=1)
        return self.sigmoid(self.linear(cat))