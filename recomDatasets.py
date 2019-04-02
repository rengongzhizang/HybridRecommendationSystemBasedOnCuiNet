import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class AutoEncoderDataset(Dataset):
    def __init__(self, x_train):
        self.x = x_train
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
         temp = torch.from_numpy(self.x[idx]).type(torch.FloatTensor).view(-1)
         sample = (temp, temp)
         return sample

class GenresDataset(Dataset):
    def __init__(self, x_train):
        self.x = x_train
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        temp = self.x[idx,:].view(-1)
        sample = (temp, temp)
        return sample

class OverviewDataset(Dataset):
    def __init__(self, x_train):
        self.x = x_train
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        temp = self.x[idx,:].view(-1)
        sample = (temp, temp)
        return sample

class RecsysDataset(Dataset):
    def __init__(self, user_feature, item_feature, labels):
        self.user_feature = user_feature                        # user_feature = user_onehot
        self.item_feature = item_feature                        # item_feature = (item_onehot, item_overview)
        self.labels = labels.float().view(-1)

        self.count = len(labels)

    def __getitem__(self, idx):
        features = (self.user_feature[idx,:], self.item_feature[0][idx,:], self.item_feature[1][idx,:]) # item_feature = (item_onehot, item_overview)
        return (features, self.labels[idx])

    def __len__(self):
        return self.count
