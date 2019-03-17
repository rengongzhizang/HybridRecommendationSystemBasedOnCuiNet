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
         