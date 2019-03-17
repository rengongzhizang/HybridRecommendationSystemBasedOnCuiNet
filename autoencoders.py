import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class TagEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size=200, encoding_size=50, dropout_rate=0.2):
        super(TagEncoder, self).__init__()
        self.tagEncoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, encoding_size),
            nn.BatchNorm1d(encoding_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):      # prepocessed TF-IDF embeddings with dimension: 
        x = self.tagEncoder(x)
        return x

class TagDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size=200, encoding_size=50, dropout_rate=0.2):
        super(TagDecoder, self).__init__()
        self.tagDecoder = nn.Sequential(
            nn.Linear(encoding_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, vocab_size),
            nn.BatchNorm1d(vocab_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.tagDecoder(x)
        return x