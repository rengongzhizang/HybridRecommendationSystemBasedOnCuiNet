import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from autoencoders import *
from recomDatasets import *
from utils import *

## two kinds of embeddings: for GMF, for MLP
## user_features: user_tag, jobs, ages, genders, zipcode
## movie_features: genres, descriptions, (ratings?)
## user_id
## movie_id

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = "data/meta_data.csv"
    df = pd.read_csv(path)
    corpus = df['user_tag']
    tagsVec = tfidf_generator(corpus)
    embeds = tag_encoder(device, tagsVec)
    print(embeds.shape)

if __name__ == "__main__":
    main()