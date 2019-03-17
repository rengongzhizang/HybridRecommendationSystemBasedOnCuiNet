import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from utils import *
from autoencoders import *

path = "data/meta_data.csv"
df = pd.read_csv(path)
#print(data_loader(path))
corpus = df['user_tag']
tagsVecterizer = TfidfVectorizer(ngram_range=(1,2), min_df=1e-3, stop_words='english')
tagsVec = tagsVecterizer.fit_transform(corpus)
#print(tagsVecterizer.get_feature_names())
print(tagsVec.todense().shape)
#print(type(tagsVec.todense()))
#print(torch.from_numpy(tagsVec.todense()[0]).type(torch.FloatTensor))
tagsVec = tfidf_generator(corpus)
print(len(tagsVec))
#tagNums, dim = tagsVec.shape
#print(dim)
#print(int(tagNums/3))
#trainVec, valVec = tagsVec[:tagNums - int(tagNums/3),:], tagsVec[tagNums - int(tagNums/3):,:]
#print(trainVec.shape, valVec.shape)
#net = nn.Sequential(TagEncoder(1000),TagDecoder(1000))
#print(net[0])
