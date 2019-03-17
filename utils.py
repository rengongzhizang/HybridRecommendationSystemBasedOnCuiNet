import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from recomDatasets import *
from autoencoders import *

def data_loader(path):
    df = pd.read_csv(path)
    headers = list(df)
    return headers, df

def tfidf_generator(corpus):  # return a scipy sparse tf-idf embedding, 
    tagsVecterizer = TfidfVectorizer(ngram_range=(1,2), min_df=1e-3, stop_words='english')                                              # preprocessing for autoencoder
    tagsVec = tagsVecterizer.fit_transform(corpus)
    return tagsVec.todense()

def train_eval(device, net, loss_fn, optimizer, train_loader, val_loader, epoch_num=15):
    print('Start Training')
    best_params = 0.0
    for epoch in range(epoch_num):
        print('Epoch Num: {} / {} \n -------------------------'.format((epoch + 1), epoch_num))
        net.train()
        running_loss = 0.0
        running_acc = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            scores = net(inputs)
            loss = loss_fn(scores, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_loader)
        print('Training loss: {:.4f}'.format(epoch_loss))

        net.eval()
        val_running_loss = 0.0
        best_loss = 100.0
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            scores = net(inputs)
            loss = loss_fn(scores, labels.float())
            optimizer.zero_grad()
            val_running_loss += loss.item() * inputs.size(0)

        val_loss = val_running_loss / len(val_loader)
        print('Validation loss: {:.4f}'.format(val_loss))
        if val_loss < best_loss:
            best_loss = val_loss
            best_params = net.state_dict()
    net.load_state_dict(best_params)
    return net

def generator(device, net, loader):     # this net here is the encoder net
    embeddings = []
    for inputs, _ in loader:
        inputs = inputs.to(device)
        scores = net(inputs)            # batch_size * 50 
        embeddings.append(scores)
    embeddings = torch.cat(embeddings, 0)
    return embeddings

def tag_encoder(device, tagsVec): # users' tag autoencoder dim = 50, hidden_dim = 200, a sparse matrix
    tagNums, dim = tagsVec.shape
    trainVec, valVec = tagsVec[:tagNums - int(tagNums/3),:], tagsVec[tagNums - int(tagNums/3):,:]

    train_data = AutoEncoderDataset(trainVec)
    val_data = AutoEncoderDataset(valVec)
    data = AutoEncoderDataset(tagsVec)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=100, shuffle=True)
    loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=False)

    net = nn.Sequential(TagEncoder(dim), TagDecoder(dim))
    loss_fn = nn.MSELoss()

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    trained_net = train_eval(device, net, loss_fn, optimizer, train_loader, val_loader)
    embeds = generator(device, trained_net[0], loader)
    return embeds
