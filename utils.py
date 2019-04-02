import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from recomDatasets import *
from autoencoders import *
import numpy as np

def data_loader(path):
    df = pd.read_csv(path)
    headers = list(df)
    return headers, df

def tfidf_generator(corpus):  # return a scipy sparse tf-idf embedding, 
    tagsVecterizer = TfidfVectorizer(ngram_range=(1,2), min_df=1e-3, stop_words='english')                                              # preprocessing for autoencoder
    tagsVec = tagsVecterizer.fit_transform(corpus)
    return tagsVec.todense()
'''
    This function load the word_dict of plots' overview
'''
def make_words_dict(corpus): # corpus is a dictionary of strings
    word_to_ix = dict()
    max_length = 0
    lines = {}
    for i, line in corpus.items():
        line = line.lower()
        line = re.sub(r'[^\w\s]','',line)
        words = line.strip().split(" ")
        max_length = max(max_length, len(words))
        lines[i] = words
        for word in words:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix) + 1
    return word_to_ix, max_length, lines

'''
    This function load the word_dict of genres
'''
def make_genre_dict(corpus):
    word_to_ix = dict()
    max_length = 0
    lines = dict()
    for i, line in corpus.items():
        #line = line[1]
        #line = line.lower()
        lines[i] = line[1]
        max_length = max(len(line[1]), max_length)
        for word in line[1]:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix) + 1
    return word_to_ix, max_length, lines

def make_embed_vec(lines, word_to_ix, max_length): # lines is a list of lists of strings
    vec = torch.zeros((len(lines), max_length))
    for i,line in enumerate(lines):
        for j, word in enumerate(line):
            vec[i][j] = word_to_ix[word]
    return vec

def make_item_vec(ls, word_to_ix, max_length):
    vec = torch.zeros((1,max_length))
    for i, word in enumerate(ls):
        vec[0][i] = word_to_ix[word]
    return vec

def make_onehot(lines, word_to_ix, max_length):
    onehot = dict()
    for ids, words in lines.items():
        onehot_emb = [0] * max_length
        for i, word in enumerate(words):
            onehot_emb[i] = word_to_ix[word]
        onehot[ids] = onehot_emb
    return onehot

def train_eval(device, net, loss_fn, optimizer, train_loader, val_loader, epoch_num=5):
    print('Start Training')
    best_params = 0.0
    training_loss_list = []
    val_loss_list = []
    for epoch in range(epoch_num):
        print('Epoch Num: {} / {} \n -------------------------'.format((epoch + 1), epoch_num))
        net.train()
        running_loss = 0.0
        running_acc = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            _, hidden = net[0](inputs)
            scores = net[1](inputs, hidden)
            loss = loss_fn(scores, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
        epoch_loss = running_loss / len(train_loader)
        training_loss_list.append(epoch_loss)
        print('Training loss: {:.4f}'.format(epoch_loss))

        net.eval()
        val_running_loss = 0.0
        best_loss = 10000.0
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            _, hidden = net[0](inputs)
            scores = net[1](inputs, hidden)
            loss = loss_fn(scores, labels)
            optimizer.zero_grad()
            val_running_loss += loss.item()

        val_loss = val_running_loss / len(val_loader)
        val_loss_list.append(val_loss)
        print('Validation loss: {:.4f}'.format(val_loss))
        if val_loss < best_loss:
            best_loss = val_loss
            best_params = net.state_dict()
    net.load_state_dict(best_params)
    return net, training_loss_list, val_loss_list

def generator(device, net, loader):     # this net here is the encoder net
    embeddings = []
    for inputs, _ in loader:
        inputs = inputs.to(device)
        scores = net(inputs)            # batch_size * 50 
        embeddings.append(scores)
    embeddings = torch.cat(embeddings, 0)
    return embeddings

def tag_encoder(device, tagsVec, em=30, es=50 ,dr=0.2): # users' tag autoencoder dim = 50, hidden_dim = 200, a sparse matrix
    tagNums, dim = tagsVec.shape
    trainVec, valVec = tagsVec[:tagNums - int(tagNums/4),:], tagsVec[tagNums - int(tagNums/4):,:]

    train_data = AutoEncoderDataset(trainVec)
    val_data = AutoEncoderDataset(valVec)
    data = AutoEncoderDataset(tagsVec)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=100, shuffle=True)
    loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=False)

    net = nn.Sequential(TagEncoder(dim, encoding_size=es, dropout_rate=dr), TagDecoder(dim, encoding_size=es, dropout_rate=dr))
    loss_fn = nn.NLLLoss()

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    trained_net, train_loss_list, val_loss_list = train_eval(device, net, loss_fn, optimizer, train_loader, val_loader, epoch_num=em)
    embeds = generator(device, trained_net[0], loader)
    return embeds, train_loss_list, val_loss_list

def plot_encoder(device, plot_vec, word_to_ix, en=30, es=200):
    plot_num, dim = plot_vec.shape
    trainVec, valVec = plot_vec[:plot_num - int(plot_num/4),:], plot_vec[plot_num - int(plot_num/4):,:]
    
    train_data = OverviewDataset(trainVec)
    val_data = OverviewDataset(valVec)
    data = OverviewDataset(plot_vec)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=100, shuffle=True)
    loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=False)

    net = nn.Sequential(RNNEncoder(len(word_to_ix)+1, es), RNNDecoder(len(word_to_ix)+1, es))
    loss_fn = nn.MSELoss()

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    trained_net, train_loss_list, val_loss_list = train_eval(device, net, loss_fn, optimizer, train_loader, val_loader, epoch_num=en)
    embeds = generator(device, trained_net[0], loader)
    return embeds, train_loss_list, val_loss_list
