import os
import re
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from autoencoders import *
from recomDatasets import *
from utils import *
from preprocessData import *

## two kinds of embeddings: for GMF, for MLP
## user_features: user_tag, jobs, ages, genders, zipcode
## movie_features: genres, descriptions, (ratings?)
## user_id
## movie_id

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = 'data/'
    ratings, rating_num = ratings_loader(path + 'ratings.dat')
    movie_data, movie_plot = item_features_loader(path)
    user_data, user_dict_size = user_features_loader(path)
    #plot_corpus = list(movie_plot.values())
    plot_word_to_ix, plot_max_length, plot_lines = make_words_dict(movie_plot)
    genre_word_to_ix, genre_max_length, genre_lines = make_genre_dict(movie_data)
    #plot_vec = make_item_vec(plot_lines[ratings[1][1]], plot_word_to_ix, plot_max_length)
    #genre_vec = make_item_vec(genre_lines[ratings[1][1]],genre_word_to_ix, genre_max_length)
    plot_onehot = make_onehot(plot_lines, plot_word_to_ix, plot_max_length)
    genre_onehot = make_onehot(genre_lines, genre_word_to_ix, genre_max_length)
    train_data = training_testing_generator(path, ratings, user_data, genre_onehot, plot_onehot)
    pdb.set_trace()
    #dict_size = len(word_to_ix)
    #plot_vec = make_embed_vec(lines, word_to_ix, max_length)
    #plot_encoder(device, plot_vec, word_to_ix) 
if __name__ == "__main__":
    main()