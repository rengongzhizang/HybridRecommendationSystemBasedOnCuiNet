import os
import numpy as np
import pandas as pd
import random
import pdb

'''
    This function loads the ratings.dat
'''
def ratings_loader(path):
    all_ratings = dict()
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line is not None and line is not "":
                line = line.strip().split("::")
                line[0] = int(line[0])
                line[1] = int(line[1])
                #line[2] = int(line[2])
                if line[0] not in all_ratings:
                    all_ratings[line[0]] = [line[1]]
                else:
                    all_ratings[line[0]].append(line[1])
        print("You have loaded all rating data!")
    return all_ratings, len(all_ratings)
'''
    This function loads the movies.csv as two dictionaries: movie_data containing movie features such as genres, ids, (titles) and movie_plot containing 
    plot overviews of each movie
'''
def item_features_loader(path):
    movie_data = {}
    movie_plot = {}
    random_word = []
    with open(path + 'randomtext', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(' ')
            for word in line[:-1]:
                random_word.append(word)
    print("You have loaded all random words!")
    df = pd.read_csv(path + 'overview_modified.csv')
    headlist = ['0', '1', '2', 'overview']
    df = df[headlist]
    for i in range(df.shape[0]):
        words = []
        words += [df['1'][i][:df['1'][i].find('(')].strip().split(' ')]
        words += [df['2'][i].strip().split('|')]
        if df['0'][i] not in movie_data:
            movie_data[df['0'][i]] = words
        if df['0'][i] not in movie_plot:
            if df['overview'][i][:-1] != "None" and df['overview'][i][:-1] != "N/A":
                movie_plot[df['0'][i]] = df['overview'][i][:-1].strip()
            else:
                temp = ""
                randnums = random.sample(range(len(random_word)), 50)
                for randnum in randnums:
                    temp += random_word[int(randnum)] + " " 
                movie_plot[df['0'][i]] = temp.strip()
    print("You have loaded movie dictionaries!")
    return movie_data, movie_plot
'''
    This function loads the users.dat as a dictionary user_features whose key are userIDs, values are one-hot encodings of users'age, gender, job and 
    first 3 digits of zipcodes.
'''   
def user_features_loader(path):
    user_features = dict()
    user_num = 6040
    gender_num = 2
    age_num = 56
    job_num = 21
    max_num = 0
    with open(path + 'users.dat', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split("::")
            line[0] = int(line[0])
            line[1] = 1 + user_num if line[1] == "F" else 2 + user_num
            line[2] = int(line[2]) + gender_num + user_num
            line[3] = int(line[3]) + gender_num + user_num + age_num
            line[4] = int(line[4][:3]) + job_num + + gender_num + user_num + age_num
            max_num = max(line[4], max_num)
            user_features[line[0]] = line[1:]
    print("You have loaded user features!")
    return user_features, max_num

'''
    This function generates 4 negative samples for each training data and 100 negative samples for each testing data.
    Training data = [(user_data, (item_features, item_overview), label), *4, ...]
    Testing data = [(user_data, (item_features, item_overview), labels), *100, ...]
'''

def training_testing_generator(path, all_ratings, user_features, genre_onehot, plot_onehot, negative_num=(4,100)):
    train_data = []
    test_data = []
    for user, hist in all_ratings.items():
        for j,i in enumerate(hist[:-1]):
            posi = ([user]+user_features[user], (genre_onehot[i],plot_onehot[i]), 1)
            train_data.append(posi)
            
            for _ in range(negative_num[0]):
                rand_item = random.choice(list(set(genre_onehot.keys()) - set(hist)))
                nega = ([user]+user_features[user], (genre_onehot[rand_item], plot_onehot[rand_item]), 0)
                train_data.append(nega)
            
            #print("You have generated {}/{} training data for one movie".format(j, len(hist)))
    print("You have generated all training data for user {}".format(user))
    return train_data
