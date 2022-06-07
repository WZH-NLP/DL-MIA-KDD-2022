import numpy as np
import pandas as pd
import random

import torch

numOfRecommend = 100


def set_seed(seed=2021):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def recommend(filename, filepath):
    rec = open(filepath, 'w')
    data = pd.read_csv(filename, sep=',', names=['SessionID', 'ItemID', 'Rating', 'Time'])
    Popularity = data.groupby('ItemID').size()
    sorted_Popularity = Popularity.sort_values(ascending=False)
    popular_item = sorted_Popularity.index.tolist()
    user_data = data.groupby('SessionID')
    for user, value in user_data:
        interacted_item = value['ItemID'].tolist()
        index = 0
        for j in range(numOfRecommend):
            while popular_item[index] in interacted_item:
                index = index + 1
            rec.write(str(user) + '\t' + str(popular_item[index]) + '\t' + '0' + '\n')
            index = index + 1


def recommend_defense(filename, filepath):
    rec = open(filepath, 'w')
    data = pd.read_csv(filename, sep=',', names=['SessionID', 'ItemID', 'Rating', 'Time'])
    Popularity = data.groupby('ItemID').size()
    sorted_Popularity = Popularity.sort_values(ascending=False)
    popular_item = sorted_Popularity.index.tolist()
    user_data = data.groupby('SessionID')
    candidate_items = set(popular_item[:800])
    for user, value in user_data:
        interacted_items = set(value['ItemID'].tolist())
        recommend_candidate_items = list(candidate_items - interacted_items)
        np.random.shuffle(recommend_candidate_items)
        recommend_items = recommend_candidate_items[:numOfRecommend]
        for j in range(numOfRecommend):
            rec.write(str(user) + '\t' + str(recommend_items[j]) + '\t' + '0' + '\n')


if __name__ == '__main__':
    # set_seed(2022)
    # nonmem_train = "ml-1m_Snonmem_train"
    # nonmem_rec = "ml-1m_Snonmem_recommendation"
    # recommend(nonmem_train, nonmem_rec)
    #
    # nonmem_train = "ml-1m_Tnonmem_train"
    # nonmem_rec = "ml-1m_Tnonmem_recommendation"
    # recommend(nonmem_train, nonmem_rec)

    # nonmem_train = "beauty_Snonmem_train"
    # nonmem_rec = "beauty_Snonmem_recommendation"
    # recommend(nonmem_train, nonmem_rec)
    #
    # nonmem_train = "beauty_Tnonmem_train"
    # nonmem_rec = "beauty_Tnonmem_recommendation"
    # recommend(nonmem_train, nonmem_rec)

    nonmem_train = "ml-1m_Snonmem_train"
    nonmem_rec = "ml-1m_Snonmem_recommendation_defense"
    recommend_defense(nonmem_train, nonmem_rec)

    nonmem_train = "ml-1m_Tnonmem_train"
    nonmem_rec = "ml-1m_Tnonmem_recommendation_defense"
    recommend_defense(nonmem_train, nonmem_rec)

    nonmem_train = "beauty_Snonmem_train"
    nonmem_rec = "beauty_Snonmem_recommendation_defense"
    recommend_defense(nonmem_train, nonmem_rec)

    nonmem_train = "beauty_Tnonmem_train"
    nonmem_rec = "beauty_Tnonmem_recommendation_defense"
    recommend_defense(nonmem_train, nonmem_rec)
