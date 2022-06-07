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
    data = pd.read_csv(filename, sep='\t', names=['UserID', 'ItemID', 'Rating', 'Time'])
    Popularity = data.groupby('ItemID').size()
    sorted_Popularity = Popularity.sort_values(ascending=False)
    popular_item = sorted_Popularity.index.tolist()
    user_data = data.groupby('UserID')
    for user, value in user_data:
        interacted_item = value['ItemID'].tolist()
        index = 0
        for j in range(numOfRecommend):
            while popular_item[index] in interacted_item:
                index = index + 1
            rec.write(str(user) + '\t' + str(popular_item[index]) + '\t' + '0' + '\n')
            index = index + 1


if __name__ == '__main__':
    # nonmem_train = "datasets/ml-1m_Snonmem_train"
    # nonmem_rec = "datasets/ml-1m_Snonmem_recommendation"
    # recommend(nonmem_train, nonmem_rec)
    #
    # Tnonmem_train = "datasets/ml-1m_Tnonmem_train"
    # Tnonmem_rec = "datasets/ml-1m_Tnonmem_recommendation"
    # recommend(Tnonmem_train, Tnonmem_rec)

    nonmem_train = "datasets/amazon_Snonmem_train"
    nonmem_rec = "datasets/amazon_Snonmem_recommendation"
    recommend(nonmem_train, nonmem_rec)

    Tnonmem_train = "datasets/amazon_Tnonmem_train"
    Tnonmem_rec = "datasets/amazon_Tnonmem_recommendation"
    recommend(Tnonmem_train, Tnonmem_rec)
