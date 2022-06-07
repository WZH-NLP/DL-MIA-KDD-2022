import json
import os
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.utils import shuffle
from ast import literal_eval
from torch.utils.data import Dataset, DataLoader


def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def preprocess(fpath1, fpath2, fpath3, fpath4):
    # construct dataset (pair)
    format = ['vector', 'label']
    data1 = pd.read_csv(fpath1, sep='\t', header=None, names=format, engine='python')
    data2 = pd.read_csv(fpath2, sep='\t', header=None, names=format, engine='python')
    set_seed(1234)
    print("len(data1):", len(data1['vector']))
    print("len(data2):", len(data2['vector']))
    state1 = np.random.get_state()
    data2 = shuffle(data2).reset_index()
    f3 = open(fpath3, 'w')
    embed_file = open(fpath4, 'w')
    embed_file_Dic = {}
    x = len(data1['vector'])
    y = len(data2['vector'])
    i1 = i2 = 0
    for i in range(max(x, y)):
        if x < y:
            i2 = i
            if i >= x:
                i1 = random.randint(0, x - 1)
            else:
                i1 = i
        elif x > y:
            i1 = i
            if i >= y:
                i2 = random.randint(0, y - 1)
            else:
                i2 = i
        else:
            i1 = i2 = i
        vector1 = data1.at[i1, 'vector']
        vector2 = data2.at[i2, 'vector']
        vectork1 = literal_eval(vector1)
        vectork2 = literal_eval(vector2)
        for m in range(len(vectork1[0])):
            if m == len(vectork1[0]) - 1:
                f3.write(str(vectork1[0][m]) + "\t")
            else:
                f3.write(str(vectork1[0][m]) + " ")
            if vectork1[0][m] not in embed_file_Dic.keys():
                embed_file_Dic[vectork1[0][m]] = vectork1[0][m]
                embed_file.write(str(vectork1[0][m]) + ' ' + str(vectork1[0][m]) + '\n')
        for n in range(len(vectork2[0])):
            if n == len(vectork2[0]) - 1:
                f3.write(str(vectork2[0][n]))
            else:
                f3.write(str(vectork2[0][n]) + " ")
            if vectork2[0][n] not in embed_file_Dic.keys():
                embed_file_Dic[vectork2[0][n]] = vectork2[0][n]
                embed_file.write(str(vectork2[0][n]) + ' ' + str(vectork2[0][n]) + '\n')
        f3.write('\n')
    return len(embed_file_Dic), len(data1['vector'])


if __name__ == '__main__':
    fpath1 = "dataset/PairVectorToTrain/"
    fpath2 = "dataset/EmbeddingFile/"
    fpath3 = "dataset/VocabSize/"
    if not os.path.exists(fpath1):
        os.makedirs(fpath1)
    if not os.path.exists(fpath2):
        os.makedirs(fpath2)
    if not os.path.exists(fpath3):
        os.makedirs(fpath3)

    Dir_root = "../Biased-Attack/SMDD/AttackData/"

    # ACMC
    trainACMC = os.path.join(Dir_root, "trainForClassifier_ACMC.txt")
    testACMC = os.path.join(Dir_root, "testForClassifier_ACMC.txt")
    ACMCpairvector = os.path.join(fpath1, "ACMCpairvector.txt")
    ACMCembedding = os.path.join(fpath2, "ACMCembedding.txt")
    ACMCvocabsize, ACMCpairsize = preprocess(trainACMC, testACMC, ACMCpairvector, ACMCembedding)
    print("size of ACMC vocab:" + str(ACMCvocabsize))
    print("size of ACMC pair:" + str(ACMCpairsize))
    with open(os.path.join(fpath3, "ACMCvocabsize.txt"), 'w') as f3:
        f3.write("size of ACMC vocab:" + str(ACMCvocabsize))
        f3.write("\n" + "size of ACMC pair:" + str(ACMCpairsize))

    # AGMG
    trainAGMG = os.path.join(Dir_root, "trainForClassifier_AGMG.txt")
    testAGMG = os.path.join(Dir_root, "testForClassifier_AGMG.txt")
    AGMGpairvector = os.path.join(fpath1, "AGMGpairvector.txt")
    AGMGembedding = os.path.join(fpath2, "AGMGembedding.txt")
    AGMGvocabsize, AGMGpairsize = preprocess(trainAGMG, testAGMG, AGMGpairvector, AGMGembedding)
    print("size of AGMG vocab:" + str(AGMGvocabsize))
    print("size of AGMG pair:" + str(AGMGpairsize))
    with open(os.path.join(fpath3, "AGMGvocabsize.txt"), 'w') as f3:
        f3.write("size of AGMG vocab:" + str(AGMGvocabsize))
        f3.write("\n" + "size of AGMG pair:" + str(AGMGpairsize))

  

