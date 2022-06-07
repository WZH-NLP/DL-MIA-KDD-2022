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


def preproess(fpath1, fpath2, fpath3, fpath4):
    # construct dataset (pair)
    format = ['vector', 'label']
    data1 = pd.read_csv(fpath1, sep='\t', header=None, names=format, engine='python')
    data2 = pd.read_csv(fpath2, sep='\t', header=None, names=format, engine='python')
    set_seed(1234)
    state1 = np.random.get_state()
    data2 = shuffle(data2).reset_index()
    f3 = open(fpath3, 'w')
    embed_file = open(fpath4, 'w')
    embed_file_Dic = {}
    print("len(data1):", len(data1['vector']))
    print("len(data2):", len(data2['vector']))
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

    # DMDD
    Dir_root = "../Biased-Attack/DMDD/AttackData/"

    train_AIML = os.path.join(Dir_root, "trainForClassifier_AIML.txt")
    test_AIML = os.path.join(Dir_root, "testForClassifier_AIML.txt")
    train_ALMI = os.path.join(Dir_root, "trainForClassifier_ALMI.txt")
    test_ALMI = os.path.join(Dir_root, "testForClassifier_ALMI.txt")

    train_MIAL = os.path.join(Dir_root, "trainForClassifier_MIAL.txt")
    test_MIAL = os.path.join(Dir_root, "testForClassifier_MIAL.txt")
    train_MLAI = os.path.join(Dir_root, "trainForClassifier_MLAI.txt")
    test_MLAI = os.path.join(Dir_root, "testForClassifier_MLAI.txt")

    # AIML
    AIMLpairvector = os.path.join(fpath1, "AIMLpairvector.txt")
    AIMLembedding = os.path.join(fpath2, "AIMLembedding.txt")
    AIMLvocabsize, AIMLpairsize = preproess(train_AIML, test_AIML, AIMLpairvector, AIMLembedding)
    print("size of ml-1m AIML vocab:" + str(AIMLvocabsize))
    print("size of ml-1m AIML pair:" + str(AIMLpairsize))
    with open(os.path.join(fpath3, "AIMLvocabsize.txt"), 'w') as f3:
        f3.write("size of AIML vocab:" + str(AIMLvocabsize))
        f3.write("\n" + "size of AIML pair:" + str(AIMLpairsize))

    # ALMI
    ALMIpairvector = os.path.join(fpath1, "ALMIpairvector.txt")
    ALMIembedding = os.path.join(fpath2, "ALMIembedding.txt")
    ALMIvocabsize, ALMIpairsize = preproess(train_ALMI, test_ALMI, ALMIpairvector, ALMIembedding)
    print("size of ml-1m ALMI vocab:" + str(ALMIvocabsize))
    print("size of ml-1m ALMI pair:" + str(ALMIpairsize))
    with open(os.path.join(fpath3, "ALMIvocabsize.txt"), 'w') as f3:
        f3.write("size of ALMI vocab:" + str(ALMIvocabsize))
        f3.write("\n" + "size of ALMI pair:" + str(ALMIpairsize))


    # MIAL
    MIALpairvector = os.path.join(fpath1, "MIALpairvector.txt")
    MIALembedding = os.path.join(fpath2, "MIALembedding.txt")
    MIALvocabsize, MIALpairsize = preproess(train_MIAL, test_MIAL, MIALpairvector, MIALembedding)
    print("size of ml-1m MIAL vocab:" + str(MIALvocabsize))
    print("size of ml-1m MIAL pair:" + str(MIALpairsize))
    with open(os.path.join(fpath3, "MIALvocabsize.txt"), 'w') as f3:
        f3.write("size of MIAL vocab:" + str(MIALvocabsize))
        f3.write("\n" + "size of MIAL pair:" + str(MIALpairsize))

    # MLAI
    MLAIpairvector = os.path.join(fpath1, "MLAIpairvector.txt")
    MLAIembedding = os.path.join(fpath2, "MLAIembedding.txt")
    MLAIvocabsize, MLAIpairsize = preproess(train_MLAI, test_MLAI, MLAIpairvector, MLAIembedding)
    print("size of ml-1m MLAI vocab:" + str(MLAIvocabsize))
    print("size of ml-1m MLAI pair:" + str(MLAIpairsize))
    with open(os.path.join(fpath3, "MLAIvocabsize.txt"), 'w') as f3:
        f3.write("size of MLAI vocab:" + str(MLAIvocabsize))
        f3.write("\n" + "size of MLAI pair:" + str(MLAIpairsize))
