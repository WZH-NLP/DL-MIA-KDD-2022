# -*- coding: utf-8 -*-
import ast
import pickle

import pandas as pd


# for trainVector
def postprocess(vocabfile, filetohelp, filetoprocess, filetowrite, istest):
    with open(vocabfile, "rb") as fp:
        data = pickle.load(fp)
        vocabDict = data[1]
    newDict = {}
    for key, value in vocabDict.items():
        newDict[value] = key
    print("newDict done!")
    DictMatchFile = {}
    format1 = ['feature_vector', 'label']
    matchfile = pd.read_csv(filetohelp, sep='\t', names=format1)
    for j in range(len(matchfile)):
        vector = matchfile.iloc[j, 0]
        vector = ast.literal_eval(vector)
        vector = [float(v) for v in vector]
        label = matchfile.iloc[j, 1]
        DictMatchFile[str(vector)] = label
    print("DictMatchFile done!")

    fw = open(filetowrite, 'w')
    format2 = ['sentence', 'vector1', 'vector2']
    oldfile = pd.read_csv(filetoprocess, sep='::', header=None, names=format2, engine='python')
    for i in range(len(oldfile)):
        idx = oldfile.iloc[i, 0]
        idx = ast.literal_eval(idx)
        if istest:
            idx = list(map(int, idx))
        else:
            idx = list(map(int, idx))[:-1]
        vec = [float(newDict[i]) for i in idx]
        label = DictMatchFile[str(vec)]
        v1 = oldfile.iloc[i, 1]
        v2 = oldfile.iloc[i, 2]
        fw.write(str(vec) + '\t' + v1 + '\t' + v2 + '\t' + str(label) + '\n')


if __name__ == '__main__':
    istest = False
    # ACMC
    vocabfile = "vocab_file/ACMC/vocab_1092735"
    filetohelp = "/data/huangna-slurm/HN/y1/code/MIA/MIA-against-SR/Data/ml-1m/processed/user-level/gru4rec/attackData/" \
                 "trainForClassifier.txt"
    filetoprocess = "disentangled_Vector/GG_Train.txt"
    filetowrite = "/data/huangna-slurm/HN/y1/code/MIA/MIA-against-SR/Data/ml-1m/processed/user-level/gru4rec/" \
                  "DisentangledAttackData/trainForClassifier.txt"
    postprocess(vocabfile, filetohelp, filetoprocess, filetowrite, istest)
    print("train data processed!")

    filetoprocess1 = "disentangled_Vector/GG_Test.txt"
    filetowrite1 = "/data/huangna-slurm/HN/y1/code/MIA/MIA-against-SR/Data/ml-1m/processed/user-level/gru4rec/" \
                   "DisentangledAttackData/testForClassifier.txt"
    postprocess(vocabfile, filetohelp, filetoprocess1, filetowrite1, istest=True)
    print("test data processed!")

    # AGMG
    _vocabfile = "vocab_file/GC/vocab_889697"
    _filetohelp = "/data/huangna-slurm/HN/y1/code/MIA/MIA-against-SR/Data/ml-1m/processed/user-level/gru4rec/attackData/" \
                 "trainForClassifier.txt"
    _filetoprocess = "disentangled_Vector/GC_Train.txt"
    _filetowrite = "/data/huangna-slurm/HN/y1/code/MIA/MIA-against-SR/Data/ml-1m/processed/user-level/" \
                  "DisentangledAttackData/GCtrainForClassifier.txt"
    postprocess(_vocabfile, _filetohelp, _filetoprocess, _filetowrite, istest)
    print("train data processed!")

    _filetohelp1 = "/data/huangna-slurm/HN/y1/code/MIA/MIA-against-SR/Data/ml-1m/processed/user-level/caser/attackData/" \
                  "trainForClassifier.txt"
    _filetoprocess1 = "disentangled_Vector/GC_Test.txt"
    _filetowrite1 = "/data/huangna-slurm/HN/y1/code/MIA/MIA-against-SR/Data/ml-1m/processed/user-level/" \
                   "DisentangledAttackData/GCtestForClassifier.txt"
    postprocess(_vocabfile, _filetohelp1, _filetoprocess1, _filetowrite1, istest=True)
    print("test data processed!")


