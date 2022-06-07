# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 20:36:48 2018

@author: lanlandetian
"""

import UserCF
import ItemCF
import ItemCF_IUF
import random
import Evaluation
import LFM

import imp

imp.reload(UserCF)
imp.reload(ItemCF)
imp.reload(ItemCF_IUF)
imp.reload(Evaluation)
imp.reload(LFM)


def readData():
    data = []
    fileName = '../data/processed_movielens/ml-1m_vector'
    # fileName = '../data/processed_amazon/beauty_vector'
    fr = open(fileName, 'r')
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        if lineArr[0] != 'SessionID':
            data.append([lineArr[0], lineArr[1], 1.0])
    return data


def SplitData(data, M, k, seed):
    # test = []
    train = []
    random.seed(seed)
    for user, item, rating in data:
        train.append([user, item, rating])
        # if random.randint(0, M - 1) == k:
        #     test.append([user, item, rating])
        # else:
        #     train.append([user, item, rating])
    return train

def SplitData_test(data, M, k, seed):
    test = []
    # train = []
    random.seed(seed)
    for user, item, rating in data:
        if random.randint(0, M - 1) == k:
            test.append([user, item, rating])
        # else:
        #     train.append([user, item, rating])
    return test


def transform(oriData):
    ret = dict()
    for user, item, rating in oriData:
        if user not in ret:
            ret[user] = dict()
        ret[user][item] = rating
    return ret


if __name__ == '__main__':
    data = readData()
    numFlod = 1
    precision = 0
    recall = 0
    coverage = 0
    popularity = 0
    for i in range(0, numFlod):
        oriTrain = SplitData(data, numFlod, i, 0)
        oriTest = SplitData_test(data, numFlod, i, 0)
        train = transform(oriTrain)
        test = transform(oriTest)

        [P, Q] = LFM.LatentFactorModel(train, 100, 30, 0.02, 0.01)
        fw = open('../Vectorized_itemEmbed/ml-1m_itemMatrix_pre', 'w')
        for key, value in Q.items():
            fw.write(str(key)+'\t'+str(value)+'\n')
        # fw = open('../../../Vectorized_itemEmbed/beauty_itemMatrix_pre', 'w')
        # for key, value in Q.items():
        #     fw.write(str(key) + '\t' + str(value) + '\n')

        rank = LFM.Recommend('2', train, P, Q)
        result = LFM.Recommendation(test.keys(), train, P, Q)

        N = 10
        precision += Evaluation.Precision(train, test, result, N)
        recall += Evaluation.Recall(train, test, result, N)
        coverage += Evaluation.Coverage(train, test, result, N)
        popularity += Evaluation.Popularity(train, test, result, N)

    precision /= numFlod
    recall /= numFlod
    coverage /= numFlod
    popularity /= numFlod

    # 输出结果
    print('precision = %f' % precision)
    print('recall = %f' % recall)
    print('coverage = %f' % coverage)
    print('popularity = %f' % popularity)

