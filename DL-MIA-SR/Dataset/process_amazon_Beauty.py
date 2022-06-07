import numpy as np
import pandas as pd
import random


def removeShortSeq(data):
    seqLen = data.groupby('UserID').size()
    data = data[np.in1d(data['UserID'], seqLen[seqLen > 4].index)]
    return data


# anonymize & remove users and items with less than 5 interactions records
# and record dataset statistics
def step0(file1, file2):
    dict_user = {}
    dict_item = {}
    index_user = 0
    index_item = 0
    set_user = set()
    set_item = set()
    fw = open(file2, 'w')
    data = pd.read_csv(file1, sep=',', header=None)
    data.columns = ['UserID', 'ItemID', 'Rating', 'TimeStamp']
    data = removeShortSeq(data)
    itemLen = data.groupby('ItemID').size()
    data = data[np.in1d(data.ItemID, itemLen[itemLen > 4].index)]
    data = removeShortSeq(data)

    print("users:", data['UserID'].nunique())
    print("Items:", data['ItemID'].nunique())
    print("Avg.Item:", data.groupby('UserID').count().mean())
    print("Avg.User:", data.groupby('ItemID').count().mean())

    userGroup = data.groupby('UserID')
    for userID, userSeq in userGroup:
        if userID not in set_user:
            dict_user[userID] = index_user
            index_user = index_user + 1
            set_user.add(userID)
        ItemList = userSeq['ItemID'].to_list()
        RatingList = userSeq['Rating'].to_list()
        TimeStampList = userSeq['TimeStamp'].to_list()
        for k in range(len(ItemList)):
            item = ItemList[k]
            rating = RatingList[k]
            timestamp = TimeStampList[k]
            if item not in set_item:
                dict_item[item] = index_item
                index_item = index_item + 1
                set_item.add(item)
            fw.write(str(dict_user[userID]) + ',' + str(dict_item[item]) + ','
                     + str(rating) + "," + str(timestamp) + '\n')

    fuser = open("data/processed_amazon/Beauty_usermap", 'w')
    fitem = open("data/processed_amazon/Beauty_itemmap", 'w')

    for key, value in dict_user.items():
        fuser.write(str(key) + ':' + str(value) + '\n')
    for key, value in dict_item.items():
        fitem.write(str(key) + ':' + str(value) + '\n')


# split dataset to shadow/target/vectorize_pre
def step1(file1):
    data = pd.read_csv(file1, sep=',', header=None)
    data.columns = ['UserID', 'ItemID', 'Rating', 'TimeStamp']
    userGroup = data.groupby('UserID')

    path = "data/processed_amazon/"
    num_target = num_shadow = num_vector = 0

    for userID, userSeq in userGroup:
        rand = random.randint(1, 3)
        if rand == 1:
            userSeq.to_csv(path + 'beauty_vector_pre', sep=',', index=False, mode='a', header=None)
            num_vector = num_vector + 1
        elif rand == 2:
            userSeq.to_csv(path + 'beauty_shadow_pre', sep=',', index=False, mode='a', header=None)
            num_shadow = num_shadow + 1
        else:
            userSeq.to_csv(path + 'beauty_target_pre', sep=',', index=False, mode='a', header=None)
            num_target = num_target + 1

    print("num_vector:", num_vector)
    print("num_shadow:", num_shadow)
    print("num_target:", num_target)


# reset index for user
def step2(fr, fw):
    path = "data/processed_amazon/"
    f1 = pd.read_csv(path + fr, sep=',')
    f1.columns = ['UserID', 'ItemID', 'Rating', 'TimeStamp']
    f2 = open(path + fw, 'w')  # 1
    userGroup = f1.groupby('UserID')
    user_index = -1
    f2.write('SessionID' + ',' + 'ItemID' + ',' + 'Rating' + ',' + 'Time' + '\n')
    for user, Items in userGroup:
        user_index = user_index + 1
        ItemIds = Items['ItemID'].to_list()
        Ratings = Items['Rating'].to_list()
        TimeStamps = Items['TimeStamp'].to_list()
        for i in range(len(ItemIds)):
            f2.write(
                str(user_index) + ',' + str(ItemIds[i]) + ',' + str(Ratings[i]) + ',' + str(TimeStamps[i]) + '\n')

    f2.close()
    return user_index


# reset index for item of vector dataset
def step3(fr, fw1, fw2):
    path = "data/processed_amazon/"
    f2 = open(path + fw1, 'w')  # 1
    f3 = open(path + fw2, 'w')  # 1
    ItemDict = {}
    ItemSet = set()
    item_index = 0
    first_line = True
    with open(path + fr, 'r') as f:
        for line in f.readlines():
            line = line.strip().split(',')
            SessionID = line[0]
            ItemID = line[1]
            Rating = line[2]
            if first_line:
                f2.write(str(SessionID) + '\t' + str(ItemID) + '\t' + str(Rating) + '\n')
                first_line = False
            else:
                if ItemID not in ItemSet:
                    ItemSet.add(ItemID)
                    ItemDict[ItemID] = item_index
                    item_index = item_index + 1
                f2.write(str(SessionID) + '\t' + str(ItemDict[ItemID]) + '\t' + str(Rating) + '\n')

    for key, value in ItemDict.items():
        f3.write(str(key) + '\t' + str(value) + '\n')

    f2.close()
    f3.close()

    return item_index


if __name__ == '__main__':

    filename = "ratings_Beauty.csv"
    filename1 = "data/processed_amazon/ratings_Beauty_after_step0.csv"
    step0(filename, filename1)
    step1(filename1)

    fr_target = 'beauty_target_pre'
    fr_shadow = 'beauty_shadow_pre'
    fr_vector = 'beauty_vector_pre'

    fw_target = 'beauty_target'
    fw_shadow = 'beauty_shadow'
    fw_vector = 'beauty_vector_pre1'

    target_num = step2(fr_target, fw_target)
    shadow_num = step2(fr_shadow, fw_shadow)
    vector_num = step2(fr_vector, fw_vector)
    print("target_num:{}, shadow_num:{}, vector_num:{}".format(target_num, shadow_num, vector_num))

    fw_vector1 = 'beauty_vector'
    fw_itemdict = 'beauty_itemDict'
    item_num_of_vectorize = step3(fw_vector, fw_vector1, fw_itemdict)
    print("item_num_of_vectorize:", item_num_of_vectorize)
