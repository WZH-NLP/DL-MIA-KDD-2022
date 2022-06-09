# generate recommendations for member(both target and shadow model) by lfm, non-member by popoularity

import numpy as np
import math
import random

topk = 100
num_negative = 1
num_latent = 100
num_items_shadow = 59735
num_items_target = 60120
num_target = 4442  # users in target 0-2634
num_shadow = 4442  # users in shadow 0-2634
# num_vector = 4441  # users in vector 0-2634

path = "/data/huangna-slurm/HN/y1/code/MIA/MIA-against-SR/Data/data/processed_amazon/"
directory_target = "digital_music_target"  # users 0-104510
directory_shadow = "digital_music_shadow"  # users 0-104267
# directory_vector = "digital_music_vector"  # users 0-104408


def lfm(a, k):
    assert type(a) == np.ndarray
    m, n = a.shape
    # print(m, n)
    alpha = 0.01
    lambda_ = 0.01
    u = np.random.rand(m, k)
    v = np.random.randn(k, n)
    t = 1
    err_total = 1
    while err_total >= 100 and t <= 20:
        err_total = 0
        for i in range(m):
            for j in range(n):
                if a[i][j] != -1:
                    err = a[i][j] - np.dot(u[i], v[:, j])
                    err_total = err_total + err
                    for r in range(k):
                        gu = err * v[r][j] - lambda_ * u[i][r]
                        gv = err * u[i][r] - lambda_ * v[r][j]
                        u[i][r] += alpha * gu
                        v[r][j] += alpha * gv
        print('total error in round ' + str(t) + ' is: ' + str(err_total))
        t = t + 1
    return u, v


def init_matrix(num_users, num_items):
    A = []
    for i in range(num_users):
        A.append([])
        for j in range(num_items):
            A[i].append(-1)
    return A


def sample(direct, matrix_, visit_, num_items):
    A = matrix_
    B = visit_
    dict_temp = {}
    for i in range(len(A)):
        dict_temp[i] = []
    f = open(direct, 'r')
    line = f.readline()
    while line != '' and line is not None and line != ' ':
        arr = line.split('\t')  # arr[0] user   arr[1] item
        dict_temp[int(arr[0])].append(int(arr[1]))
        A[int(arr[0])][int(arr[1])] = 1
        B[int(arr[0])][int(arr[1])] = 1
        line = f.readline()
    for i in range(len(A)):
        temp_times = len(dict_temp[i])
        for j in range(num_negative * temp_times):
            rand = random.randint(0, num_items - 1)
            while rand in dict_temp[i]:
                rand = random.randint(0, num_items - 1)
            dict_temp[i].append(rand)
            A[i][rand] = 0
            B[i][rand] = 1
    f.close()
    return A, B


def sample_(direct, matrix_, visit_, num_users, num_items):
    A = matrix_
    B = visit_
    dict_temp = {}
    for i in range(len(A)):
        dict_temp[i] = []
    f = open(direct, 'r')
    line = f.readline()
    while line != '' and line is not None and line != ' ':
        arr = line.split('\t')  # arr[0] user   arr[1] item
        if int(arr[0]) >= num_users:
            break
        dict_temp[int(arr[0])].append(int(arr[1]))
        A[int(arr[0])][int(arr[1])] = 1
        B[int(arr[0])][int(arr[1])] = 1
        line = f.readline()

    for i in range(len(A)):
        temp_times = len(dict_temp[i])
        for j in range(num_negative * temp_times):
            rand = random.randint(0, num_items - 1)
            while rand in dict_temp[i]:
                rand = random.randint(0, num_items - 1)
            dict_temp[i].append(rand)
            A[i][rand] = 0
            B[i][rand] = 1
    f.close()
    return A, B


'''
A = np.array([[5, 5, 0, 5], [5, 0, 3, 4], [3, 4, 0, 3], [0, 0, 5, 3], [5, 4, 4, 5], [5, 4, 5, 5]])
b, c = lfm(A, 3)
'''

matrix_target = init_matrix(num_target // 2, num_items_target)
print('matrix_target init done')
matrix_shadow = init_matrix(num_shadow // 2, num_items_shadow)
print('matrix_shadow init done')
# matrix_vector = init_matrix(num_vector)
# print('matrix_vector init done')

visit_target = init_matrix(num_target // 2, num_items_target)
print('visit_target init done')
visit_shadow = init_matrix(num_shadow // 2, num_items_shadow)
print('visit_shadow init done')
# visit_vector = init_matrix(num_vector)
# print('visit_vector init done')

matrix_target, visit_target = sample_(directory_target, matrix_target, visit_target, num_target // 2, num_items_target)
print('target sample done')
matrix_shadow, visit_shadow = sample_(directory_shadow, matrix_shadow, visit_shadow, num_shadow // 2, num_items_shadow)
print('shadow sample done')
# matrix_vector, visit_vector = sample(directory_vector, matrix_vector, visit_vector)
# print('vector sample done')

matrix_target = np.array(matrix_target)
matrix_shadow = np.array(matrix_shadow)
# matrix_vector = np.array(matrix_vector)

user_target, item_target = lfm(matrix_target, num_latent)
print('lfm for target done')
user_shadow, item_shadow = lfm(matrix_shadow, num_latent)
print('lfm for shadow done')
# user_vector, item_vector = lfm(matrix_vector, num_latent)
# print('lfm for vector done')

fw_target = open('recommendations/amazon_music_topk_target_popular_lfm', 'w')
fw_shadow = open('recommendations/amazon_music_topk_shadow_popular_lfm', 'w')
# fw_vector = open('amazon_music_topk_vector', 'w')
'''
# vector
item_vector = item_vector.tolist()
for i in range(num_items):
    for j in range(num_latent):
        fw_vector.write(str(item_vector[j][i])+'\t')
    fw_vector.write('\n')
    print('vector of item '+str(i)+' done')
fw_vector.close()
'''
# target
matrix_target = np.dot(user_target, item_target)
matrix_target = matrix_target.tolist()
for i in range(num_target // 2):
    dict_target = {}
    for j in range(num_items_target):
        dict_target[j] = float(matrix_target[i][j])
    a_target = sorted(dict_target.items(), key=lambda x: x[1], reverse=True)
    index = 0
    for j in range(topk):
        # while visit_target[i][a_target[index][0]] == 1:
        #    index += 1
        print(str(j) + 'th item for target user ' + str(i) + 'is ' + str(a_target[index][0])
              + ' and its score is ' + str(a_target[index][1]))
        fw_target.write(str(i) + '\t' + str(a_target[index][0]) + '\t' + str(a_target[index][1]) + '\t' + '\n')
        index += 1

f_popular = open(directory_target, 'r')
list_temp = {}
done_temp = []
for i in range(num_items_target):
    list_temp[i] = 0
for i in range(num_target - num_target // 2):
    done_temp.append([])
line = f_popular.readline()
while line != '' and line is not None and line != ' ':
    arr = line.split('\t')
    if int(arr[0]) >= (num_target // 2):
        list_temp[int(arr[1])] = list_temp[int(arr[1])] + 1
        done_temp[int(arr[0]) - num_target // 2].append(int(arr[1]))
    line = f_popular.readline()
a_target = sorted(list_temp.items(), key=lambda x: x[1], reverse=True)
for i in range(num_target - num_target // 2):
    index = 0
    for j in range(topk):
        # while a_target[index][0] in done_temp[i]:
        #    index += 1
        print(str(j) + 'th item for target user ' + str(i) + 'is ' + str(a_target[index][0])
              + ' and its score is ' + str(a_target[index][1]))
        fw_target.write(
            str(i + num_target // 2) + '\t' + str(a_target[index][0]) + '\t' + str(a_target[index][1]) + '\t' + '\n')
        index += 1
fw_target.close()

# shadow
matrix_shadow = np.dot(user_shadow, item_shadow)
matrix_shadow = matrix_shadow.tolist()
for i in range(num_shadow // 2):
    dict_shadow = {}
    for j in range(num_items_target):
        dict_shadow[j] = float(matrix_shadow[i][j])
    a_shadow = sorted(dict_shadow.items(), key=lambda x: x[1], reverse=True)
    index = 0
    for j in range(topk):
        # while visit_shadow[i][a_shadow[index][0]] == 1:
        #    index += 1
        print(str(j) + 'th item for shadow user ' + str(i) + 'is ' + str(a_shadow[index][0])
              + ' and its score is ' + str(a_shadow[index][1]))
        fw_shadow.write(str(i) + '\t' + str(a_shadow[index][0]) + '\t' + str(a_shadow[index][1]) + '\t' + '\n')
        index += 1

f_popular = open(directory_shadow, 'r')
list_temp = {}
done_temp = []
for i in range(num_items_target):
    list_temp[i] = 0
for i in range(num_shadow - num_shadow // 2):
    done_temp.append([])
line = f_popular.readline()
while line != '' and line is not None and line != ' ':
    arr = line.split('\t')
    if int(arr[0]) >= (num_shadow // 2):
        list_temp[int(arr[1])] = list_temp[int(arr[1])] + 1
        done_temp[int(arr[0]) - num_shadow // 2].append(int(arr[1]))
    line = f_popular.readline()
a_shadow = sorted(list_temp.items(), key=lambda x: x[1], reverse=True)
for i in range(num_shadow - num_shadow // 2):
    index = 0
    for j in range(topk):
        # while a_shadow[index][0] in done_temp[i]:
        #    index += 1
        print(str(j) + 'th item for shadow user ' + str(i) + 'is ' + str(a_shadow[index][0])
              + ' and its score is ' + str(a_shadow[index][1]))
        fw_shadow.write(
            str(i + num_shadow // 2) + '\t' + str(a_shadow[index][0]) + '\t' + str(a_shadow[index][1]) + '\t' + '\n')
        index += 1
fw_shadow.close()

'''
# shadow
matrix_shadow = np.dot(user_shadow, item_shadow)
matrix_shadowt = matrix_shadow.tolist()
for i in range(num_shadow):
    dict_shadow = {}
    for j in range(num_items):
        dict_shadow[j] = float(matrix_shadow[i][j])
    a_shadow = sorted(dict_shadow.items(), key=lambda x: x[1], reverse=True)
    index = 0
    for j in range(topk):
        while visit_shadow[i][a_shadow[index][0]] == 1:
            index += 1
        print(str(j)+'th item for shadow user '+str(i)+'is '+str(a_shadow[index][0])
            +' and its score is '+str(a_shadow[index][1]))
        fw_shadow.write(str(i)+'\t'+str(a_shadow[index][0])+'\t'+str(a_shadow[index][1])+'\t'+'\n')
        index += 1
fw_shadow.close()
'''
