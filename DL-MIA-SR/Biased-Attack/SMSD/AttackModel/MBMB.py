import os

import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# debug
# import pydevd_pycharm
# pydevd_pycharm.settrace('172.25.231.221', port=3931, stdoutToServer=True, stderrToServer=True)

num_latent = 100
num_epoch = 15

def set_seed(seed=2021):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(num_latent, 32)
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, 2)
        # self.dropout = nn.Dropout(p=0.6)

    def forward(self, x):
        x = x.view(-1, num_latent)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.dropout(self.fc1(x)))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.dropout(self.fc2(x)))
        x = F.softmax(self.fc3(x))
        return x


topk = 100

pathS = "../../../Recommender/BERT4Rec-Pytorch-master/mydata"
pathT = "../../../Recommender/BERT4Rec-Pytorch-master/mydata"
Vpath = "/data/huangna-slurm/HN/y1/code/MIA/MIA-against-SR/Data/DataProcessor/Vectorized_itemEmbed"
path = "/data/huangna-slurm/HN/y1/code/MIA/MIA-against-SR/Data/data/processed_movielens/"
# ------------------shadow------------------
fr_vector_shadow = open(Vpath + "/ml-1m_itemMatrix.txt", 'r')  # vector for shadow items
itemDict = {}
with open(path + "ml-1m_itemDict", 'r') as f:
    for line0 in f.readlines():
        line0 = line0.strip().split('\t')
        itemDict[int(line0[1])] = int(line0[0])

vectors1 = {}  # vectors for shadow items
index = -1
for line in fr_vector_shadow.readlines():
    index = index + 1
    line = line.split(' ')
    line = list(map(float, line))
    t_vectors = torch.tensor(line)
    vectors1[itemDict[index]] = t_vectors
# read recommends
Smember_rec = open(pathS + "/ml-1m_Smember_recommendations", 'r')  # recommend for shadow
recommend_Smember = {}
for line in Smember_rec.readlines():
    line = line.split('\t')
    sessionID = line[0]
    itemID = line[1]
    recommend_Smember.setdefault(int(sessionID), []).append(int(itemID))
# print("length of recommend_Smember.keys:", len(recommend_Smember.keys()))
# print("recommend_Smember.keys:", sorted(recommend_Smember.keys()))
recommend_Snonmem = {}  # recommends for shadow
Snonmem_rec = open(pathS + "/ml-1m_Snonmem_recommendation", 'r')  # recommend for shadow
for line in Snonmem_rec.readlines():
    line = line.split('\t')
    sessionID = line[0]
    itemID = line[1]
    recommend_Snonmem.setdefault(int(sessionID), []).append(int(itemID))

# read interactions
itm = open(pathS + "/ml-1m_Smember_train", 'r')  # interactions for target_member
itn = open(pathS + "/ml-1m_Snonmem_train", 'r')  # interactions for target_member
interaction_Smember = {}  # interactions for target_member
interaction_Snonmem = {}  # interactions for target_nonmem
for line in itm.readlines():
    line = line.split('\t')
    if line[0] != 'SessionID':
        sessionID = line[0]
        itemID = line[1]
        interaction_Smember.setdefault(int(sessionID), []).append(int(itemID))
# print("length of interactions_Smember.keys:", len(interaction_Smember.keys()))
# print("interactions_Smember.keys:", sorted(interaction_Smember.keys()))

for line in itn.readlines():
    line = line.split(',')
    if line[0] != 'SessionID':
        sessionID = line[0]
        itemID = line[1]
        interaction_Snonmem.setdefault(int(sessionID), []).append(int(itemID))

# print("length of recommend_Snonmem.keys:", len(recommend_Snonmem.keys()))
# print("recommend_Snonmem.keys:", sorted(recommend_Snonmem.keys()))
# print("length of interaction_Snonmem.keys:", len(interaction_Snonmem.keys()))
# print("interaction_Snonmem.keys:", sorted(interaction_Snonmem.keys()))
# print("recommend_Snonmem == interaction_Snonmem(key):", recommend_Snonmem.keys() == interaction_Snonmem.keys())

# vectorization for shadow_member
label_shadow_member = {}
vector_shadow_member = {}
vector_shadow_member1 = {}  # vectors for shadow member interaction
for key, value in interaction_Smember.items():
    # key是userID， value为推荐列表
    label_shadow_member[key] = torch.tensor([1])
    temp_vector = torch.zeros(num_latent)
    length = len(value)
    for i in range(len(value)):
        if value[i] in vectors1.keys():
            temp_vector = temp_vector + vectors1[value[i]]
        else:
            length = length - 1
    temp_vector = temp_vector / length
    vector_shadow_member1[key] = temp_vector
vector_shadow_member2 = {}  # vectors for shadow member recommendation
for key, value in recommend_Smember.items():
    temp_vector = torch.zeros(num_latent)
    length = len(value)
    for i in range(len(value)):
        if value[i] in vectors1.keys():
            temp_vector = temp_vector + vectors1[value[i]]
        else:
            length = length - 1
    temp_vector = temp_vector / length
    vector_shadow_member2[key] = temp_vector
    vector_shadow_member[key] = vector_shadow_member1[key] - vector_shadow_member2[key]


# vectorization for shadow_nonmember
label_shadow_nonmem = {}
vector_shadow_nonmem = {}
vector_shadow_nonmem1 = {}  # vectors for shadow nonmember interaction
for key, value in interaction_Snonmem.items():
    # key是userID， value为交互历史
    label_shadow_nonmem[key] = torch.tensor([0])
    temp_vector = torch.zeros(num_latent)
    length = len(value)
    for i in range(len(value)):
        if value[i] in vectors1.keys():
            temp_vector = temp_vector + vectors1[value[i]]
        else:
            length = length - 1
    temp_vector = temp_vector / length
    vector_shadow_nonmem1[key] = temp_vector
vector_shadow_nonmem2 = {}  # vectors for shadow nonmember recommendation
for key, value in recommend_Snonmem.items():
    temp_vector = torch.zeros(num_latent)
    length = len(value)
    for i in range(len(value)):
        if value[i] in vectors1.keys():
            temp_vector = temp_vector + vectors1[value[i]]
        else:
            length = length - 1
    temp_vector = temp_vector / length
    vector_shadow_nonmem2[key] = temp_vector
    vector_shadow_nonmem[key] = vector_shadow_nonmem1[key] - vector_shadow_nonmem2[key]

num_shadow = len(vector_shadow_member)+len(vector_shadow_nonmem)  # 2069
vector_shadow = [[]]*num_shadow
label_shadow = [[]]*num_shadow
idx = -1
for key, value in vector_shadow_member.items():
    idx = idx + 1
    vector_shadow[idx] = value
    label_shadow[idx] = label_shadow_member[key].long()  # member
for key, value in vector_shadow_nonmem.items():
    idx = idx + 1
    vector_shadow[idx] = value
    label_shadow[idx] = label_shadow_nonmem[key].long()  # non_member

# ------------Target----------------
fr_vector_target = open(Vpath + "/ml-1m_itemMatrix.txt", 'r')  # vector for target items
vectors2 = {}  # vectors for target items
index1 = -1
for line in fr_vector_target.readlines():
    index1 = index1 + 1
    line = line.split(' ')
    line = list(map(float, line))
    t_vectors = torch.tensor(line)
    vectors2[itemDict[index1]] = t_vectors

# read recommends
recommend_Tmember = {}  # recommends for target_member
Tmember_rec = open(pathT + "/ml-1m_Tmember_recommendations", 'r')  # recommend for target
for line in Tmember_rec.readlines():
    line = line.split('\t')
    sessionID = line[0]
    itemID = line[1]
    recommend_Tmember.setdefault(int(sessionID), []).append(int(itemID))
recommend_Tnonmem = {}  # recommends for target_nonmem
Tnonmem_rec = open(pathT + "/ml-1m_Tnonmem_recommendation", 'r')  # recommend for target
for line in Tnonmem_rec.readlines():
    line = line.split('\t')
    sessionID = line[0]
    itemID = line[1]
    recommend_Tnonmem.setdefault(int(sessionID), []).append(int(itemID))

# read interactions
itm = open(pathT + "/ml-1m_Tmember_train", 'r')  # interactions for target_member
interaction_Tmember = {}  # interactions for shadow
for line in itm.readlines():
    line = line.split('\t')
    if line[0] != 'SessionID':
        sessionID = line[0]
        itemID = line[1]
        interaction_Tmember.setdefault(int(sessionID), []).append(int(itemID))
itn = open(pathT + "/ml-1m_Tnonmem_train", 'r')  # interactions for target_nonmember
interaction_Tnonmem = {}  # interactions for target
for line in itn.readlines():
    line = line.split(',')
    if line[0] != 'SessionID':
        sessionID = line[0]
        itemID = line[1]
        interaction_Tnonmem.setdefault(int(sessionID), []).append(int(itemID))

# print("length of recommend_Tmember.keys:", len(recommend_Tmember.keys()))
# print("recommend_Tmember.keys:", sorted(recommend_Tmember.keys()))
# print("length of interactions_Tmember.keys:", len(interaction_Tmember.keys()))
# print("interactions_Tmember.keys:", sorted(interaction_Tmember.keys()))
#
#
# print("length of recommend_Tnonmem.keys:", len(recommend_Tnonmem.keys()))
# print("recommend_Tnonmem.keys:", sorted(recommend_Tnonmem.keys()))
# print("length of interaction_Tnonmem.keys:", len(interaction_Tnonmem.keys()))
# print("interaction_Tnonmem.keys:", sorted(interaction_Tnonmem.keys()))

# vectorization for target_member
label_target_member = {}
vector_target_member = {}

vector_target_member1 = {}
for key, value in interaction_Tmember.items():
    # key是userID， value为推荐列表
    label_target_member[key] = torch.tensor([1])
    temp_vector = torch.zeros(num_latent)
    length = len(value)
    for i in range(len(value)):
        if value[i] in vectors2.keys():
            temp_vector = temp_vector + vectors2[value[i]]
        else:
            length = length - 1
    temp_vector = temp_vector / length
    vector_target_member1[key] = temp_vector
vector_target_member2 = {}
for key, value in recommend_Tmember.items():
    temp_vector = torch.zeros(num_latent)
    length = len(value)
    for i in range(len(value)):
        if value[i] in vectors2.keys():
            temp_vector = temp_vector + vectors2[value[i]]
        else:
            length = length - 1
    temp_vector = temp_vector / length
    vector_target_member2[key] = temp_vector
    vector_target_member[key] = vector_target_member1[key] - vector_target_member2[key]

# vectorization for target_nonmember
label_target_nonmem = {}
vector_target_nonmem = {}
vector_target_nonmem1 = {}  # vectors for shadow member interaction
for key, value in interaction_Tnonmem.items():
    # key是userID， value为推荐列表
    label_target_nonmem[key] = torch.tensor([0])
    temp_vector = torch.zeros(num_latent)
    length = len(value)
    for i in range(len(value)):
        if value[i] in vectors2.keys():
            temp_vector = temp_vector + vectors2[value[i]]
        else:
            length = length - 1
    temp_vector = temp_vector / length
    vector_target_nonmem1[key] = temp_vector
vector_target_nonmem2 = {}  # vectors for shadow member recommendation
for key, value in recommend_Tnonmem.items():
    temp_vector = torch.zeros(num_latent)
    length = len(value)
    for i in range(len(value)):
        if value[i] in vectors2.keys():
            temp_vector = temp_vector + vectors2[value[i]]
        else:
            length = length - 1
    temp_vector = temp_vector / length
    vector_target_nonmem2[key] = temp_vector
    vector_target_nonmem[key] = vector_target_nonmem1[key] - vector_target_nonmem2[key]

num_target = len(vector_target_member)+len(vector_target_nonmem)  # 1939
# print("num_target(1939):", num_target)

vector_target = [[]] * num_target
label_target = [[]] * num_target
idx1 = -1
for key, value in vector_target_member.items():
    idx1 = idx1 + 1
    vector_target[idx1] = value
    label_target[idx1] = label_target_member[key].long()  # member
for key, value in vector_target_nonmem.items():
    idx1 = idx1 + 1
    vector_target[idx1] = value
    label_target[idx1] = label_target_nonmem[key].long()  # non_member

DataDir = "../AttackData/"
if not os.path.exists(DataDir):
    os.mkdir(DataDir)
ShadowDataset = open("../AttackData/trainForClassifier_MBMB.txt", 'w')
TargetDataset = open("../AttackData/testForClassifier_MBMB.txt", 'w')
for i in range(num_shadow):
    ShadowDataset.write(str(vector_shadow[i].unsqueeze(0).tolist()) + '\t' + str(label_shadow[i].tolist()) + '\n')
for i in range(num_target):
    TargetDataset.write(str(vector_target[i].unsqueeze(0).tolist()) + '\t' + str(label_target[i].tolist()) + '\n')


set_seed(2021)
state = np.random.get_state()
np.random.shuffle(vector_shadow)
np.random.set_state(state)
np.random.shuffle(label_shadow)

mlp = MLP()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(mlp.parameters(), lr=0.01, momentum=0.7)


# accuarcy
def AccuarcyCompute(pred, label):
    pred = pred.cpu().data.numpy()
    label = label.cpu().data.numpy()
    #    print(pred.shape(),label.shape())
    test_np = (np.argmax(pred, 1) == label)
    test_np = np.float32(test_np)
    return np.mean(test_np)


losses = []
acces = []
eval_losses = []
eval_acces = []

for x in range(num_epoch):
    train_loss = 0
    train_acc = 0
    for i in range(num_shadow):
        optimizer.zero_grad()

        inputs = torch.autograd.Variable(vector_shadow[i])
        labels = torch.autograd.Variable(label_shadow[i])

        outputs = mlp(inputs)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()
        train_loss += loss.item()

        _, pred = outputs.max(1)
        if(int(pred)) == labels.numpy()[0]:
            train_acc += 1

        # print(x, ":", AccuarcyCompute(outputs, labels))
    losses.append(train_loss / num_shadow)
    acces.append(train_acc / num_shadow)
    print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}'.format(x, train_loss / (num_shadow),
                                                                    train_acc / (num_shadow)))

acc_ans = 0
TruePositive = 0
FalsePositive = 0
TrueNegative = 0
FalseNegative = 0
for i in range(num_target):
    inputs = torch.autograd.Variable(vector_target[i])
    labels = torch.autograd.Variable(label_target[i])
    outputs = mlp(inputs)
    # print(outputs)
    _, pred = outputs.max(1)
    if int(pred) == labels.numpy()[0]:
        acc_ans += 1
        if int(pred) == 1:
            TruePositive = TruePositive + 1
        else:
            TrueNegative = TrueNegative + 1
    else:
        if int(pred) == 1:
            FalsePositive = FalsePositive + 1
        else:
            FalseNegative = FalseNegative + 1

print('TruePositive:', TruePositive)
print('FalsePositive:', FalsePositive)
print('TrueNegative:', TrueNegative)
print('FalseNegative', FalseNegative)
print("accuarcy: ")
print((acc_ans / num_target))
print("precsion: ")
print((TruePositive / (TruePositive + FalsePositive)))
print("recall: ")
print((TruePositive / (TruePositive + FalseNegative)))

TPRate = TruePositive / (TruePositive + FalseNegative)
FPRate = FalsePositive / (FalsePositive + TrueNegative)
area = 0.5 * TPRate * FPRate + 0.5 * (TPRate + 1) * (1 - FPRate)
print("AUC: ")
print(area)
