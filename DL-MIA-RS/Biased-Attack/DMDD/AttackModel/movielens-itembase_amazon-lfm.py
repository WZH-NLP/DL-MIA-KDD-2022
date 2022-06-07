import os

import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(100, 32)
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, 2)
        # self.dropout = nn.Dropout(p=0.6)
    def forward(self, x):
        x = x.view(-1, 100)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.dropout(self.fc1(x)))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.dropout(self.fc2(x)))
        x = F.softmax(self.fc3(x))
        return x

topk = 100
num_latent = 100

num_shadow = 3938 # train
num_target = 2635 # test

num_vector_shadow = 3706
num_vector_target = 284253

path = "../../../Dataset/data_from_ori"

f_shadow = open(path+"/ml-1m/movielens_target", 'r') # interactions for shadow
f_target = open(path+"/amazon/amazon_music_shadow", 'r') # interactions for target

fr_shadow = open(path+"/ml-1m/movielens_topk_target_popular_itembase", 'r') # recommend for shadow
fr_target = open(path+"/amazon/amazon_music_topk_shadow_popular_lfm", 'r') # recommend for target

fr_vector_shadow = open(path+"/ml-1m/movielens_topk_vector", 'r') # vector for shadow items
fr_vector_target = open(path+"/amazon/amazon_music_topk_vector", 'r') # vector for target items

interaction_target = [] # interactions for target
interaction_shadow = [] # interactions for shadow
recommend_target = []   # recommends for target
recommend_shadow = []   # recommends for shadow
vector_target = []  # vectors for target
vector_shadow = []  # vectors for shadow
vectors1 = []    # vectors for shadow items
vectors2 = []    # vectors for target items
label_target = []
label_shadow = []

# vectors for shadow items
for i in range(num_vector_shadow):
    vectors1.append([])
    line = fr_vector_shadow.readline()
    arr = line.split('\t')
    for j in range(100):
        arr[j] = float(arr[j])
        vectors1[i].append(arr[j])
    vectors1[i] = torch.tensor(vectors1[i])   # tensor
    # vectors[i] = F.softmax(vectors[i])      # sum to 1
# print(vectors[0])

# vectors for target items
for i in range(num_vector_target):
    vectors2.append([])
    line = fr_vector_target.readline()
    arr = line.split('\t')
    for j in range(100):
        arr[j] = float(arr[j])
        vectors2[i].append(arr[j])
    vectors2[i] = torch.tensor(vectors2[i])   # tensor
    # vectors[i] = F.softmax(vectors[i])      # sum to 1
# print(vectors[0])

# init for target
for i in range(num_target):
    recommend_target.append([])
    interaction_target.append([])
# read recommends
line = fr_target.readline()
while line != '' and line is not None and line != ' ':
    arr = line.split('\t')
    recommend_target[int(arr[0])].append(int(arr[1]))
    line = fr_target.readline()
# read interactions
line = f_target.readline()
while line != '' and line is not None and line != ' ':
    arr = line.split('\t')
    interaction_target[int(arr[0])].append(int(arr[1]))
    line = f_target.readline()

# init for shadow
for i in range(num_shadow):
    recommend_shadow.append([])
    interaction_shadow.append([])
# read recommends
line = fr_shadow.readline()
while line != '' and line is not None and line != ' ':
    arr = line.split('\t')
    recommend_shadow[int(arr[0])].append(int(arr[1]))
    line = fr_shadow.readline()
# read interactions
line = f_shadow.readline()
while line != '' and line is not None and line != ' ':
    arr = line.split('\t')
    interaction_shadow[int(arr[0])].append(int(arr[1]))
    line = f_shadow.readline()

# vectorization for shadow
member_bound = 0
random_bound = 0
for i in range(num_shadow):
    if i < num_shadow//2:
        # member
        label_shadow.append([1])
    else:
        # random
        label_shadow.append([0])
    temp_vector = torch.zeros(100)
    # the center of the ineractions
    len_shadow = len(interaction_shadow[i])
    for j in range(len_shadow):
        temp_vector = temp_vector + vectors1[interaction_shadow[i][j]]
    temp_vector = temp_vector / len_shadow
    temp_vector = temp_vector.numpy().tolist()
    # subtracted
    vector_shadow.append([])
    vector_shadow[i].append(temp_vector) # list
    temp_vector = torch.zeros(100)
    #the center of the recommends
    for j in range(topk):
        # temp_vector = temp_vector + ((topk-j)/5050)*vectors1[recommend_shadow[i][j]]
        temp_vector = temp_vector + (0.01)*vectors1[recommend_shadow[i][j]]
    temp_vector = temp_vector.numpy().tolist()
    # vector_shadow[i][0] = vector_shadow[i][0] + temp_vector # list
    for j in range(100):
        vector_shadow[i][0][j] = vector_shadow[i][0][j] - temp_vector[j]

# vectorization for target
for i in range(num_target):
    if i < num_target // 2:
        # member
        label_target.append([1])
    else:
        # random
        label_target.append([0])
    temp_vector = torch.zeros(100)
    # the center of the ineractions
    len_target = len(interaction_target[i])
    for j in range(len_target):
        temp_vector = temp_vector + vectors2[interaction_target[i][j]]
    temp_vector = temp_vector / len_target
    temp_vector = temp_vector.numpy().tolist()
    # subtracted
    vector_target.append([])
    vector_target[i].append(temp_vector) # lsit
    temp_vector = torch.zeros(100)
    #the center of the recommends
    for j in range(topk):
        # temp_vector = temp_vector + ((topk-j)/5050)*vectors2[recommend_target[i][j]]
        temp_vector = temp_vector + (0.01)*vectors2[recommend_target[i][j]]
    temp_vector = temp_vector.numpy().tolist()
    # vector_target[i][0] = vector_target[i][0] + temp_vector # list
    for j in range(100):
        vector_target[i][0][j] = vector_target[i][0][j] - temp_vector[j]


DataDir = "../AttackData/"
if not os.path.exists(DataDir):
    os.mkdir(DataDir)
ShadowDataset = open("../AttackData/trainForClassifier_MIAL.txt", 'w')
TargetDataset = open("../AttackData/testForClassifier_MIAL.txt", 'w')
for i in range(num_shadow):
    ShadowDataset.write(str(vector_shadow[i]) + '\t' + str(label_shadow[i]) + '\n')
    vector_shadow[i] = torch.Tensor(np.array(vector_shadow[i])) # train
    label_shadow[i] = torch.Tensor(np.array(label_shadow[i])).long() # train

for i in range(num_target):
    TargetDataset.write(str(vector_target[i]) + '\t' + str(label_target[i]) + '\n')
    vector_target[i] = torch.Tensor(np.array(vector_target[i])) # test
    label_target[i] = torch.Tensor(np.array(label_target[i])).long() # test

mlp = MLP()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(mlp.parameters(), lr=0.01, momentum=0.9)

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

for x in range(20):
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
        if i < num_shadow//2:
            # 1
            if int(pred)==1:
                train_acc += 1
        else:
            # 0
            if int(pred)==0:
                train_acc += 1

        # print(x, ":", AccuarcyCompute(outputs, labels))
    losses.append(train_loss / num_shadow)
    acces.append(train_acc / num_shadow)
    print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}'.format(x, train_loss / (num_shadow), train_acc / (num_shadow)))

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
    if i < num_target//2:
        # 1
        if int(pred)==1:
            acc_ans += 1
            TruePositive = TruePositive + 1
        else:
            FalseNegative = FalseNegative + 1
    else:
        # 0
        if int(pred)==0:
            acc_ans += 1
            TrueNegative = TrueNegative + 1
        else:
            FalsePositive = FalsePositive + 1
print("accuarcy: ")
print((acc_ans / num_target))
print("precsion: ")
print((TruePositive / (TruePositive+FalsePositive)))
print("recall: ")
print((TruePositive / (TruePositive+FalseNegative)))

TPRate = TruePositive / (TruePositive+FalseNegative)
FPRate = FalsePositive / (FalsePositive+TrueNegative)
area = 0.5*TPRate*FPRate+0.5*(TPRate+1)*(1-FPRate)
print("AUC: ")
print(area)
