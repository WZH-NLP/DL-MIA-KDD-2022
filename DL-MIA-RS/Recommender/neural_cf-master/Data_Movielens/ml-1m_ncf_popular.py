
frdir = "/home/zhangminxing/code/ml-1m/movielens_target"
fr = open(frdir, 'r')

fwdir_shadow = "/home/zhangminxing/code/ml-1m/movielens_topk_shadow_popular_ncf"
fwdir_target = "/home/zhangminxing/code/ml-1m/movielens_topk_target_popular_ncf"
fw_shadow = open(fwdir_shadow, 'w')
fw_target = open(fwdir_target, 'w')

num_users1 = 3938 # 393800
topk = 100       # 1969 196900

num_shadow = num_users1//2
num_target = num_users1 - num_users1//2

for i in range(num_users1*topk):
    if i < topk*(num_users1//2):
        fw_shadow.write(str(i//topk) + '\t' + str(ranklist[i]) + '\t1\t\n' )
    else:
        fw_target.write(str((i//topk)-(num_users1//2)) + '\t' + str(ranklist[i]) + '\t1\t\n')

num_users2 = 3938
num_items = 3706
item_list = []
for i in range(num_items):
    item_list.append(0)
line = fr.readline()
while line != '' and line is not None and line != ' ':
    arr = line.split('\t')
    item_list[int(arr[1])] = item_list[int(arr[1])] + 1
    line = fr.readline()
sorted_items = sorted(sorted(enumerate(item_list), key=lambda x: x[1]))
idx = [i[0] for i in sorted_items]
idx.reverse()
idx = idx[0:topk]
for i in range(num_users2//2):
    for item in idx:
        fw_shadow.write(str(i+num_shadow) + '\t' + str(item) + '\t1\t\n')
for i in range(num_users2 - num_users2//2):
    for item in idx:
        fw_target.write(str(i+num_target) + '\t' + str(item) + '\t1\t\n')

fr.close()
fw_shadow.close()
fw_target.close()
