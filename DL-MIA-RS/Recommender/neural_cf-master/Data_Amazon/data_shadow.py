import random

frdir = "/home/zhangminxing/code/amazon/amazon_music_shadow"
fr = open(frdir, 'r')

fwdir = "/home/zhangminxing/code/neural_cf-master/Data_Amazon/ratings.dat"
fw = open(fwdir, 'w')

sep = "::"

count_ind = 0

line = fr.readline()
while line != ' ' and line is not None and line != '':
    arr = line.split('\t')
    fw.write(arr[0] + "::" + arr[1] + "::3::" + str(random.randint(1000000, 9999999999)) + '\n')
    count_ind = count_ind + 1
    line = fr.readline()

fr.close()
fw.close()
