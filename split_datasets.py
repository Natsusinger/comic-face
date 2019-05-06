# -*- encoding:utf-8 -*-
# @Author  : LuisYang
# @Size    : 
# @Time    : 2019/4/21 下午5:18
# @File    : split_datasets.py
# @Software: PyCharm
# note     : 随机切分数据集
import random

data = []
result = []
with open('face_train.data','r') as f:
    data = f.readlines()


result = [i for i in range(len(data))]

train = []
test = []

print(result)
random.shuffle(result)
print(result)

tmp = int(len(result) * 0.2)
print(tmp)

for j in range(len(result)):
    if j < tmp:
        test.append(data[result[j]])
    else:
        train.append(data[result[j]])

print(train)
print(test)

with open('train.data','w') as fw:
    fw.writelines(train)
    
with open('test.data','w') as fw:
    fw.writelines(test)

