# -*- encoding:utf-8 -*-
import csv
from _csv import writer

import pandas as pd
import numpy as np
import math

with open('B_predict_result.csv','r') as file:
    data = file.readlines()

title_id,titile_con = data[0].split(',')
data = data[1:]

row = []
row.append(title_id)
row.append(titile_con)
with open("predictions.csv",'w') as f:
    writer = csv.writer(f)
    writer.writerow(row)

content = []
id = []
for line in data:
    id.append(int(line.split(',')[0]))
    content.append(float(line.split(',')[1].rstrip('\n')))

temp = content.copy()
list.sort(temp)
print(temp[1999])
print(temp[0])
qujian = float(temp[1999]) - float(temp[0])
print(qujian)
for i in range(len(content)):
    row = []
    content[i] = 1.0*(float(content[i]) - float(temp[0]))/qujian
    print("id=%d,confidence=%f" % (id[i],content[i]))
    row.append(id[i])
    row.append(content[i])
    with open("predictions.csv",'a') as f:
        writer = csv.writer(f)
        writer.writerow(row)