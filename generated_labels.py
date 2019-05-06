# -*- encoding:utf-8 -*-
# @Author  : LeonRun
# @Size    : 
# @Time    : 2019/4/21 上午9:59
# @File    : generated_labels.py
# @Software: PyCharm
# note     :

import os

rootdir = './data/face/'

print("hello world")

dir_list = os.listdir(rootdir)

dir_name = ''

for i in range(len(dir_list)):
    path = os.path.join(rootdir, dir_list[i])
    if os.path.isdir(path):
        dir_name = dir_list[i]
        dir_list2 = os.listdir(path)
        for j in range(len(dir_list2)):
            path2 = os.path.join(path, dir_list2[j])
            if os.path.isdir(path2):
                # content = path2.lstrip('./data/') + ' ' + path2.split('/')[0]
                print("%s is still a dir" % path2)
                # with open('pest.data','a+') as f:
                #     content = path2.lstrip('./data/')+' '+path2.split('/')[0]
                    # f.writelines(path2.lstrip('./data/')+' '+path2.split('/')[0])
            else:
                # print("%s is a file" % path2)
                # content = path2.lstrip('./data/') + ' ' + dir_name
                with open('face_train.data','a+') as f:
                    content = path2.lstrip('./')+' '+str(i)+'\n'
                    f.writelines(content)
                # print(content)
    else:
        print(path + ' is files')

# with open('pest.txt','a') as f:
#
#     for i in range(len(dir_list)):



