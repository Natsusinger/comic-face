#coding:utf-8
from __future__ import print_function
import argparse
import csv
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity

from torchvision import transforms, models
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
import PIL

# print(models.resnet50(pretrained=False))

# root数据集路径
root = './data/'
# default_loader返回类型：<PIL.Image.Image image mode=RGB size=widthxheight at 0x7F3E275CD550>
def default_loader(path):
    return PIL.Image.open(path).convert('RGB')
# class MyDataset(Dataset):
#     def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
# 	# txt文本里的每一行格式 Ampullariagigas/Ampullariagigas_30.jpg 0 。0表示类别
#         nowfile = open(txt, 'r')
#         imgs = []
#         for line in nowfile:
#             line = line.strip('\n')
#             line = line.rstrip()
#             words = line.split()
#             words[0] = root+words[0]
#             # imgs.append((words[0],int(words[1])))
#             imgs.append((words[0]))
#         self.imgs = imgs
#         self.transform = transform
#         self.target_transform = target_transform
#         self.loader = loader
#     def __getitem__(self, index):
#         #imgpath, label = self.imgs[index]
#         imgpath = self.imgs[index]
#         img = self.loader(imgpath)
#
#         if self.transform is not None:
#             img = self.transform(img)
# #        return img,label
#         return img
#
#     def __len__(self):
#         return len(self.imgs)

class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        # txt文本里的每一行格式 Ampullariagigas/Ampullariagigas_30.jpg 0 。0表示类别
        # nowfile = open(txt, 'r')
        imgs = []

        imgs.append(txt.strip('\n'))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        # imgpath, label = self.imgs[index]
        imgpath = self.imgs[index]
        img = self.loader(imgpath)

        if self.transform is not None:
            img = self.transform(img)
        #        return img,label
        return img

    def __len__(self):
        return len(self.imgs)

def train(args, model, device, train_loader, optimizer, loss_func, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

log_loss = []
log_acc = []
# cls = [0,1,2,3]
def test(args, model, device, test_loader, loss_func):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data)
            # test_loss += loss_func(output, target).sum().item() # sum up batch loss
            # pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            #correct += pred.eq(target.view_as(pred)).sum().item()
            # print(pred.item())
            print(output.shape)
    return output

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** min((epoch // args.step_epoch),1))
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--step_epoch', default=30, type=int, metavar='N',
                        help='decend the lr in epoch number')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=8, metavar='N',
                        help='input batch size for testing (default: 32)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    print(torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")


    rSize = 512
    cSize = 448
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    transform = transforms.Compose([
        transforms.Resize((rSize, rSize)),
        transforms.RandomCrop(cSize),  # 对图像大小统一
        transforms.RandomHorizontalFlip(),  # 图像翻转
        # transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[  # 图像归一化
            0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        # transforms.Resize((cSize, cSize)),
        transforms.Resize((rSize, rSize)),
        transforms.CenterCrop(cSize),  # 对图像大小统一
        # transforms.RandomHorizontalFlip(),  # 图像翻转
        # transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[  # 图像归一化
            0.229, 0.224, 0.225])
    ])
    #train_data = MyDataset(txt=root+'pests.data',transform=transform)
    # test_data = MyDataset(txt=root+'aaa.data',transform=test_transform)
    #train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
    # test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False, **kwargs)

    model = models.resnet152(pretrained=True) # True表示加载预训练模型
    print(device)
    print(model)
    model.fc = nn.Linear(model.fc.in_features, 1000) # 线性模型

    model.avgpool = nn.AvgPool2d(kernel_size=14, stride=1, padding=0)
    print(model)
    print("model have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    # model.load_state_dict(torch.load('/home/leonrun/model/restnet/pests_result/model_resnet152_1/model11/res152_cub200_params_acc=100.0_epoch=18.pkl'))
    model.load_state_dict(torch.load('/home/xlc/zjr/Resnet/data/model/model_resnet152_1/model11/res152_cub200_params_acc=59_epoch=50.pkl'))
    model.to(device)

    loss_func = torch.nn.CrossEntropyLoss()
    base_lr = args.lr
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=args.momentum, weight_decay=0.00005)

    # test_data = MyDataset(txt=root + 'aaa.data', transform=test_transform)
    # # train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
    # test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False, **kwargs)
    # is_acc_change = test(args, model, device, test_loader, loss_func)

    result = []

    with open('./data/face_testB/list.csv', 'r') as file:
        data = file.readlines()
        data = data[1:]
        count_id = 1
        for i in range(len(data)):
            row = []
            result = []
            id, path1, path2 = data[i].split(',')
            print(path1.strip('\n'))
            print(path2.strip('\n'))
            test_data = MyDataset(txt=root +r'face_testB/images/'+ path1, transform=test_transform)
            # train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
            test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False, **kwargs)
            temp1 = test(args, model, device, test_loader, loss_func)
            temp1 = temp1.data.cpu().numpy().tolist()
            result.append(temp1[0])
            test_data = MyDataset(txt=root + r'face_testB/images/' + path2, transform=test_transform)
            # train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
            test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False, **kwargs)
            temp2 = test(args, model, device, test_loader, loss_func)
            temp2 = temp2.data.cpu().numpy().tolist()
            result.append(temp2[0])
            res = cosine_similarity(result)[0][1]
            print(res)
            row.append(count_id)
            row.append(res)

            with open('B_predict_result.csv', 'a') as f1:
                writer = csv.writer(f1)
                writer.writerow(row)
            count_id = count_id + 1
            # result.append(temp)
            # temp_result = cosine_similarity(result)
            # print("sklearn cos = %f" % temp_result[0][1])
            # cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            # res = cos(temp1.data.cpu(), temp2.data.cpu())
            # print(res)
    # a = result[0].tolist()
    # b = result[1].tolist()
    # fi = []
    # fi.append(a[0])
    # fi.append(b[0])
    # print(type(a))
    # print(len(a[0]))
    # # print(a[0])
    # # print(a.size)
    # t = cosine_similarity(fi)[0][1]
    # print(t)

    # print(len(result))

if __name__ == '__main__':
    row = []
    row.append('group_id')
    row.append('confidence')
    with open('B_predict_result.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(row)
    main()
    # model = models.resnet50(pretrained=True)
    # model.fc = nn.Linear(model.fc.in_features, 200)
    # print(model)
