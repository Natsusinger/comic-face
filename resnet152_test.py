#coding:utf-8
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
from torchvision import transforms, models
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
import PIL

# print(models.resnet50(pretrained=False))

# root数据集路径
root = '/home/xlc/zjr/Resnet/'
# default_loader返回类型：<PIL.Image.Image image mode=RGB size=widthxheight at 0x7F3E275CD550>
def default_loader(path):
    return PIL.Image.open(path).convert('RGB')
class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
	# txt文本里的每一行格式 Ampullariagigas/Ampullariagigas_30.jpg 0 .0表示类别
        nowfile = open(txt, 'r')
        imgs = []
        for line in nowfile:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            words[0]=root+'data/'+words[0]
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
    def __getitem__(self, index):
        imgpath, label = self.imgs[index]
        img = self.loader(imgpath)
 
        if self.transform is not None:
            img = self.transform(img)
        return img,label
    
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
def test(args, model, device, test_loader, loss_func):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_func(output, target).sum().item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    log_loss.append(test_loss)
    is_acc_change = False
    nowacc = 100. * correct / len(test_loader.dataset)
    if len(log_acc) == 0 or log_acc[len(log_acc) - 1] < nowacc:
        log_acc.append(nowacc)
        is_acc_change = True
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return is_acc_change

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
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
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
    # train_data = MyDataset(txt=root+'train.data', transform=transform)
    test_data = MyDataset(txt=root+'test.data', transform=test_transform)
    # train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(dataset=test_data, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    
    model = models.resnet152(pretrained=True) # True表示加载预训练模型
    print(device)
    print(model)
    model.fc = nn.Linear(model.fc.in_features, 1000) # 线性模型

    model.avgpool = nn.AvgPool2d(kernel_size=14, stride=1, padding=0)
    print(model)
    print("model have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    # model.load_state_dict(torch.load('/home/sysu/syudy_net/model_resnet/model4/res152_cub200_params_acc=80.6523990335_epoch=45.pkl'))
    model.load_state_dict(
        torch.load('/home/xlc/zjr/Resnet/data/model/model_resnet152_1/model11' + '/res152_cub200_params_acc=59_epoch=50.pkl'))
    model.to(device)

    loss_func = torch.nn.CrossEntropyLoss()
    base_lr = args.lr
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=args.momentum, weight_decay=0.00005)
    is_acc_change = test(args, model, device, test_loader, loss_func)
    # if is_acc_change:
    #     torch.save(model.state_dict(),
    #                '/home/xlc/zjr/Resnet/data/model/model_resnet152_1/model11' + '/res152_cub200_params_acc=59_epoch=50.pkl')
    # for epoch in range(1, args.epochs + 1):
    #     start = time.time()
    #     # train(args, model, device, train_loader, optimizer, loss_func, epoch)
	# #边训练边测试
    #     is_acc_change = test(args, model, device, test_loader, loss_func)
    #     end = time.time()
    #     print("循环运行时间:%.2f秒" % (end - start))
    #     adjust_learning_rate(optimizer, epoch, args)
    #     # if epoch%10==0:
    #         # torch.save(model.state_dict(), '/home/xlc/zjr/Resnet/data/model/ResNet_cub200'+'/params' + str(epoch) + '.pkl')
    #     if is_acc_change:
    #         torch.save(model.state_dict(), '/home/xlc/zjr/Resnet/data/model/model_resnet152_1/model11' + '/res152_cub200_params_acc=' + str(log_acc[len(log_acc) - 1]) + '_epoch=' + str(epoch) + '.pkl')
    #     # adjust_learning_rate(optimizer, epoch, args)
    #     # if epoch%10==0:
    #     #     torch.save(model.state_dict(), '/home/yaoliyao/PyTorch/model/ResNet_cub200'+'/params' + str(epoch) + '.pkl')
    #     # if is_acc_change:
    #     #     torch.save(model.state_dict(), '/home/sysu/syudy_net/model_resnet/model11' + '/res152_cub200_params_acc=' + str(log_acc[len(log_acc) - 1]) + '_epoch=' + str(epoch) + '.pkl')



if __name__ == '__main__':
    main()
    # model = models.resnet50(pretrained=True)
    # model.fc = nn.Linear(model.fc.in_features, 200)
    # print(model)
