import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from skimage import transform
from torchvision import transforms, models

model = models.resnet152(pretrained=True) # True表示加载预训练模型
# print(device)
# print(model)
# for layer in model.state_dict():
#     print(layer)

#
# class TheModelClass(nn.Module):
#     def __init__(self):
#         super(TheModelClass, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
#
#
# # initial model
# model1 = TheModelClass()
# optimizer = optim.SGD(model.parameters(),lr=0.001,momentum=0.9)

# for layer in model1.state_dict():
#     print(layer)

# print('\n',model.conv1.weight.size())
# print('\n',model.conv1.weight)

# conv1_weight_state = torch.load('/home/xlc/zjr/Resnet/data/model/ResNet_cub200/params50.pkl')['conv1.weight']

# print(conv1_weight_state.size())
# for i in model.parameters():
#     print(i)

image = io.imread('/home/xlc/zjr/IMFDB_final/ValidationData/00054.jpg')
plt.imshow(image)
plt.show()

image = transform.resize(image, (208, 208), mode='constant')
img = image * 255  # 将图片的取值范围改成（0~255）
img = img.astype(np.uint8)

plt.imshow(img)
plt.show()