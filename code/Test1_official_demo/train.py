import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt



transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

"""50000张训练图片"""
# 第一次使用时要将download设置为True才会自动去下载数据集
train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                         download=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=36,
                                           shuffle=True, num_workers=0)

"""10000张验证图片"""
# 第一次使用时要将download设置为True才会自动去下载数据集
val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=10000,
                                         shuffle=False, num_workers=0)
val_data_iter = iter(val_loader)
val_image, val_label = next(val_data_iter)  # 将10000张测试样本一次性提取出来

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# """查看图片"""
# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
# # 打印标签
# print(' '.join(f'{classes[val_label[j]]:5s}' for j in range(4)))
# # 显示图片
# imshow(torchvision.utils.make_grid(val_image))


net = LeNet()                                         # 定义网络
loss_function = nn.CrossEntropyLoss()                 # 定义损失函数
optimizer = optim.Adam(net.parameters(), lr=0.001)    # 定义优化器，parameters()表示将LeNet所有可训练的参数都进行训练，lr是学习率

for epoch in range(7):  # 将训练集迭代多少轮

    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):  # enumerate不仅会返回data，还会返回当前步数（从0开始）
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # 将参数的历史梯度清零
        optimizer.zero_grad()

        outputs = net(inputs)                         # 正向传播
        loss = loss_function(outputs, labels)         # 计算损失
        loss.backward()                               # 反向传播
        optimizer.step()                              # 优化器更新参数

        # print statistics
        running_loss += loss.item()
        if step % 500 == 499:    # print every 500 mini-batches
            with torch.no_grad():  # 验证测试阶段不用计算梯度
                outputs = net(val_image)  # [batch, 10]，正向传播
                predict_y = torch.max(outputs, dim=1)[1]
                accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)

                print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                      (epoch + 1, step + 1, running_loss / 500, accuracy))
                running_loss = 0.0

print('Finished Training')

save_path = './Lenet7.pth'
torch.save(net.state_dict(), save_path)  # [7,  1000] train_loss: 0.735  test_accuracy: 0.689


