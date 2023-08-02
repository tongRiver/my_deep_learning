## 训练的流程

使用PyTorch框架，主要涉及以下几个步骤：

1. 定义模型：首先，需要定义一个模型并初始化其参数。可以使用PyTorch提供的各种预定义模型，或者自定义模型。

2. 定义损失函数：为了使用优化器进行参数更新，需要定义一个损失函数来衡量模型输出与目标值之间的差距。常见的损失函数包括均方误差损失（MSE Loss）、交叉熵损失（Cross-Entropy Loss）等。

3. 定义优化器：选择合适的优化器，并传入模型的参数。PyTorch提供了多种常用的优化器，如SGD、Adam、RMSprop等。需要指定学习率和其他可选的超参数。

4. 进行训练迭代：在每次训练迭代中，按照以下步骤更新模型参数：
   - 将模型的梯度置零，以避免梯度累积影响更新；
   - 前向传播：将输入数据输入到模型中，得到模型的输出；
   - 计算损失：将模型的输出与目标值计算损失；
   - 反向传播：根据损失函数，计算损失对于模型参数的梯度；
   - 更新参数：利用优化器的`step()`方法根据梯度来更新模型参数。

以下是一个使用PyTorch框架优化器的简单示例代码：

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 定义模型
model = nn.Linear(in_features=10, out_features=1)

# 定义损失函数
criterion = nn.MSELoss()

# 定义优化器，传入模型参数和学习率
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练迭代
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(inputs)
    
    # 计算损失
    loss = criterion(outputs, targets)
    
    # 梯度置零
    optimizer.zero_grad()
    
    # 反向传播
    loss.backward()
    
    # 更新参数
    optimizer.step()
```

在每次训练迭代中，将输入数据传入模型进行前向传播，计算损失后进行反向传播并更新参数。通过调整优化器的超参数（如学习率），可以对模型的训练过程进行调整和优化。



## LeNet神经网络的搭建

>  搭建的视频：[2.1 pytorch官方demo(Lenet)](https://www.bilibili.com/video/BV187411T7Ye/?share_source=copy_web&vd_source=340f3fe423cd0f0f68fb72953f912437)

### 原理





### 神经网络



```python
"""model.py"""

import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))    # input(3, 32, 32) output(16, 28, 28)
        x = self.pool1(x)            # output(16, 14, 14)
        x = F.relu(self.conv2(x))    # output(32, 10, 10)
        x = self.pool2(x)            # output(32, 5, 5)
        x = x.view(-1, 32*5*5)       # output(32*5*5)
        x = F.relu(self.fc1(x))      # output(120)
        x = F.relu(self.fc2(x))      # output(84)
        x = self.fc3(x)              # output(10)
        return x


# import torch
# input1 = torch.rand([32, 3, 32, 32])
# model = LeNet()
# print(model)
# output = model(input1)

"""
LeNet(
  (conv1): Conv2d(3, 16, kernel_size=(5, 5), stride=(1, 1))
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1))
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fc1): Linear(in_features=800, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
"""
```



### 训练







### 预测

















## 资料

视频教学：[2.1 pytorch官方demo(Lenet)](https://www.bilibili.com/video/BV187411T7Ye/?share_source=copy_web&vd_source=340f3fe423cd0f0f68fb72953f912437)

官方教程：[Training a Classifier — PyTorch Tutorials 2.0.1+cu117 documentation](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

