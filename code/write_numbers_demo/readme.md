## 项目说明

这个项目完全是ai写的，使用PyTorch框架搭建了LeNet神经网络，在MNIST数据集上训练




















## 详细流程

很高兴您选择使用PyTorch框架和LeNet神经网络。下面是详细的步骤：

1. 导入所需的库和模块：

```python
import torch
import torch.nn as nn
import torch.optim as optim
```

2. 定义LeNet神经网络模型。LeNet由卷积层、池化层和全连接层组成：

```python
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)   # 输入通道数为1，输出通道数为6，卷积核大小为5x5
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)   # 最大池化层，核大小为2x2，步长为2
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16*4*4, 120)   # 全连接层，输入大小为16x4x4，输出大小为120
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)   # 输出大小为10，对应10个类别

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(-1, 16*4*4)   # 展开为一维向量
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        return x

# 创建LeNet实例
lenet = LeNet()
```

3. 定义损失函数和优化器：

```python
criterion = nn.CrossEntropyLoss()   # 使用交叉熵损失函数
optimizer = optim.SGD(lenet.parameters(), lr=0.001, momentum=0.9)   # 使用随机梯度下降（SGD）优化器
```

4. 加载数据集并进行预处理。这里以MNIST手写数字数据集为例：

```python
from torchvision import datasets, transforms

# 设置数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),   # 将图像转换为Tensor格式
    transforms.Normalize((0.1307,), (0.3081,))   # 标准化图像
])

# 加载训练集和测试集
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
```

5. 进行模型训练和测试：

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lenet.to(device)   # 将模型移到对应设备（GPU或CPU）

num_epochs = 10

for epoch in range(num_epochs):
    lenet.train()   # 设置为训练模式
    train_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()   # 梯度清零

        outputs = lenet(images)   # 前向传播
        loss = criterion(outputs, labels)   # 计算损失
        loss.backward()   # 反向传播
        optimizer.step()   # 更新参数

        train_loss += loss.item() * images.size(0)   # 累计损失

    train_loss /= len(train_loader.dataset)

    lenet.eval()   # 设置为评估模式
    test_loss = 0.0
    correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = lenet(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")
```

这就是使用PyTorch框架和LeNet神经网络的详细步骤。您可以根据实际需求对其进行调整和修改。希望对您有所帮助！如果有任何问题，请随时提问。





## 详细解答

train_loss /= len(train_loader.dataset)的作用是什么呢？



`train_loss /= len(train_loader.dataset)` 是一个针对训练损失的计算和调整操作。在这个表达式中，`train_loss` 是用来存储累积的训练损失值的变量，`len(train_loader.dataset)` 表示训练数据集的样本数量。

`train_loss /= len(train_loader.dataset)` 的作用是将累积的训练损失值除以训练数据集的样本数量，从而得到平均训练损失。这一操作可以帮助我们更好地理解和比较模型在不同尺度上的损失情况，使得训练过程更加可解释和可比较。





## 存在的问题

问题一

当我训练时，准确率看上去很好：
Epoch [1/10], Train Loss: 2.0311, Test Loss: 0.6145, Accuracy: 0.8355
Epoch [2/10], Train Loss: 0.3167, Test Loss: 0.1972, Accuracy: 0.9395
Epoch [3/10], Train Loss: 0.1589, Test Loss: 0.1250, Accuracy: 0.9607
Epoch [4/10], Train Loss: 0.1171, Test Loss: 0.0908, Accuracy: 0.9704
Epoch [5/10], Train Loss: 0.0954, Test Loss: 0.0798, Accuracy: 0.9737
Epoch [6/10], Train Loss: 0.0826, Test Loss: 0.0676, Accuracy: 0.9782
Epoch [7/10], Train Loss: 0.0742, Test Loss: 0.0673, Accuracy: 0.9789
Epoch [8/10], Train Loss: 0.0667, Test Loss: 0.0592, Accuracy: 0.9798
Epoch [9/10], Train Loss: 0.0619, Test Loss: 0.0553, Accuracy: 0.9824
Epoch [10/10], Train Loss: 0.0571, Test Loss: 0.0527, Accuracy: 0.9834
Finished Training
可是在预测时，我换了不同的图片，可是每次的结果都是6，class 6的概率总是最高的：
Predicted class: 6
Probabilities:
Class 0: 0.0007
Class 1: 0.0000
Class 2: 0.0000
Class 3: 0.0000
Class 4: 0.0006
Class 5: 0.0833
Class 6: 0.7781
Class 7: 0.0000
Class 8: 0.1371
Class 9: 0.0001
你知道问题出在哪里吗？



现在问题一好像解决了，但是用手写的图像识别的结果仍然很不好，大概只能对一两张。





