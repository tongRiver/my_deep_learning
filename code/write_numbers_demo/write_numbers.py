import torch
import torch.nn as nn
import torch.optim as optim

from model import LeNet


# 创建LeNet实例
lenet = LeNet()


"""3. 定义损失函数和优化器："""
criterion = nn.CrossEntropyLoss()   # 使用交叉熵损失函数
optimizer = optim.SGD(lenet.parameters(), lr=0.001, momentum=0.9)   # 使用随机梯度下降（SGD）优化器


"""4. 加载数据集并进行预处理。这里以MNIST手写数字数据集为例："""
from torchvision import datasets, transforms

# 设置数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),   # 将图像转换为Tensor格式
    transforms.Normalize((0.5,), (0.5,))   # 标准化图像
])

# 加载训练集和测试集
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=False)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)


"""5. 进行模型训练和测试："""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lenet.to(device)   # 将模型移到对应设备（GPU或CPU）

num_epochs = 10

for epoch in range(num_epochs):
    lenet.train()   # 设置为训练模式，管理dropout，训练的时候神经元随机失活
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

    lenet.eval()   # 设置为评估模式，管理dropout，预测的时候神经元不能失活
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

print('Finished Training')

save_path = './Lenet3.pth'
torch.save(lenet.state_dict(), save_path)

"""
训练的输出：
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

看上去挺好的，但是为啥自己写的数据就不行。。。
"""
