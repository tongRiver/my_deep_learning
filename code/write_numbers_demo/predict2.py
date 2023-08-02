"""
ai第二次写的
"""

import torch
from torchvision import transforms
from PIL import Image
from model import LeNet

# 加载模型
model = LeNet()  # 这里需要用到LeNet模型的定义
model.load_state_dict(torch.load('Lenet2.pth'))  # 加载已训练好的模型参数
model.eval() # 管理dropout，预测的时候神经元不能失活

# 图像预处理和转换
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 打开并预处理手写数字图像
image = Image.open('five.jpg').convert('L')  # 打开图像并转为灰度图像
image = transform(image)  # 转换为Tensor并标准化

# 添加批次维度，因为模型输入需要为四维张量
image = image.unsqueeze(0)

# 使用模型进行预测
with torch.no_grad():
    output = model(image)

# 获取预测结果
_, predicted = torch.max(output, 1)
prediction = predicted.item()
print("预测的数字是:", prediction)
