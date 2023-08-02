import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm

from model import AlexNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), ".."))  # 拿到 mycode
    # print(data_root)
    image_path = os.path.join(data_root, "data_set", "flower_data")  # 拿到 mycode/data_set/flower_data
    # print(image_path)
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)  # 断言警告
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])  # 加载数据集
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())

    # 写入json文件
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))  # 在windows下只有主线程，下面的num_workers=0，Linux下可以有多个线程

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)  # num_workers=0

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])  # 加载测试集
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,  # 显示图片时修改
                                                  num_workers=nw)  # num_workers=0

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    """显示图片的代码"""
    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.next()
    #
    # def imshow(img):
    #     img = img / 2 + 0.5  # unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    #
    # print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
    # imshow(utils.make_grid(test_image))

    net = AlexNet(num_classes=5, init_weights=True)

    net.to(device)
    loss_function = nn.CrossEntropyLoss()  # 交叉熵损失函数
    # pata = list(net.parameters())
    optimizer = optim.Adam(net.parameters(), lr=0.0002)  # Adam优化器

    epochs = 10  # 训练10个epochs
    save_path = './AlexNetgpu10.pth'  # 保存训练结果的路径
    best_acc = 0.0  # 保存准确率最高的模型
    train_steps = len(train_loader)

    for epoch in range(epochs):
        # train 训练
        net.train()  # 管理dropout，训练过程中会调用dropout方法
        running_loss = 0.0  # 训练过程中的平均损失

        # tqdm是一个用于在循环中添加进度条的Python库
        train_bar = tqdm(train_loader, file=sys.stdout)

        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()  # 清空梯度信息
            outputs = net(images.to(device))  # 正向传播
            loss = loss_function(outputs, labels.to(device))  # 得到损失
            loss.backward()  # 反向传播到每个节点中
            optimizer.step()  # 更新每个节点的参数

            # print statistics
            running_loss += loss.item()
            # 打印训练进度
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate 验证
        net.eval()  # 管理dropout，验证过程中不会调用dropout方法
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))  # 将图片放到网络中进行正向传播，得到输出
                predict_y = torch.max(outputs, dim=1)[1]  # 求得输出的最大值作为预测
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()  # acc累计验证集中预测正确的样本数

        val_accurate = acc / val_num  # 将正确的个数/总数=正确率
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:  # 当前准确率大于历史最优
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)  # 保存权重

    print('Finished Training')


"""
电脑训练5轮的输出：
using cpu device.
Using 8 dataloader workers every process
using 3306 images for training, 364 images for validation.
train epoch[1/5] loss:1.036: 100%|██████████| 104/104 [00:55<00:00,  1.88it/s]
100%|██████████| 12/12 [00:19<00:00,  1.61s/it]
[epoch 1] train_loss: 1.373  val_accuracy: 0.456
train epoch[2/5] loss:1.154: 100%|██████████| 104/104 [00:54<00:00,  1.92it/s]
100%|██████████| 12/12 [00:18<00:00,  1.58s/it]
[epoch 2] train_loss: 1.213  val_accuracy: 0.497
train epoch[3/5] loss:1.168: 100%|██████████| 104/104 [00:54<00:00,  1.92it/s]
100%|██████████| 12/12 [00:18<00:00,  1.56s/it]
[epoch 3] train_loss: 1.135  val_accuracy: 0.558
train epoch[4/5] loss:0.911: 100%|██████████| 104/104 [00:52<00:00,  1.99it/s]
100%|██████████| 12/12 [00:19<00:00,  1.61s/it]
[epoch 4] train_loss: 1.050  val_accuracy: 0.618
train epoch[5/5] loss:0.752: 100%|██████████| 104/104 [00:53<00:00,  1.93it/s]
100%|██████████| 12/12 [00:18<00:00,  1.57s/it]
[epoch 5] train_loss: 0.982  val_accuracy: 0.629
Finished Training
"""
"""
using cuda:0 device.
Using 8 dataloader workers every process
using 3306 images for training, 364 images for validation.
train epoch[1/5] loss:0.987: 100%|██████████| 104/104 [00:45<00:00,  2.31it/s]
100%|██████████| 12/12 [00:21<00:00,  1.81s/it]
[epoch 1] train_loss: 1.338  val_accuracy: 0.475
train epoch[2/5] loss:1.285: 100%|██████████| 104/104 [00:34<00:00,  3.00it/s]
100%|██████████| 12/12 [00:22<00:00,  1.90s/it]
[epoch 2] train_loss: 1.173  val_accuracy: 0.464
train epoch[3/5] loss:1.393: 100%|██████████| 104/104 [00:31<00:00,  3.26it/s]
100%|██████████| 12/12 [00:22<00:00,  1.84s/it]
[epoch 3] train_loss: 1.136  val_accuracy: 0.525
train epoch[4/5] loss:1.158: 100%|██████████| 104/104 [00:32<00:00,  3.23it/s]
100%|██████████| 12/12 [00:21<00:00,  1.83s/it]
[epoch 4] train_loss: 1.006  val_accuracy: 0.629
train epoch[5/5] loss:1.180: 100%|██████████| 104/104 [00:31<00:00,  3.27it/s]
100%|██████████| 12/12 [00:21<00:00,  1.83s/it]
[epoch 5] train_loss: 0.994  val_accuracy: 0.602
Finished Training
"""
"""
using cuda:0 device.
Using 8 dataloader workers every process
using 3306 images for training, 364 images for validation.
train epoch[1/10] loss:1.009: 100%|██████████| 104/104 [00:24<00:00,  4.21it/s]
100%|██████████| 12/12 [00:16<00:00,  1.41s/it]
[epoch 1] train_loss: 1.370  val_accuracy: 0.451
train epoch[2/10] loss:1.099: 100%|██████████| 104/104 [00:20<00:00,  5.06it/s]
100%|██████████| 12/12 [00:17<00:00,  1.46s/it]
[epoch 2] train_loss: 1.194  val_accuracy: 0.451
train epoch[3/10] loss:1.038: 100%|██████████| 104/104 [00:20<00:00,  5.12it/s]
100%|██████████| 12/12 [00:17<00:00,  1.50s/it]
[epoch 3] train_loss: 1.131  val_accuracy: 0.555
train epoch[4/10] loss:0.624: 100%|██████████| 104/104 [00:20<00:00,  4.99it/s]
100%|██████████| 12/12 [00:17<00:00,  1.45s/it]
[epoch 4] train_loss: 1.020  val_accuracy: 0.613
train epoch[5/10] loss:1.216: 100%|██████████| 104/104 [00:21<00:00,  4.86it/s]
100%|██████████| 12/12 [00:17<00:00,  1.46s/it]
[epoch 5] train_loss: 0.990  val_accuracy: 0.654
train epoch[6/10] loss:1.412: 100%|██████████| 104/104 [00:19<00:00,  5.24it/s]
100%|██████████| 12/12 [00:16<00:00,  1.40s/it]
[epoch 6] train_loss: 0.923  val_accuracy: 0.698
train epoch[7/10] loss:0.565: 100%|██████████| 104/104 [00:19<00:00,  5.22it/s]
100%|██████████| 12/12 [00:17<00:00,  1.45s/it]
[epoch 7] train_loss: 0.882  val_accuracy: 0.684
train epoch[8/10] loss:0.666: 100%|██████████| 104/104 [00:20<00:00,  5.03it/s]
100%|██████████| 12/12 [00:17<00:00,  1.45s/it]
[epoch 8] train_loss: 0.866  val_accuracy: 0.654
train epoch[9/10] loss:0.973: 100%|██████████| 104/104 [00:21<00:00,  4.85it/s]
100%|██████████| 12/12 [00:17<00:00,  1.48s/it]
[epoch 9] train_loss: 0.842  val_accuracy: 0.731
train epoch[10/10] loss:0.420: 100%|██████████| 104/104 [00:22<00:00,  4.59it/s]
100%|██████████| 12/12 [00:18<00:00,  1.54s/it]
[epoch 10] train_loss: 0.793  val_accuracy: 0.668
Finished Training
"""

if __name__ == '__main__':
    main()
