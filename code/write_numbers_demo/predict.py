"""
ai第一次写的
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import LeNet


def main():
    transform = transforms.Compose(
        [transforms.Resize((28, 28)),
         transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])


    net = LeNet()
    net.load_state_dict(torch.load('Lenet.pth'))

    im = Image.open('sex.jpg').convert('L')
    im = transform(im)  # [C, H, W]
    im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]

    # plt.imshow(im.squeeze().numpy(), cmap='gray')
    # plt.title('Input Image')
    # plt.axis('off')
    # plt.show()

    with torch.no_grad():
        outputs = net(im)
        predict = torch.softmax(outputs, dim=1)
    print(f'outputs: {outputs}')
    print(f'predict: {predict}')


if __name__ == '__main__':
    main()
