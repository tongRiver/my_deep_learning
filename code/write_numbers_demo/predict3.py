"""
chatgpt写的

用的Lenet4.pth，结果也不太行，但是至少每张图识别出来的结果不一样了
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from model import LeNet

def load_model(model_path):
    lenet = LeNet()
    lenet.load_state_dict(torch.load(model_path))
    lenet.eval()  # 管理dropout，预测的时候神经元不能失活
    return lenet

def resize_image(image, size):
    return image.resize(size)

def predict_single_image(image, model):
    transform = transforms.Compose([
        # transforms.Resize((28, 28)),  # Resize image to 28x28
        transforms.Grayscale(),  # Convert image to grayscale (1 channel)
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image = resize_image(image, (28, 28))
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = nn.functional.softmax(output, dim=1)
        _, predicted_class = torch.max(probabilities, 1)

    return predicted_class.item(), probabilities.squeeze().cpu().numpy()

if __name__ == '__main__':
    model_path = 'Lenet4.pth'
    image_path = '4.jpg'  # Replace with the path to your image

    lenet = load_model(model_path)
    image = Image.open(image_path)
    predicted_class, probabilities = predict_single_image(image, lenet)

    print(f"Predicted class: {predicted_class}")
    print("Probabilities:")
    for i, prob in enumerate(probabilities):
        print(f"Class {i}: {prob:.4f}")
