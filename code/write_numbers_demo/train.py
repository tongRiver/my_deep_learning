"""
chatgpt给的训练过程
"""

import torch
import torch.nn as nn
import torch.optim as optim
from model import LeNet
from torchvision import datasets, transforms

lenet = LeNet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(lenet.parameters(), lr=0.001, momentum=0.9)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=False)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lenet.to(device)

num_epochs = 10

for epoch in range(num_epochs):
    lenet.train()
    train_loss = 0.0
    correct = 0  # Initialize correct predictions counter

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = lenet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

        # Calculate correct predictions
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()

    train_loss /= len(train_loader.dataset)

    lenet.eval()
    test_loss = 0.0
    correct_test = 0  # Initialize correct predictions counter for test set

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = lenet(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            correct_test += (predicted == labels).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct_test / len(test_dataset)  # Calculate accuracy using the test set size

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")

print('Finished Training')

save_path = 'Lenet4.pth'
torch.save(lenet.state_dict(), save_path)

"""
Epoch [1/10], Train Loss: 1.7271, Test Loss: 0.4261, Accuracy: 0.8719
Epoch [2/10], Train Loss: 0.3034, Test Loss: 0.2000, Accuracy: 0.9386
Epoch [3/10], Train Loss: 0.1801, Test Loss: 0.1390, Accuracy: 0.9539
Epoch [4/10], Train Loss: 0.1319, Test Loss: 0.1036, Accuracy: 0.9679
Epoch [5/10], Train Loss: 0.1077, Test Loss: 0.0857, Accuracy: 0.9735
Epoch [6/10], Train Loss: 0.0921, Test Loss: 0.0823, Accuracy: 0.9720
Epoch [7/10], Train Loss: 0.0825, Test Loss: 0.0713, Accuracy: 0.9758
Epoch [8/10], Train Loss: 0.0741, Test Loss: 0.0721, Accuracy: 0.9767
Epoch [9/10], Train Loss: 0.0680, Test Loss: 0.0576, Accuracy: 0.9819
Epoch [10/10], Train Loss: 0.0632, Test Loss: 0.0540, Accuracy: 0.9813
Finished Training
"""
