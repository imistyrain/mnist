import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet4(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

class ConvNet4_32(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*64, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 24, 5, 1)
#         self.conv2 = nn.Conv2d(24, 32, 3, 1)
#         self.fc1 = nn.Linear(800, 256)
#         self.fc2 = nn.Linear(256, 10)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.max_pool2d(x, 2)
#         x = F.relu(x)
        
#         x = self.conv2(x)
#         x = F.max_pool2d(x, 2)
#         x = F.relu(x)
        
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x)
#         output = F.log_softmax(x, dim=1)
#         return output

class ConvNet(nn.Module):
    def __init__(self,embedding_size=2):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 24, 5, 1)
        self.conv2 = nn.Conv2d(24, 32, 3, 1)
        self.fc1 = nn.Linear(800, 256)
        self.fc2 = nn.Linear(256, embedding_size)
        self.fc3 = nn.Linear(embedding_size, 10,bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output

if __name__=="__main__":
    model = ConvNet()
    print(model)