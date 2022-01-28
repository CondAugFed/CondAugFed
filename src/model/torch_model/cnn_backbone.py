"""
Taken from https://github.com/med-air/FedBN/blob/master/nets/models.py
for the reproducibility and the fair comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

### small size of CNN for CIFAR-10, SVHN dataset
class SimpleCNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class MNIST2NN(nn.Module):
    """
    2NN model that is used for FedAvg paper.
    """
    def __init__(self, num_classes=10):
        super(MNIST2NN, self).__init__()
        self.fc1 = nn.Linear(784, 200, bias=True)
        self.fc2 = nn.Linear(200, 200, bias=True)
        self.fc3 = nn.Linear(200, num_classes, bias=True)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MNISTCNN32(nn.Module):
    """
    CNN model that is used for FedAvg paper.
    """
    def __init__(self, num_classes=10):
        super(MNISTCNN32, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, bias=True, padding=2) # padding = 'same'
        self.conv2 = nn.Conv2d(32,64, 5, bias=True, padding=2) # padding = 'same'
        
        self.fc1 = nn.Linear(8*8*64, 512, bias=True)
        self.fc2 = nn.Linear(512, num_classes, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)

        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class MNISTCNN(nn.Module):
    """
    CNN model that is used for FedAvg paper.
    """
    def __init__(self, num_classes=10):
        super(MNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, bias=True, padding=2) # padding = 'same'
        self.conv2 = nn.Conv2d(32,64, 5, bias=True, padding=2) # padding = 'same'
        
        self.fc1 = nn.Linear(7*7*64, 512, bias=True)
        self.fc2 = nn.Linear(512, num_classes, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)

        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def get_feature(self, x, GP, step, noise_to='input'):
        if noise_to == 'input':
            x = GP.add_noise(x, step)
        
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        
        if noise_to == 'latent':
            x = GP.add_noise(x, step)
        x = x.view(x.shape[0], -1)

        return x

class DigitModel(nn.Module):
    """
    Model for benchmark experiment on Digits. 
    """
    def __init__(self, num_classes=10, **kwargs):
        super(DigitModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 5, 1, 2)
        self.bn3 = nn.BatchNorm2d(128)
    
        self.fc1 = nn.Linear(6272, 2048)
        self.bn4 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, num_classes)


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn3(self.conv3(x)))

        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = self.bn4(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn5(x)
        x = F.relu(x)

        x = self.fc3(x)
        return x


class AlexNet(nn.Module):
    """
    used for DomainNet and Office-Caltech10
    """
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2)),

                ('conv2', nn.Conv2d(64, 192, kernel_size=5, padding=2)),
                ('bn2', nn.BatchNorm2d(192)),
                ('relu2', nn.ReLU(inplace=True)),
                ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2)),

                ('conv3', nn.Conv2d(192, 384, kernel_size=3, padding=1)),
                ('bn3', nn.BatchNorm2d(384)),
                ('relu3', nn.ReLU(inplace=True)),

                ('conv4', nn.Conv2d(384, 256, kernel_size=3, padding=1)),
                ('bn4', nn.BatchNorm2d(256)),
                ('relu4', nn.ReLU(inplace=True)),

                ('conv5', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
                ('bn5', nn.BatchNorm2d(256)),
                ('relu5', nn.ReLU(inplace=True)),
                ('maxpool5', nn.MaxPool2d(kernel_size=3, stride=2)),
            ])
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(256 * 6 * 6, 4096)),
                ('bn6', nn.BatchNorm1d(4096)),
                ('relu6', nn.ReLU(inplace=True)),

                ('fc2', nn.Linear(4096, 4096)),
                ('bn7', nn.BatchNorm1d(4096)),
                ('relu7', nn.ReLU(inplace=True)),
            
                ('fc3', nn.Linear(4096, num_classes)),
            ])
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x