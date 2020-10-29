
"""
Pytorch implementation of InceptionNet/GoogLeNet
from 'Going Deeper with Convolutions' (https://arxiv.org/abs/1409.4842)

"""

import torch
import torch.nn as nn


class GoogLeNet(nn.Module):
    """
    GoogLeNet/Inception network
        Expects images of size 3x224x224
    """
    def __init__(self, in_channels=3, num_classes=1000):
        super(GoogLeNet, self).__init__()

        self.conv1 = ConvBlock(in_channels, out_channels=64, kernel_size=(7,7), stride=(2,2), padding=(3,3))
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = ConvBlock(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Order: in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool

        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)

        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(1024, num_classes)

        print(f'GoogLeNet initialized with in_channels={in_channels}, outdim={num_classes}')

    def forward(self, x):
        # x is (n, in_channels, 224, 224)
        x = self.conv1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)

        x = x.reshape(x.shape[0], -1)

        x = self.dropout(x)
        x = self.fc1(x)
        return x


class InceptionBlock(nn.Module):
    """
        Inception Block as it is defined in the original paper
    """
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):
        super(InceptionBlock, self).__init__()

        self.branch1 = ConvBlock(in_channels, out_1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, red_3x3, kernel_size=1),
            ConvBlock(red_3x3, out_3x3, kernel_size=3, padding=1)  # To keep same size
        )

        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, red_5x5, kernel_size=1),
            ConvBlock(red_5x5, out_5x5, kernel_size=5, padding=2)  # To keep same size
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, out_1x1pool, kernel_size=1)
        )

    def forward(self, x):
        """Concatinate all channels"""
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)


class ConvBlock(nn.Module):
    """
        Conv block with batch normalization and relu
    """
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBlock, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))



if __name__ == '__main__':
    x = torch.randn(5, 3, 224, 224)
    model = GoogLeNet()
    print(model(x).shape)