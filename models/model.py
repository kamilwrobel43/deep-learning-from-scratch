import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size, stride = 1, padding = 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Identity()
        if stride!=1 or out_channels!=in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, padding = 0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        identity = self.shortcut(x)

        x = self.conv1(x)
        x = self.conv2(x)
        output = self.relu(x + identity)
        return output

class ResNet18(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias = False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Sequential(
            BasicBlock(64, 64, 3, 1),
            BasicBlock(64,64,3,1)
        )

        self.conv3 = nn.Sequential(
            BasicBlock(64, 128, 3, 2),
            BasicBlock(128, 128, 3, 1)
        )

        self.conv4 = nn.Sequential(
            BasicBlock(128, 256, 3, 2),
            BasicBlock(256,256, 3, 1)
        )

        self.conv5 = nn.Sequential(
            BasicBlock(256, 512, 3, 2),
            BasicBlock(512, 512, 3, 1)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x       
        




class ResNet(nn.Module):
    def __init__(self, in_channels: int, num_layers: list, block, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias = False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 3, 1, num_layers[0], block)
        self.layer2 = self._make_layer(64, 128, 3, 2, num_layers[1], block)
        self.layer3 = self._make_layer(128, 256, 3, 2, num_layers[2], block)        
        self.layer4 = self._make_layer(256, 512, 3, 2, num_layers[3], block)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(in_features=512, out_features=num_classes)



    def _make_layer(self, in_channels, out_channels, kernel_size, stride, num_layer: int, block):
        layers = []
        layers.append(block(in_channels, out_channels, kernel_size, stride))
        for _ in range(num_layer-1):
            layers.append(block(out_channels, out_channels, kernel_size, 1))
        return nn.Sequential(*layers)
    

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x 
        