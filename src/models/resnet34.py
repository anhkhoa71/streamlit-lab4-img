from torch import nn
import torch

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, first_stage=False):
        super().__init__()
        stride = 2 if first_stage else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(p=0.5)  # giữ 50% units
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class ResNet34(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2_x = nn.Sequential(
            BasicBlock(in_channels=64, out_channels=64),
            BasicBlock(in_channels=64, out_channels=64),
            BasicBlock(in_channels=64, out_channels=64)
        )
        self.conv3_x = nn.Sequential(
            BasicBlock(in_channels=64, out_channels=128, first_stage=True),
            BasicBlock(in_channels=128, out_channels=128),
            BasicBlock(in_channels=128, out_channels=128),
            BasicBlock(in_channels=128, out_channels=128)
        )
        self.conv4_x = nn.Sequential(
            BasicBlock(in_channels=128, out_channels=256, first_stage=True),
            BasicBlock(in_channels=256, out_channels=256),
            BasicBlock(in_channels=256, out_channels=256),
            BasicBlock(in_channels=256, out_channels=256),
            BasicBlock(in_channels=256, out_channels=256),
            BasicBlock(in_channels=256, out_channels=256)
        )
        self.conv5_x = nn.Sequential(
            BasicBlock(in_channels=256, out_channels=512, first_stage=True),
            BasicBlock(in_channels=512, out_channels=512),
            BasicBlock(in_channels=512, out_channels=512)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
