import torch
import torch.nn as nn
import torch.nn.functional as F

class ShuffleNetUnits(nn.Module):
    def __init__(self, in_channels, out_channels, group=4, stride=1):
        super().__init__()
        mid_channels = out_channels // 4
        self.stride = stride
        self.group = group
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, padding=0, groups=group),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        out_stage2_channels = out_channels if stride == 1 else out_channels - in_channels
        self.stage2 = nn.Sequential(
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=stride, padding=1, groups=mid_channels),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_stage2_channels, kernel_size=1, stride=1, padding=0, groups=group),
            nn.BatchNorm2d(out_stage2_channels)
        )
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1) if stride == 2 else nn.Identity()

    
    def forward(self, x):
        residual = self.pool(x)
        x = self.stage1(x)
        x = self.channel_shuffle(x, self.group)
        x = self.stage2(x)
        if self.stride == 1:
            x = x + residual
        else:
            x = torch.cat((residual, x), 1)
        x = self.relu(x)
        return x
    
    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups

        x = x.view(batchsize, groups, channels_per_group, height, width)

        x = torch.transpose(x, 1, 2).contiguous()

        x = x.view(batchsize, -1, height, width)

        return x

class ShuffleNet(nn.Module):
    def __init__(self, num_classes=6, group=4):
        super().__init__()
        self.group = group
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.stage2 = self._make_stage(24, 272, num_units=4)
        self.stage3 = self._make_stage(272, 544, num_units=8)
        self.stage4 = self._make_stage(544, 1088, num_units=4)
        self.globalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(1088, 1000)
        self.fc2 = nn.Linear(1000, num_classes)

    def _make_stage(self, in_channels, out_channels, num_units):
        layers = []
        layers.append(ShuffleNetUnits(in_channels, out_channels, group=self.group, stride=2))
        for _ in range(num_units - 1):
            layers.append(ShuffleNetUnits(out_channels, out_channels, group=self.group, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.globalpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x