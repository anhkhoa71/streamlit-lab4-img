import torch
import torch.nn as nn

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0.1):
        super(ConvNeXtBlock, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.conv2d_1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.gelu = nn.GELU()
        self.conv2d_2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv2d_1(x)
        x = self.gelu(x)
        x = self.conv2d_2(x)
        x = input + self.drop_path(x)
        return x
    
class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)
    def forward(self, x):
        if x.ndim == 2:
            return self.norm(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x
    
class DownSample(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DownSample, self).__init__()
        self.norm = LayerNorm(in_dim)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)
    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        return x

class ConvNeXt(nn.Module):
    def __init__(self, num_classes):
        super(ConvNeXt, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=4, stride=4),
            LayerNorm(96, eps=1e-6)
        )
        self.stage1 = nn.Sequential(
            self._make_stage(96, 3)
        )
        self.stage2 = nn.Sequential(
            DownSample(96, 192),
            self._make_stage(192, 3)
        )
        self.stage3 = nn.Sequential(
            DownSample(192, 384),
            self._make_stage(384, 9),
        )
        self.stage4 = nn.Sequential(
            DownSample(384, 768),
            self._make_stage(768, 3),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            LayerNorm(768, eps=1e-6),
            nn.Linear(768, num_classes)
        )
    def _make_stage(self, dim, depth):
        layers = []
        for _ in range(depth):
            layers.append(ConvNeXtBlock(dim))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x
    
        