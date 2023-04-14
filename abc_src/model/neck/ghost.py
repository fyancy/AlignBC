import math
import torch
import torch.nn as nn


class GhostModule(nn.Module):
    def __init__(self, inp, oup, norm_func=nn.BatchNorm2d, conv_func=nn.Conv2d,
                 kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        # 第一次卷积：得到通道数为init_channels，是输出的 1/ratio
        self.primary_conv = nn.Sequential(
            conv_func(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            norm_func(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential())

        # 第二次卷积：注意有个参数groups，为分组卷积
        # 每个feature map被卷积成 ratio-1 个新的 feature map
        self.cheap_operation = nn.Sequential(
            conv_func(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            norm_func(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        # 第一次卷积得到的 feature map，被作为 identity
        # 和第二次卷积的结果拼接在一起
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class GhostFeat(nn.Module):
    def __init__(self, inp, oup, norm_func=nn.BatchNorm2d, conv_func=nn.Conv2d,
                 kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super().__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        # 第一次卷积：得到通道数为init_channels，是输出的 1/ratio
        self.primary_conv = nn.Sequential(
            conv_func(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            norm_func(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential())

        # 第二次卷积：注意有个参数groups，为分组卷积
        # 每个feature map被卷积成 ratio-1 个新的 feature map
        self.cheap_operation = nn.Sequential(
            conv_func(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            norm_func(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return [x1, x2]

