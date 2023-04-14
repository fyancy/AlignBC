"""
1) https://github.com/developer0hye/PyTorch-Deformable-Convolution-v2
2) https://github.com/4uiiurz1/pytorch-deform-conv-v2/blob/master/deform_conv_v2.py
"""

import math
import torch
from torch import nn
from torchvision.ops import deform_conv2d
from torch.nn.modules.utils import _pair


# Implementation 2: (I used this)
class DCNv2(nn.Module):
    """
    https://github.com/liyier90/pytorch-dcnv2/blob/master/dcn.py
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation=1, deformable_groups=1):
        super().__init__()

        self.num_chunks = 3  # Num channels for offset + mask
        self.in_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

        num_offset_mask_channels = (self.deformable_groups * self.num_chunks *
                                    self.kernel_size[0] * self.kernel_size[1])
        self.conv_offset_mask = nn.Conv2d(self.in_channels,
                                          num_offset_mask_channels,
                                          self.kernel_size,
                                          self.stride,
                                          self.padding,
                                          bias=True, )
        self.init_offset()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.conv_offset_mask(input)
        offset_1, offset_2, mask = torch.chunk(out, self.num_chunks, dim=1)
        offset = torch.cat((offset_1, offset_2), dim=1)
        mask = torch.sigmoid(mask)
        return deform_conv2d(input=input, offset=offset, weight=self.weight, bias=self.bias,
                             stride=self.stride, padding=self.padding,
                             dilation=self.dilation, mask=mask, )

    def init_offset(self) -> None:
        """Initializes the weight and bias for `conv_offset_mask`."""
        self.conv_offset_mask.weight.data.zero_()
        if self.conv_offset_mask.bias is not None:
            self.conv_offset_mask.bias.data.zero_()

    def reset_parameters(self) -> None:
        """Re-initialize parameters using a method similar to He
        initialization with mode='fan_in' and gain=1.
        """
        fan_in = self.in_channels
        for k in self.kernel_size:
            fan_in *= k
        std = 1.0 / math.sqrt(fan_in)
        self.weight.data.uniform_(-std, std)
        self.bias.data.zero_()
