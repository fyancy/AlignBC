import math
import torch
from torch import nn

import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from config import cfg


def _fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


def _make_conv_level(in_channels, out_channels, num_convs, norm_func,
                     stride=(1, 1), dilation=(1, 1)):
    """
    make conv layers based on its number.
    """
    modules = []
    for i in range(num_convs):
        modules.extend([
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3),
                      stride=stride if i == 0 else (1, 1),
                      padding=dilation, bias=False, dilation=dilation),
            norm_func(out_channels),
            nn.ReLU(inplace=True)])
        in_channels = out_channels

    return nn.Sequential(*modules)


def group_norm(out_channels):
    num_groups = cfg.MODEL.GROUP_NORM.NUM_GROUPS
    if out_channels % 32 == 0:
        return nn.GroupNorm(num_groups, out_channels)
    else:
        return nn.GroupNorm(num_groups // 2, out_channels)


def _fill_fc_weights(layers, value=0):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, value)


###########################


class BatchNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, running_var, eps, momentum,
                buffer_x2, buffer_gz, iters, buffer_size, debug, warmup_iters):
        ctx.eps = eps
        ctx.debug = debug
        ctx.buffer_size = buffer_size
        current_iter = iters.item()
        ctx.current_iter = current_iter
        ctx.warmup_iters = warmup_iters

        N, C, H, W = x.size()
        x2 = (x * x).mean(dim=3).mean(dim=2).mean(dim=0)

        # if current_iter <= buffer_size:
        #     buffer_x2[current_iter % buffer_size].copy_()
        #     var = x2.view(1, C, 1, 1)

        # elif current_iter < warmup_iters:
        #     var = x2.view(1, C, 1, 1)
        #     drop_x2 = buffer_x2[current_iter % buffer_size]
        #     sum_x2 = pre_x2 - drop_x2 +x2
        #     pre_x2.copy_(sum_x2)

        # else:
        #     drop_x2 = buffer_x2[current_iter % buffer_size]
        #     sum_x2 = pre_x2 - drop_x2 + x2
        #     pre_x2.copy_(sum_x2)
        #     var = sum_x2 / buffer_size
        #     var = var.view(1, C, 1, 1)

        buffer_x2[current_iter % buffer_size].copy_(x2)

        if current_iter <= buffer_size or current_iter < warmup_iters:
            var = x2.view(1, C, 1, 1)
        else:
            var = buffer_x2.mean(dim=0).view(1, C, 1, 1)

        z = x /(var + eps).sqrt()
        r = (var + eps).sqrt() / (running_var.view(1, C, 1, 1) + eps).sqrt()

        if current_iter <= max(1000, warmup_iters):
            r = torch.clamp(r, 1, 1)
        else:
            r = torch.clamp(r, 1/5, 5)

        y = r * z

        ctx.save_for_backward(z, var, weight, buffer_gz, r)

        running_var.copy_(momentum*running_var + (1-momentum)*var)
        y = weight.view(1,C,1,1) * y + bias.view(1,C,1,1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        debug = ctx.debug
        buffer_size = ctx.buffer_size
        current_iter = ctx.current_iter
        warmup_iters = ctx.warmup_iters

        N, C, H, W = grad_output.size()
        z, var, weight, buffer_gz, r = ctx.saved_variables

        y = r * z
        g = grad_output * weight.view(1, C, 1, 1)
        g = g * r
        gz = (g * z).mean(dim=3).mean(dim=2).mean(dim=0)

        buffer_gz[current_iter % buffer_size].copy_(gz)

        if current_iter <= buffer_size or current_iter < warmup_iters:
            mean_gz = gz.view(1, C, 1, 1)
        else:
            mean_gz = buffer_gz.mean(dim=0).view(1, C, 1, 1)

        gx = 1. / torch.sqrt(var + eps) * (g - z * mean_gz)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0),  None, None, None, None, None, None, None, None, None, None, None


class MABN2d(nn.Module):

    def __init__(self, channels, eps=1e-5, momentum=0.98, debug=False, B=2,
                 real_B=32, warmup_iters=100):
        super(MABN2d, self).__init__()
        assert real_B % B == 0
        self.buffer_size = real_B // B
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.register_buffer('running_var', torch.ones(1, channels, 1, 1))
        # self.register_buffer('running_var', torch.ones(channels))
        self.register_buffer('iters', torch.zeros(1).type(torch.LongTensor))
        self.register_buffer('buffer_x2', torch.zeros(self.buffer_size, channels))
        self.register_buffer('buffer_gz', torch.zeros(self.buffer_size, channels))

        self.eps = eps
        self.momentum = momentum
        self.debug = debug
        self.warmup_iters = warmup_iters

    def forward(self, x):
        if self.training:
            self.iters.copy_(self.iters + 1)
            x = BatchNormFunction.apply(x, self.weight, self.bias, self.running_var,
                                        self.eps, self.momentum, self.buffer_x2, self.buffer_gz,
                                        self.iters, self.buffer_size, self.debug, self.warmup_iters)
            return x
        else:
            N, C, H, W = x.size()
            var = self.running_var.view(1, C, 1, 1)
            x = x / (var + self.eps).sqrt()

        return self.weight.view(1,C,1,1) * x + self.bias.view(1,C,1,1)


class CenConv2d(nn.Module):
    """Conv2d layer with Weight Centralization.
    The args is exactly same as torch.nn.Conv2d. It's suggested to set bias=False when
    using CenConv2d with MABN.
    """
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        super(CenConv2d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight = nn.Parameter(torch.randn(out_planes, in_planes//groups, *_pair(kernel_size)))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_planes))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
