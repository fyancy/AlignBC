
import math
import os
import torch
from torch import nn
import numpy as np
from torch.nn.modules.utils import _pair
import torch.utils.model_zoo as model_zoo

from model.layers.make_layers import group_norm, MABN2d, CenConv2d
from model.layers.deform_conv import DeformConv


# ---------------- Hyper-params --------------

_NORM_SPECS = {"BN": nn.BatchNorm2d, "GN": group_norm, 'MABN': MABN2d}
_CONV_SPECS = {"Conv": nn.Conv2d, "CenCon": CenConv2d}
conv_func = _CONV_SPECS["Conv"]


def build_backbone(cfg):
    global conv_func
    conv_func = _CONV_SPECS["CenCon"] if cfg.MODEL.BACKBONE.USE_NORMALIZATION is "MABN" \
        else _CONV_SPECS["Conv"]
    return DLASeg(cfg=cfg, last_level=5)


class DLASeg(nn.Module):
    def __init__(self, cfg, last_level=5, out_channel=0):
        super().__init__()

        down_ratio = cfg.MODEL.BACKBONE.DOWN_RATIO
        assert down_ratio in [2, 4, 8, 16]

        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level

        # self.base = globals()[base_name]
        norm_func = _NORM_SPECS[cfg.MODEL.BACKBONE.USE_NORMALIZATION]
        self.base = dla34(pretrained=cfg.MODEL.PRETRAIN, norm_func=norm_func)

        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        # [16, 32, 64, 128, 256, 512], selected: 64, 128, 256, 512

        self.dla_up = DLAUp(startp=self.first_level,
                            channels=channels[self.first_level:],
                            scales=scales,
                            norm_func=norm_func)

        if out_channel == 0:
            out_channel = channels[self.first_level]
        self.out_channels = out_channel

        up_scales = [2 ** i for i in range(self.last_level - self.first_level)]
        self.ida_up = IDAUp(in_channels=channels[self.first_level:self.last_level],
                            out_channel=out_channel,
                            up_f=up_scales,  # up-strides: [1,2,4]
                            norm_func=norm_func)

    def forward(self, x):
        x = self.base(x)
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        # y: [(n*64*128*128), (n*128*64*64), (n*256*32*32), (n*512*16*16)]

        self.ida_up(y, 0, len(y))
        # ida(y, 0, 3)
        # y[0] --> n*64*128*128 (no up-sample)
        # y[1] = (y[1]->proj->up+y[0])->node, n*128*64*64->n*64*128*128 (up=2)
        # y[2] = (y[2]->proj->up+y[1])->node, n*256*32*32->n*64*128*128 (up=4)

        return y[-1]
        # return y[-2:]


# --------------------
# Modules
# --------------------


def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
    return os.path.join('http://dl.yf.io/dla/models', data, '{}-{}.pth'.format(name, hash))


class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_func,
                 stride=1,
                 dilation=1):
        super(BasicBlock, self).__init__()

        self.conv1 = conv_func(in_channels,
                               out_channels,
                               kernel_size=(3, 3),
                               stride=_pair(stride),
                               padding=_pair(dilation),
                               bias=False,
                               dilation=_pair(dilation))
        self.bn1 = norm_func(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv_func(out_channels,
                               out_channels,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=_pair(dilation),
                               bias=False,
                               dilation=_pair(dilation))
        self.bn2 = norm_func(out_channels)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_func,
                 kernel_size,
                 residual):
        super(Root, self).__init__()

        self.conv = conv_func(in_channels,
                              out_channels,
                              kernel_size=(1, 1),
                              stride=(1, 1),
                              bias=False,
                              padding=(kernel_size - 1) // 2)

        self.bn = norm_func(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self,
                 level,
                 block,
                 in_channels,
                 out_channels,
                 norm_func,
                 stride=1,
                 level_root=False,
                 root_dim=0,
                 root_kernel_size=1,
                 dilation=1,
                 root_residual=False
                 ):
        super(Tree, self).__init__()

        if root_dim == 0:
            root_dim = 2 * out_channels

        if level_root:
            root_dim += in_channels

        if level == 1:
            self.tree1 = block(in_channels,
                               out_channels,
                               norm_func,
                               stride,
                               dilation=dilation)

            self.tree2 = block(out_channels,
                               out_channels,
                               norm_func,
                               stride=1,
                               dilation=dilation)
        else:
            new_level = level - 1
            self.tree1 = Tree(new_level,
                              block,
                              in_channels,
                              out_channels,
                              norm_func,
                              stride,
                              root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation,
                              root_residual=root_residual)

            self.tree2 = Tree(new_level,
                              block,
                              out_channels,
                              out_channels,
                              norm_func,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation,
                              root_residual=root_residual)
        if level == 1:
            self.root = Root(root_dim,
                             out_channels,
                             norm_func,
                             root_kernel_size,
                             root_residual)

        self.level_root = level_root
        self.root_dim = root_dim
        self.level = level

        self.downsample = None
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)

        self.project = None
        if in_channels != out_channels:
            self.project = nn.Sequential(
                conv_func(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False),
                norm_func(out_channels)
            )

    def forward(self, x, residual=None, children=None):
        if children is None:
            children = []

        if self.downsample:
            bottom = self.downsample(x)
        else:
            bottom = x

        if self.project:
            residual = self.project(bottom)
        else:
            residual = bottom

        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)

        if self.level == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLABase(nn.Module):
    def __init__(self,
                 levels,
                 channels,
                 block=BasicBlock,
                 residual_root=False,
                 norm_func=nn.BatchNorm2d,
                 ):
        super(DLABase, self).__init__()

        self.channels = channels
        self.level_length = len(levels)

        self.base_layer = nn.Sequential(conv_func(3, channels[0], (7, 7), (1, 1), (3, 3), bias=False),
                                        norm_func(channels[0]),
                                        nn.ReLU(inplace=True))

        self.level0 = self._make_conv_level(in_channels=channels[0],
                                            out_channels=channels[0],
                                            num_convs=levels[0],
                                            norm_func=norm_func)

        self.level1 = self._make_conv_level(in_channels=channels[0],
                                            out_channels=channels[1],
                                            num_convs=levels[0],
                                            norm_func=norm_func,
                                            stride=2)

        self.level2 = Tree(level=levels[2],
                           block=block,
                           in_channels=channels[1],
                           out_channels=channels[2],
                           norm_func=norm_func,
                           stride=2,
                           level_root=False,
                           root_residual=residual_root)

        self.level3 = Tree(level=levels[3],
                           block=block,
                           in_channels=channels[2],
                           out_channels=channels[3],
                           norm_func=norm_func,
                           stride=2,
                           level_root=True,
                           root_residual=residual_root)

        self.level4 = Tree(level=levels[4],
                           block=block,
                           in_channels=channels[3],
                           out_channels=channels[4],
                           norm_func=norm_func,
                           stride=2,
                           level_root=True,
                           root_residual=residual_root)

        self.level5 = Tree(level=levels[5],
                           block=block,
                           in_channels=channels[4],
                           out_channels=channels[5],
                           norm_func=norm_func,
                           stride=2,
                           level_root=True,
                           root_residual=residual_root)

    def _make_conv_level(self, in_channels, out_channels, num_convs, norm_func,
                         stride=(1, 1), dilation=(1, 1)):
        modules = []
        for i in range(num_convs):
            modules.extend([
                conv_func(in_channels, out_channels, kernel_size=(3, 3),
                          stride=stride if i == 0 else (1, 1),
                          padding=dilation, bias=False, dilation=dilation),
                norm_func(out_channels),
                nn.ReLU(inplace=True)])
            in_channels = out_channels
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)

        for i in range(self.level_length):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)

        return y

    def load_pretrained_model(self, data='imagenet', name='dla34', hash='ba72cf86'):
        if name.endswith('.pth'):
            model_weights = torch.load(data + name)
        else:
            model_url = get_model_url(data, name, hash)
            model_weights = model_zoo.load_url(model_url)
            # print(model_weights.keys())  # C:\Users\Asus\.cache\torch\hub\checkpoints
        num_classes = len(model_weights[list(model_weights.keys())[-1]])
        self.fc = conv_func(
            self.channels[-1], num_classes,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.load_state_dict(model_weights, strict=False)


def dla34(pretrained, norm_func, **kwargs):  # DLA-34
    model = DLABase([1, 1, 1, 2, 2, 1],
                    [16, 32, 64, 128, 256, 512],
                    block=BasicBlock, norm_func=norm_func, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla34', hash='ba72cf86')
        print("Load the pretrained [dla34], done.")

    return model


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, (nn.Conv2d, CenConv2d)):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class DLAUp(nn.Module):
    def __init__(self,
                 startp,
                 channels,
                 scales,
                 in_channels=None,
                 norm_func=nn.BatchNorm2d):
        super(DLAUp, self).__init__()

        self.startp = startp

        if in_channels is None:
            in_channels = channels
        self.channels = channels  # [64, 128, 256, 512]
        channels = list(channels)
        scales = np.array(scales, dtype=int)  # [1,2,4,8]
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(in_channels[j:], channels[j], scales[j:] // scales[j], norm_func))
            scales[j + 1:] = scales[j]  # scales: [1,2,4,8]=>[1,2,4,4]=>[1,2,2,2]=>
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]
            # in_chs: [64,128,256,512]=>[64,128,256,256]=>[64,128,128,128]=>

    def forward(self, layers):
        out = [layers[-1]]
        for i in range(len(layers) - self.startp - 1):  # startp=2
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) - i - 2, len(layers))
            out.insert(0, layers[-1])
            # 输出out列表长度为4，存储了初始的layer5以及三次变换后的layer5(feat size降序排, size: 128, 64, 32, 16)
            # [(n*64*128*128), (n*128*64*64), (n*256*32*32), (n*512*16*16)]
        return out


class IDAUp(nn.Module):
    def __init__(self, in_channels, out_channel, up_f, norm_func):
        super(IDAUp, self).__init__()

        for i in range(1, len(in_channels)):
            in_channel = in_channels[i]
            f = int(up_f[i])
            proj = DeformConv(in_channel, out_channel, norm_func)
            node = DeformConv(out_channel, out_channel, norm_func)

            up = nn.ConvTranspose2d(out_channel, out_channel, _pair(f * 2), _pair(f),
                                    padding=_pair(f // 2), output_padding=(0, 0),
                                    groups=out_channel, bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)

    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])


if __name__ == "__main__":
    from config import cfg
    import torch

    dla_net = DLASeg(cfg).cuda()
    in_x = torch.empty([2, 3, 512, 512]).cuda()
    out_x = dla_net(in_x)
    # for o in out_x:
    print(out_x.shape)
