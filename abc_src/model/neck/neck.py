
import torch.nn as nn

from model.neck.ghost import GhostModule, GhostFeat

from model.layers.make_layers import group_norm, MABN2d, CenConv2d


_NORM_SPECS = {"BN": nn.BatchNorm2d, "GN": group_norm, 'MABN': MABN2d}
_CONV_SPECS = {"Conv": nn.Conv2d, "CenCon": CenConv2d}
conv_func = _CONV_SPECS["Conv"]


class Neck(nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()

        norm_func = _NORM_SPECS[cfg.MODEL.HEAD.USE_NORMALIZATION]
        global conv_func
        conv_func = _CONV_SPECS["CenCon"] if cfg.MODEL.BACKBONE.USE_NORMALIZATION == "MABN" \
            else _CONV_SPECS["Conv"]
        self.ghost = GhostFeat(in_channels, int(in_channels*2), norm_func, conv_func)

    def forward(self, x):
        x = self.ghost(x)
        return x


class AdNeck(nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()

        norm_func = _NORM_SPECS[cfg.MODEL.HEAD.USE_NORMALIZATION]
        global conv_func
        conv_func = _CONV_SPECS["CenCon"] if cfg.MODEL.BACKBONE.USE_NORMALIZATION == "MABN" \
            else _CONV_SPECS["Conv"]
        self.feat_conv1 = nn.Sequential(
            conv_func(in_channels, in_channels, 1, 1, 0, bias=False),
            norm_func(in_channels),
            nn.ReLU(inplace=True),
        )
        self.feat_conv2 = nn.Sequential(
            conv_func(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False),
            norm_func(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.feat_conv1(x)
        x2 = self.feat_conv2(x)

        return [x1, x2]
