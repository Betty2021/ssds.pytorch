import torch
import torch.nn as nn

from collections import namedtuple
import functools

Conv = namedtuple('Conv', ['stride', 'depth'])
DepthSepConv = namedtuple('DepthSepConv', ['stride', 'depth'])
InvertedResidualDef = namedtuple('InvertedResidual', ['stride', 'depth', 'num', 't']) # t is the expension factor

V1_CONV_DEFS = [
    Conv(stride=2, depth=32),
    DepthSepConv(stride=1, depth=64),
    DepthSepConv(stride=2, depth=128),
    DepthSepConv(stride=1, depth=128),
    DepthSepConv(stride=2, depth=256),
    DepthSepConv(stride=1, depth=256),
    DepthSepConv(stride=2, depth=512),
    DepthSepConv(stride=1, depth=512),
    DepthSepConv(stride=1, depth=512),
    DepthSepConv(stride=1, depth=512),
    DepthSepConv(stride=1, depth=512),
    DepthSepConv(stride=1, depth=512),
    DepthSepConv(stride=2, depth=1024),
    DepthSepConv(stride=1, depth=1024)
]

V2_CONV_DEFS = [
    Conv(stride=2, depth=32),
    InvertedResidualDef(stride=1, depth=16, num=1, t=1), #1
    InvertedResidualDef(stride=2, depth=24, num=2, t=6), #2,3
    InvertedResidualDef(stride=2, depth=32, num=3, t=6), #4,5,6
    InvertedResidualDef(stride=2, depth=64, num=4, t=6), #7,8,9,10
    InvertedResidualDef(stride=1, depth=96, num=3, t=6), #11,12,13
    InvertedResidualDef(stride=2, depth=160, num=3, t=6), #14, 15, 16
    InvertedResidualDef(stride=1, depth=320, num=1, t=6), #17
]

class _conv_bn(nn.Module):
    def __init__(self, inp, oup, stride):
        super(_conv_bn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )
        self.depth = oup

    def forward(self, x):
        return self.conv(x)


class _conv_dw(nn.Module):
    def __init__(self, inp, oup, stride):
        super(_conv_dw, self).__init__()
        self.conv = nn.Sequential(
            # dw
            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),
            # pw
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )
        self.depth = oup

    def forward(self, x):
        return self.conv(x)


class _inverted_residual_bottleneck(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(_inverted_residual_bottleneck, self).__init__()
        self.use_res_connect = stride == 1 and inp == oup
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )
        self.depth = oup
        
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )


    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

def mobilenet(conv_defs, depth_multiplier=1.0, min_depth=8):
    depth = lambda d: max(int(d * depth_multiplier), min_depth)
    layers = []
    in_channels = 3
    for conv_def in conv_defs:
        outp = depth(conv_def.depth)
        if isinstance(conv_def, Conv):
            #layers += [_conv_bn(in_channels, depth(conv_def.depth), conv_def.stride)]
            #in_channels = depth(conv_def.depth)
            layers += [ nn.Sequential(
                            nn.Conv2d(in_channels, outp, 3, conv_def.stride, padding=1, bias=False),
                            nn.BatchNorm2d(outp),
                            nn.ReLU6(inplace=True)
                        )
                      ]
            in_channels = outp
        elif isinstance(conv_def, DepthSepConv):
            layers += [_conv_dw(in_channels, outp, conv_def.stride)]
            in_channels = outp
        elif isinstance(conv_def, InvertedResidualDef):
            for n in range(conv_def.num):
                stride = conv_def.stride if n == 0 else 1
                #layers += [_inverted_residual_bottleneck(in_channels, depth(conv_def.depth), stride, conv_def.t)]
                layers += [InvertedResidual(in_channels, outp, stride, conv_def.t)]
                in_channels = outp


    return layers

def wrapped_partial(func, *args, **kwargs):
    partial_func = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial_func, func)
    return partial_func

mobilenet_v1 = wrapped_partial(mobilenet, conv_defs=V1_CONV_DEFS, depth_multiplier=1.0)
mobilenet_v1_075 = wrapped_partial(mobilenet, conv_defs=V1_CONV_DEFS, depth_multiplier=0.75)
mobilenet_v1_050 = wrapped_partial(mobilenet, conv_defs=V1_CONV_DEFS, depth_multiplier=0.50)
mobilenet_v1_025 = wrapped_partial(mobilenet, conv_defs=V1_CONV_DEFS, depth_multiplier=0.25)

mobilenet_v2 = wrapped_partial(mobilenet, conv_defs=V2_CONV_DEFS, depth_multiplier=1.0)
mobilenet_v2_075 = wrapped_partial(mobilenet, conv_defs=V2_CONV_DEFS, depth_multiplier=0.75)
mobilenet_v2_050 = wrapped_partial(mobilenet, conv_defs=V2_CONV_DEFS, depth_multiplier=0.50)
mobilenet_v2_025 = wrapped_partial(mobilenet, conv_defs=V2_CONV_DEFS, depth_multiplier=0.25)