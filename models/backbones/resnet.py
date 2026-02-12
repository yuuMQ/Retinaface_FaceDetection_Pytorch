import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights


resnet_model_version = ['resnet18', 'resnet34', 'resnet50']

def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        groups=groups,
        padding=dilation,
        bias=False,
        dilation=dilation
    )

def conv1x1(in_channels, out_channels, stride=1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        bias=False
    )

# Prevented Vanishing Gradient
class BasicBlock(nn.Module):

    '''
        ! BasicBlock không mở rộng channel !
        Output Channels = out_channels * expansion
    '''
    expansion = 1 # Hệ số mở rộng


    def __init__(self, in_channels, out_channels, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups !=1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation != 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1 | Yakhyo
        # conv1
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # conv2
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# BottleNeck là residual block - Dùng 3 conv

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        '''
        :param inplanes: input channels vào block | output channels của block trước đó
        :param planes:  planes: số channel 'cơ sở' | số channel trong bottleneck | bottleneck width

        Layer               Channels
        ----------------------------
        Input               inplanes
        ----------------------------
        Conv1 (1x1)         planes
        ----------------------------
        Conv2 (3x3)         planes
        ----------------------------
        Conv3 (1x1)         planes * expansion
        ----------------------------
        Output              planes * expansion
        '''
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        # conv1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)

        # conv2
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)

        # conv3
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(x)
        out = self.bn2(x)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out