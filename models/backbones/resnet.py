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


class ResNet(nn.Module):
    '''
    ResNet: Residual Network | H(x) = F(x) + x
    '''
    def __init__(self, block, layers, num_classes=1000, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        '''
        :param block: BasicBlock | Bottleneck
        :param layers: Số block mỗi stage
        :param groups: -> Bottleneck
        :param width_per_group: -> Bottleneck
        :param replace_stride_with_dilation: -> segmentation (if exists)
        '''
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.in_channels = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}")

        self.groups = groups
        self.base_width = width_per_group

        # Conv1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)  # Ex: (1, 3, 224, 224) -> (1, 64, 112, 122)
        self.bn1 = norm_layer(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.max_pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))

        ''' 
            LayerX = 1 Stage = Nhiều residual blocks
            | Stage     |   planes   |   stride    |   Output size |
            --------------------------------------------------------
            | layer1    |     64     |      1      |     56 x 56   |
            --------------------------------------------------------
            | layer2    |    128     |      2      |     28 x 28   |
            --------------------------------------------------------
            | layer1    |    256     |      2      |     14 x 14   |
            --------------------------------------------------------
            | layer1    |    512     |      2      |      7 x 7    |
            --------------------------------------------------------
        '''
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])


        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False) -> nn.Sequential:
        '''
        :param planes: Số channel cơ sở
        :param blocks: Số block trong stage
        '''

        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.in_channels != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, planes * blocks.expansion, stride=stride),
                norm_layer(planes * blocks.expansion)
            )
        layers = []
        # Block đầu tiên của stage
        layers.append(
            block(
                self.in_channels,
                planes,
                stride,
                downsample,
                groups = self.groups,
                base_width = self.base_width,
                dilation = previous_dilation,
                norm_layer = norm_layer
            )
        )
        # Channel output của stage -> Channel input của stage tiếp theo
        self.in_channels = planes * block.expansion
        # Các block còn lại
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.in_channels,
                    planes,
                    groups = self.groups,
                    base_width = self.base_width,
                    dilation = self.dilation,
                    norm_layer = norm_layer
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        '''
            Giả sử input x: (1, 3, 224, 224)
            1. conv1 -> bn -> relu -> maxpool
                output: (1, 64, 56, 56)
            2. Layer1:
                (1, 64 * expansion, 56, 56)
                - BasicBlock -> (1, 64, 56, 56)
                - Bottleneck -> (1, 256, 56, 56)
            3. Layer2:
                (1, 128 * expansion, 28, 28)
            4. Layer3:
                (1, 256 * expansion, 14, 14)
            5. Layer4:
                (1, 512 * expansion, 7, 7)
            6. AvgPool:
                (1, 512 * expansion, 1, 1)
            7. Flatten + Fc:
                (1, num_classes)

        -----------------------------------------------
        Số layer của ResNet:
        Model                   Layers                 Tổng Layer
        Resnet-18            [2, 2, 2, 2]                   18
        Resnet-34            [3, 4, 6, 3]                   34
        Resnet-50            [3, 4, 6, 3]                   50
        ...
        '''
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pooling(x)

        x = self.layer1(x) # 1/4
        x = self.layer2(x) # 1/8
        x = self.layer3(x) # 1/16
        x = self.layer4(x) # 1/32

        x = self.avg_pooling(x)
        x = self.flatten(x, 1)

        x = self.fc(x)

        return x


# Wrapped ResNet
def _resnet(block, layers, weights, progress, **kwargs) -> ResNet:
    model = ResNet(block, layers, **kwargs)

    if weights is not None:
        state_dict = weights.get_state_dict(progress=progress, check_hash=True)
        model.load_state_dict(state_dict)

    return model


# Resnet34 | BasicBlock
def resnet34(*, pretrained=True, progress=True, **kwargs) -> ResNet:
    if pretrained:
        weights = ResNet34_Weights.DEFAULT
    else:
        weights = None
    return _resnet(BasicBlock, [3, 4, 6, 3], weights, progress, **kwargs)

# Resnet50 | BottleNeck
def resnet50(*, pretrained=True, progress=True, **kwargs) -> ResNet:
    if pretrained:
        weights = ResNet50_Weights.DEFAULT
    else:
        weights = None
    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)