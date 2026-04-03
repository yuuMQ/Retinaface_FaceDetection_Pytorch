'''
    Feature Enhancement Feature Pyramid Network (FEFPN)

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv2dNormActivation

class ChannelAttention(nn.Module):
    '''
        Squeeze-and-Excitation stype channel attention
        Dùng để enhance từng feature level trước khi đưa vào FPN
    '''
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        B, C, _, _ = x.shape
        avg = self.fc(self.avg_pool(x).view(B, C)) # view -> reshape
        max = self.fc(self.max_pool(x).view(B, C))
        attn = self.sigmoid(avg + max).view(B, C, 1, 1)
        return x * attn

class SpatialAttention(nn.Module):
    '''Dùng để focus vào khuôn mặt'''
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        max = x.max(dim=1, keepdim=True)[0]
        spatial = self.sigmoid(self.conv(torch.cat([avg, max], dim=1)))
        return x * spatial

class FeatureEnhancement(nn.Module):
    '''
    CBAM-style enhancement: Channel Attention -> Spatial Attention
    áp dụng lên từng level trước khi vào FPN (MLCA multi-level channel attention)
    '''
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

class FEFPN(nn.Module):
    '''
    Feature Enhancement FPN:
    1. Feature Enhancement (CBAM) trước projection.
    2. 1x1 projection về out-channels
    3. Top-down pathway với weighted fusion (BiFPN)
    4. 3x3 merge convolution

    input_channels_list: [C3_channel, C4_channels, C5_channels] - channels từ backbone
    output_channels: Số channels output (thường 256)
    '''
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        leaky = 0.1 if out_channels <= 64 else 0

        # 1. Feature Enhancement
        self.enhance1 = FeatureEnhancement(in_channels_list[0])
        self.enhance2 = FeatureEnhancement(in_channels_list[1])
        self.enhance3 = FeatureEnhancement(in_channels_list[2])

        # 2. 1x1 Lateral connection
        self.lateral1 = Conv2dNormActivation(in_channels=in_channels_list[0], out_channels=out_channels, kernel_size=1, negative_slope=leaky)
        self.lateral2 = Conv2dNormActivation(in_channels=in_channels_list[1], out_channels=out_channels, kernel_size=1, negative_slope=leaky)
        self.lateral3 = Conv2dNormActivation(in_channels=in_channels_list[2], out_channels=out_channels, kernel_size=1, negative_slope=leaky)

        # 3. BiFPN-style learnable fusion weights
        # w_i >= 0 -> normalize qua softmax -> weighted sum khi merge
        self.w2 = nn.Parameter(torch.ones(2)) # Merge C4 + upsample(C5)
        self.w1 = nn.Parameter(torch.ones(2)) # Merge C3 + upsample(C4_merged)

        # 4. 3x3 Merge convolutions
        self.merge1 = Conv2dNormActivation(in_channels=out_channels, out_channels=out_channels, kernel_size=3, negative_slope=leaky)
        self.merge2 = Conv2dNormActivation(in_channels=out_channels, out_channels=out_channels, kernel_size=3, negative_slope=leaky)

    def forward(self, inputs):
        '''
            inputs: dict {'layer2': C3, 'layer3': C4, 'layer4': C5}
        '''
        c3, c4, c5 = list(inputs.values())

        # 1. Feature Enhancement
        c3 = self.enhance1(c3)
        c4 = self.enhance2(c4)
        c5 = self.enhance3(c5)

        # 2. Lateral connection (Lateral projections)
        p3 = self.lateral1(c3)
        p4 = self.lateral2(c4)
        p5 = self.lateral3(c5)

        # 3. Top-down pathway with weighted fusion
        eps = 1e-4

        # Merge P5 -> P4
        w2 = F.relu(self.w2)
        w2 = w2 / (w2.sum() + eps)
        up5 = F.interpolate(p5, size=p4.shape[2:], mode='nearest')
        p4_merge = self.merge2(w2[0] * p4 + w2[1] * up5)

        # Merge P4_merged -> P3
        w1 = F.relu(self.w1)
        w1 = w1 / (w1.sum() + eps)
        up4 = F.interpolate(p4_merge, size=p3.shape[2:], mode='nearest')
        p3_merge = self.merge1(w1[0] * p3 + w1[1] * up4)

        return [p3_merge, p4_merge, p5]