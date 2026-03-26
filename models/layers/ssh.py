import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv2dNormActivation

class SSH(nn.Module):
    '''
        SSH (Single Stage Headless) Module for feature extraction
        Combine {
            3x3,
            5x5,
            7x7
        } convolutions with batch normalization and optional LeakyReLU activations function
    '''
    def __init__(self, in_channel, out_channel):
        '''

        Initializes the SSH module

        :param in_channel: số lượng input channel
        :param out_channel: số lượng output channel -> divisible (chia hết) cho 4
        '''

        super().__init__()

        assert out_channel % 4 == 0, 'Output channel must be divisible by 4'
        leaky = 0.1 if out_channel <= 64 else 0

        # 3x3 Convolution branch
        self.conv3X3 = Conv2dNormActivation(in_channel, out_channel // 2, kernel_size=3, activation_layer=None)

        # 5x5 Convolution branch
        self.conv5X5_1 = Conv2dNormActivation(in_channel, out_channel // 4, kernel_size=3, negative_slope=leaky)
        self.conv5X5_2 = Conv2dNormActivation(out_channel // 4, out_channel // 4, kernel_size=3, activation_layer=None)

        # 7x7 Convolution branch
        self.conv7X7_2 = Conv2dNormActivation(out_channel // 4, out_channel // 4, kernel_size=3, negative_slope=leaky)
        self.conv7X7_3 = Conv2dNormActivation(out_channel // 4, out_channel // 4, kernel_size=3, activation_layer=None)

    def forward(self, x):
        conv3X3 = self.conv3X3(x)
        conv5X5 = self.conv5X5_2(self.conv5X5_1(x))
        conv7X7 = self.conv7X7_3(self.conv7X7_2(self.conv5X5_1(x)))

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out, inplace=True)
        return out
