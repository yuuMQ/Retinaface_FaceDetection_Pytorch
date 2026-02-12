import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv2dNormActivation

class FPN(nn.Module):
    '''
        FPN (Feature Pyramid Network) for multi-scale feature map extraction and merging.
        Uses 1x1 convolutions for output layers and 3x3 convolutions for merging layers.
    '''

    def __init__(self, in_channels_list, out_channels):
        '''
        :param in_channels_list: List of input channel sizes for each pyramid level
        :param out_channels: Number of output channels for feature pyramid
        '''
        super().__init__()
        leaky = 0.1 if out_channels <= 64 else 0

        # 1x1 Convolution output layers
        self.output1 = Conv2dNormActivation(in_channels_list[0], out_channels, kernel_size=1, negative_slope=leaky)
        self.output2 = Conv2dNormActivation(in_channels_list[1], out_channels, kernel_size=1, negative_slope=leaky)
        self.output3 = Conv2dNormActivation(in_channels_list[2], out_channels, kernel_size=1, negative_slope=leaky)

        # Merge Layers with 3x3 Convolutions
        self.merge1 = Conv2dNormActivation(out_channels, out_channels, kernel_size=3, negative_slope=leaky)
        self.merge2 = Conv2dNormActivation(out_channels, out_channels, kernel_size=3, negative_slope=leaky)

    def forward(self, inputs):
        '''
        :param inputs: input feature maps from diff levels of the pyramid
        :return: list of merged output feature maps at diff scales.
        '''

        inputs = list(inputs.values())

        # Apply output layers to each feature map
        output1 = self.output1(inputs[0])
        output2 = self.output2(inputs[1])
        output3 = self.output3(inputs[2])

        # Merge outputs with upsampling and addition
        upsample3 = F.interpolate(output3, size=output2.shape[2:], mode='nearest')
        output2 = self.merge2(output2 + upsample3)

        upsample2 = F.interpolate(output2, size=output1.shape[2:], mode='nearest')
        output1 = self.merge1(output1 + upsample2)

        return [output1, output2, output3]