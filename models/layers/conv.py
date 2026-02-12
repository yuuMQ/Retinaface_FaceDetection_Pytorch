# Conv2d, DepthWise
import torch
import torch.nn as nn

def _make_divisible(v, divisor=8):
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v



class Conv2dNormActivation(nn.Sequential):
    '''
    Convolutional blocks: {
        Conv2d,
        BatchNorm2d,
        ReLU
    }
    '''
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size = 3,
            stride = 1,
            padding = None,
            groups = 1,
            norm_layers = nn.BatchNorm2d,
            activation_layer = torch.nn.LeakyReLU,
            dilation = 1,
            inplace = True,
            negative_slope = None,
            bias = False
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation

        layers = [
            nn.Conv2d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = kernel_size,
                stride = stride,
                padding = padding,
                dilation = dilation,
                groups = groups,
                bias = bias
            )
        ]
        if norm_layers is not None:
            layers.append(norm_layers(out_channels))

        if activation_layer is not None:
            params = {} if inplace is None else {'inplace': inplace}
            if negative_slope is not None:
                params['negative_slope'] = negative_slope
            layers.append(activation_layer(**params))
        super().__init__(*layers)

class DepthWiseSeparableConv2d(nn.Sequential):
    '''
        DepthWise Separable Convolution with
        {
            Depthwise,
            Pointwise Layers
        }
        followed by BatchNorm2d and ReLU
    '''
    def __init__(self, in_channels, out_channels, stride, norm_layer = None) -> None:

        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # Layers [Depthwise, Pointwise]
        layers = [
            Conv2dNormActivation(
                in_channels = in_channels,
                out_channels = in_channels,
                kernel_size = 3,
                stride = stride,
                groups = in_channels,
                negative_slope = 0.1
            ),
            Conv2dNormActivation(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 1,
                negative_slope = 0.1
            )
        ]

        super().__init__(*layers)
