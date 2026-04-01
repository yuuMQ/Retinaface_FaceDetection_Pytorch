import torch
import torch.nn as nn
import timm

class SwinBackbone(nn.Module):
    '''
    Wrapper for Swin Transformer, return dict feature maps
    -> compatible with IntermediateLayerGetter interface

    Output channel:
        Swin-T = [192, 384, 768]
        Swin-S = [192, 384, 768]
        Swin-B = [256, 512, 1024]
    '''

    CHANNEL_MAP = {
        'swin_tiny_patch4_window7_224': [192, 384, 768],
        'swin_small_patch4_window7_224': [192, 384, 768],
        'swin_base_patch4_window7_224': [256, 512, 1024],
        'swin_base_patch4_window12_384': [256, 512, 1024],
    }

    def __init__(self, model_name='swin_small_patch4_window7_224', pretrained=True):
        super().__init__()

        self.model = timm.create_model(model_name, pretrained=pretrained, features_only=True, out_indices=(1, 2, 3)) # stage1->C3, stage2->C4, stage3->C5
        self.out_channels = self.CHANNEL_MAP[model_name]

    def forward(self, x):
        features = self.model(x) # List[(B, H, W, C)]
        out = {}

        for i, feat in enumerate(features):
            # Convert (B, H, W, C) -> (B, C, H, W)
            if feat.dim() == 4 and feat.shape[-1] != feat.shape[-2]:
                feat = feat.permute(0, 3, 1, 2).contiguous()
            out[f'layer{i+2}'] = feat # layer2, layer3, layer4

        return out