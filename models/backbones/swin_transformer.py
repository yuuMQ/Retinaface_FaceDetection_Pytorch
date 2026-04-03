import torch
import torch.nn as nn
import timm

class SwinV2Backbone(nn.Module):
    '''
    Wrapper for Swin Transformer, return dict feature maps
    -> compatible with IntermediateLayerGetter interface

    Output channel:
        Swin-T = [192, 384, 768]
        Swin-S = [192, 384, 768]
        Swin-B = [256, 512, 1024]


    Wrapper Swin Transformer V2 cho RetinaFace.

    Giải quyết 3 vấn đề của SwinV1 (arXiv:2111.09883 - CVPR 2022):
      1. Training instability  → res-post-norm + scaled cosine attention
      2. Resolution gap        → log-spaced Continuous Position Bias (CPB)
      3. Data hunger           → SimMIM pre-training support

    Có 2 loại model trong timm:
      - swinv2_*    : Microsoft official, output NHWC → cần permute
      - swinv2_cr_* : timm reimplementation, output NCHW → KHÔNG cần permute

    Channels (giống SwinV1, chỉ thay đổi kiến trúc bên trong):
      Tiny / Small  (C=96):  Stage2=192, Stage3=384, Stage4=768
      Base          (C=128): Stage2=256, Stage3=512, Stage4=1024

    '''

    '''
        (model_name -> out_channel, is_nchw)
        is_nchw = True  -> timm returns (B, C, H, W).
        is_nchw = False -> timm returns (B, H, W, C).
    '''

    MODEL_REGISTRY = {
        # ── Official Microsoft SwinV2 (NHWC) ─────────────────────────────
        # Tiny — C=96, input 256×256
        'swinv2_tiny_window8_256.ms_in1k': ([192, 384, 768], False),
        'swinv2_tiny_window16_256.ms_in1k': ([192, 384, 768], False),
        # Small — C=96
        'swinv2_small_window8_256.ms_in1k': ([192, 384, 768], False),
        'swinv2_small_window16_256.ms_in1k': ([192, 384, 768], False),
        # Base — C=128, IN1K
        'swinv2_base_window8_256.ms_in1k': ([256, 512, 1024], False),
        'swinv2_base_window16_256.ms_in1k': ([256, 512, 1024], False),
        # Base — C=128, IN22K pretrained (tốt nhất cho downstream tasks)
        'swinv2_base_window12_192.ms_in22k': ([256, 512, 1024], False),
        'swinv2_base_window12to16_192to256.ms_in22k_ft_in1k': ([256, 512, 1024], False),
        # Large — C=192, IN22K
        'swinv2_large_window12_192.ms_in22k': ([384, 768, 1536], False),
        'swinv2_large_window12to16_192to256.ms_in22k_ft_in1k': ([384, 768, 1536], False),

        # ── timm CR reimplementation (NCHW — dễ dùng hơn) ────────────────
        'swinv2_cr_tiny_ns_224.sw_in1k': ([192, 384, 768], True),
        'swinv2_cr_tiny_384': ([192, 384, 768], True),
        'swinv2_cr_small_ns_224.sw_in1k': ([192, 384, 768], True),
        'swinv2_cr_small_224.sw_in1k': ([192, 384, 768], True),
    }

    def __init__(self, model_name='swinv2_cr_tiny_ns_224.sw_in1k', pretrained=True):
        super().__init__()

        self.out_channels, self._is_nchw = self.MODEL_REGISTRY[model_name] # Out channels [C3, C4, C5]

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3), # Stage2 (H / 8), Stage3 (H/16), Stage4 (H/32)
            img_size=640,
        )


    def forward(self, x):
        '''
        x: (B x 3 x H x W)
        returns:
            dict
            {
                'layer2': ,
                'layer3': ,
                'layer4': ,
            }
            each tensor's shape (B, C, H/8, W/8), (B, C, H/16, W/16), (B, C, H/32, W/32)
        '''
        features = self.model(x)
        out = {}

        for i, feat in enumerate(features):
            if not self._is_nchw:
                # NHWC (B, H, W, C) -> NCHW (B, C, H, W)
                feat = feat.permute(0, 3, 1, 2).contiguous()
            out[f'layer{i + 2}'] = feat  # layer2, layer3, layer4
        return out