from layers.ssh import SSH
from layers.fpn import FPN
from heads import ClassHead, BboxHead, LandmarkHead
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models._utils as _utils
from torchvision.models import resnet34, resnet50


def get_layer_extractor(cfg, backbone):
    return _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])

def build_backbone(name, pretrained=False):
    backbone_map = {
        'resnet34': lambda: resnet34(pretrained=pretrained),
        'resnet50': lambda: resnet50(pretrained=pretrained),
    }
    if name not in backbone_map:
        raise ValueError(f"Unsupported backbone name: {name}")

    return backbone_map[name]()



class RetinaFace(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        backbone = build_backbone(cfg['name'], cfg['pretrained'])
        self.fx = get_layer_extractor(cfg, backbone) # feature extraction

        num_anchors = 2
        base_in_channels = cfg['in_channel']
        out_channels = cfg['out_channel']

        fpn_in_channels = [
            base_in_channels * 2,
            base_in_channels * 4,
            base_in_channels * 8,
        ]

        self.fpn = FPN(fpn_in_channels, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.class_head = ClassHead(in_channels=cfg['out_channel'], num_anchors=num_anchors, fpn_num=3)
        self.bbox_head = BboxHead(in_channels=cfg['out_channel'], num_anchors=num_anchors, fpn_num=3)
        self.landmark_head = LandmarkHead(in_channels=cfg['out_channel'], num_anchors=num_anchors, fpn_num=3)


    def forward(self, x):
        out = self.fx(x)
        fpn = self.fpn(out)

        # single-stage headless module
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])

        features = [feature1, feature2, feature3]

        classifications = self.class_head(features)
        bbox_regressions = self.bbox_head(features)
        landmark_regressions = self.landmark_head(features)

        if self.training:
            output = (bbox_regressions, classifications, landmark_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), landmark_regressions)

        return output