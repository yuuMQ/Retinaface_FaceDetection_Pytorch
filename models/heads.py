import torch
import torch.nn as nn

class ClassHead(nn.Module):
    def __init__(self, in_channels=512, num_anchors=2, fpn_num=3):
        super().__init__()
        self.class_head = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_anchors * 2,
                kernel_size=(1, 1),
                stride=1,
                padding=0
            )
            for _ in range(fpn_num)
        ])

    def forward(self, x):
        outputs = []
        for feature, layer in zip(x, self.class_head):
            outputs.append(layer(feature).permute(0, 2, 3, 1).contiguous())

        outputs = torch.cat([out.view(out.shape[0], -1, 2) for out in outputs], dim=1)
        return outputs


class BboxHead(nn.Module):
    def __init__(self, in_channels=512, num_anchors=2, fpn_num=3):
        super().__init__()
        self.bbox_head = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_anchors * 4,
                kernel_size=(1, 1),
                stride=1,
                padding=0
            )
            for _ in range(fpn_num)
        ])

    def forward(self, x):
        outputs = []
        for feature, layer in zip(x, self.bbox_head):
            outputs.append(layer(feature).permute(0, 2, 3, 1).contiguous())
        
        outputs = torch.cat([out.view(out.shape[0], -1, 4) for out in outputs], dim=1)
        return outputs

class LandmarkHead(nn.Module):
    def __init__(self, in_channels=512, num_anchors=2, fpn_num=3):
        super().__init__()
        self.landmark_head = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_anchors * 10,
                kernel_size=(1, 1),
                stride=1,
                padding=0
            )
            for _ in range(fpn_num)
        ])

    def forward(self, x):
        outputs = []
        for feature, layer in zip(x, self.landmark_head):
            outputs.append(layer(feature).permute(0, 2, 3, 1).contiguous())

        outputs = torch.cat([out.view(out.shape[0], -1, 10) for out in outputs], dim=1)
        return outputs
