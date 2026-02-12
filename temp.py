from dataset.dataset import WiderFaceDataset
from models.backbones.resnet import BasicBlock
import numpy as np
import cv2
import torch
if __name__ == '__main__':
    # train_data = WiderFaceDataset('data/widerface', train=True)
    # image = train_data.__getitem__(0)[0] # H, W, 3
    # image = np.array(image) # scalar -> numpy array
    # cv2.imshow('image', image)
    # cv2.waitKey(0)


    x = torch.randn(1, 64, 56, 56) # B, C, H, W
    block = BasicBlock(64, 64, stride=1)
    y = block(x)
    print(x.shape)
    print(y.shape)