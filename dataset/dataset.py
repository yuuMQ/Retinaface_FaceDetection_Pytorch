import os
import cv2
import numpy as np

import torch
from PIL import Image
from torch.utils.data import Dataset


'''
The format of txt ground truth.
File name
Number of bounding box
x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose
'''

class WiderFaceDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        super(WiderFaceDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.train = train
        if train:
            self.root = os.path.join(root, 'train')
        else:
            self.root = os.path.join(root, 'val')

        self.image_paths = []
        self.bounding_boxes = []
        self._parse_labels()

    def _parse_labels(self):
        with open(os.path.join(self.root, 'label.txt'), 'r') as f:
            lines = f.read().splitlines()

        labels = []
        for line in lines:
            if line.startswith('#'):
                if labels:
                    self.bounding_boxes.append(labels.copy())
                    labels.clear()
                image_path = os.path.join(self.root, 'images', line[2:])
                self.image_paths.append(image_path)
            else:
                labels.append([float(x) for x in line.split(' ')])
            self.bounding_boxes.append(labels)


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index])
        height, width, _ = image.shape
        labels = self.bounding_boxes[index]

        annotations = np.zeros((0, 15))

        if not labels:
            image = torch.from_numpy(image)
            return image, annotations

        for label in labels:
            annotation = np.zeros((1, 15))
            # bbox (x1, x2, y1, y2)
            annotation[0, 0] = label[0] # x1
            annotation[0, 1] = label[1] # y1
            annotation[0, 2] = label[0] + label[2] # x1 + w -> x2
            annotation[0, 3] = label[1] + label[3] # y1 + h -> y2

            # Landmarks - With repo data format
            annotation[0, 4] = label[4]   # 10_x
            annotation[0, 5] = label[5]   # 10_y
            annotation[0, 6] = label[7]   # 11_x
            annotation[0, 7] = label[8]   # 11_y
            annotation[0, 8] = label[10]   # 12_x
            annotation[0, 9] = label[11]   # 12_y
            annotation[0, 10] = label[13] # 13_x
            annotation[0, 11] = label[14] # 13_y
            annotation[0, 12] = label[16] # 14_x
            annotation[0, 13] = label[17] # 14_y

            if label[4] >= 0:
                annotation[0, 14] = 1
            else:
                annotation[0, 14] = -1

            annotations = np.append(annotations, annotation, axis=0)

        target = np.array(annotations)
        if self.transform:
            image, target = self.transform(image, target)

        image = torch.from_numpy(image)
        return image, target

    @staticmethod
    def collate_fn(batch):
        '''
            collate_fn: cách ghép các sample thành 1 batch
        '''
        images = []
        targets = []
        for image, target in batch:
            images.append(image)
            targets.append(torch.from_numpy(target).float())

        return torch.stack(images, 0), targets


if __name__ == '__main__':

    train_data = WiderFaceDataset('../data/widerface', train=True)
    image = train_data.__getitem__(0)[0] # H, W, 3
    image = np.array(image) # scalar -> numpy array
    cv2.imshow('image', image)
    cv2.waitKey(0)