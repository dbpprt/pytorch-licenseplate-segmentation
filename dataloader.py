import glob
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchvision.transforms import (Compose, Normalize, Resize, ToPILImage,
                                    ToTensor)


class SegmentationDataset(Dataset):
    def __init__(self, folder_path, transforms):
        super(SegmentationDataset, self).__init__()
        self.images = glob.glob(os.path.join(folder_path, 'images', '*.jpg'))
        self.masks = []
        for image in self.images:
            mask_path = os.path.join(
                folder_path, 'masks', f'{os.path.basename(image)}.png')
            self.masks.append(mask_path)
        self.transforms = transforms

        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        img_path = self.images[index]
        target_path = self.masks[index]

        img = Image.open(img_path).convert('RGB')
        target = Image.open(target_path)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        if len(target.shape) == 2:
            target = target.reshape((1,)+target.shape)
        if len(img.shape) == 2:
            img = img.reshape((1,)+img.shape)

        return img, target

    def __len__(self):
        return len(self.images)
