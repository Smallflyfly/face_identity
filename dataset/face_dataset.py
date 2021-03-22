#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2021/03/19
"""
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FaceIdentityDataset(Dataset):
    def __init__(self):
        super(FaceIdentityDataset, self).__init__()
        self.data_path = './data/train'
        self.train_images = []
        self.labels = []
        self.num_classes = 9131
        self.image_size = (214, 214)
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.6068, 0.4517, 0.3800], [0.2492, 0.2173, 0.2082])
            ]
        )
        self._process_data()

    def _process_data(self):
        folder_list = os.listdir(self.data_path)
        num = 0
        for folder in folder_list:
            image_files = os.listdir(os.path.join(self.data_path, folder))
            for image in image_files:
                full_image = os.path.join(os.path.join(self.data_path, folder), image)
                self.train_images.append(full_image)
                self.labels.append(num)
            num += 1

    def __getitem__(self, index):
        image = self.train_images[index]
        label = self.labels[index]
        im = Image.open(image)
        im = self.transform(im)
        return im, label

    def __len__(self):
        return len(self.train_images)


