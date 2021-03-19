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
        self.data_path = ''
        self.train_images = []
        self.labels = []
        self.num_classes = []
        self.image_size = (320, 320)
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((), ())
            ]
        )

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
        im = Image.Open(image)
        im = self.transform(im)
        return im, label

    def __len__(self):
        return len(self.train_images)


