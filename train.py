#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2021/03/19
"""
import argparse

import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.face_dataset import FaceIdentityDataset
from model.osnet import osnet_x1_0
from utils.tools import build_optimizer, build_scheduler
import torch.nn as nn


def train(max_epoch, batch_size):
    dataset = FaceIdentityDataset()
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    model = osnet_x1_0(num_classes=dataset.num_classes, pretrained=True, loss='softmax', use_gpu=True)
    model = model.cuda()
    model.train()
    cudnn.benchmark = True
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer, lr_scheduler='cosine', max_epoch=max_epoch)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter()
    for epoch in range(max_epoch):
        train_tqdm = tqdm(train_loader, desc=str(epoch+1) + '/' + max_epoch(max_epoch))
        for index, data in train_tqdm:
            im, label = data
            im = im.cuda()
            label = label.cuda()
            optimizer.zero_grad()
            out = model(im)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            if index % 10 == 0:
                n_iter = epoch*len(train_loader) + index
                writer.add_scalar('loss', loss, n_iter)
        scheduler.step()
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), './weights/net_{}'.format(epoch))
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("人脸识别训练参数")
    parser.add_argument("--epoch", default=100, type=int, help="train epochs")
    parser.add_argument("--batch_size", default=16, type=int, help="train batch size")
    args = parser.parse_args()
    max_epoch = args.epoch
    batch_size = args.batch_size
    train(max_epoch, batch_size)
