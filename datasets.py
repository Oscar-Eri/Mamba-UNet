#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
import transfomer
import numpy as np
import matplotlib.pyplot as plt


class SRDataset(Dataset):
    """
    数据集加载器
    """

    def __init__(self, data_list, split, crop_size, img_type):
        """
        :参数 data_folder: # 数据列表
        :参数 split: 'train' 或者 'test'
        :参数 crop_size: 裁剪尺寸
        ：参数 img_type 目标的数据类型
        """

        self.split = split.lower()
        self.img_type = img_type
        assert self.split in {'train', 'test'}

        # 读取图像路径
        if self.split == 'train':
            self.images = data_list
            self.transform = transfomer.ImageTransforms((crop_size, crop_size), img_type=self.img_type)
        else:
            self.images = data_list
            self.transform = transfomer.ImageTransforms((crop_size, crop_size), 1.0, 1.0, self.img_type)

    def __getitem__(self, i):
        """
        为了使用PyTorch的DataLoader，必须提供该方法.

        :参数 i: 图像检索号
        :返回: 返回第i个低分辨率和高分辨率的图像对
        """
        # 读取图像
        underwater_name = self.images[i]

        underwater_img = Image.open(underwater_name).convert('RGB')
        gt_name = underwater_name.replace("underwater", "gt")
        gt_img = Image.open(gt_name).convert('RGB')

        # 删除深度图相关代码
        assert underwater_img.width == gt_img.width and underwater_img.height == gt_img.height

        if self.transform is not None:
            # 仅处理两个图像
            underwater_img, gt_img = self.transform(underwater_img, gt_img)

        return underwater_img, gt_img  # 仅返回两个图像

    def __len__(self):
        """
        为了使用PyTorch的DataLoader，必须提供该方法.

        :返回: 加载的图像总数
        """
        return len(self.images)