import os
import cv2
import pandas as pd
import numpy as np
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2
import math
from PIL import Image

from utils import rle_decode, rle_encode
from cutmix import *

class SatelliteDataset(Dataset):
    def __init__(self, csv_file, transform=None, infer=False):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.infer = infer
        
        self.crop_coords = []

        self.re_size = 224
        self.original_size = 1024
        num = math.ceil(self.original_size/self.re_size)
        x_coord = 0
        y_coord = 0
        for y in range(num):
            if y==num-1:
                y_coord -= 96
            x_coord = 0
            for x in range(num):
                if x==num-1:
                    x_coord -= 96
                self.crop_coords.append((x_coord, y_coord, x_coord + self.re_size, y_coord + self.re_size))
                x_coord += self.re_size
            y_coord += self.re_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        if isinstance(idx,slice): # 슬라이싱 처리는 현재 모델인 UNet_resnet34_cutmix_original_loss_SWA_th0.35에서는 안쓰이고 있다
            df_slice = pd.DataFrame(columns=['img_id','img_path','mask_rle'])
            # if idx.start == None:
            #     slice_type = 'train'
            #     start_idx = 0
            #     stop_idx = idx.stop
            # else :
            #     slice_type = 'val'
            #     start_idx = idx.start
            #     stop_idx = self.__len__()
                
            if idx.start == None:
                slice_type = 'one'
                start_idx = 0
                stop_idx = idx.stop
            elif idx.stop == None :
                slice_type = 'three'
                start_idx = idx.start
                stop_idx = self.__len__()
            else :
                slice_type = 'two'
                start_idx = idx.start
                stop_idx = idx.stop
            
            for each_idx in range(start_idx, stop_idx):
                each_img_path = self.data.iloc[each_idx]
                # each_mask_rle = each_img_path['mask_rle'] if not (slice_type=='val' and each_img_path['mask_rle'] =='') else -1
                # df_slice.loc[each_idx-start_idx] = [each_img_path['img_id'], each_img_path['img_path'], each_mask_rle]
                
            slice_path = './data/train_'+slice_type+'.csv'
            df_slice.to_csv(slice_path, index=False)
            return SatelliteDataset(slice_path, transform=self.transform, infer=self.infer)
        
        img_path = self.data.iloc[idx, 1]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.infer:
            if self.transform:
                image = self.transform(image=image)['image']
            return image

        mask_rle = self.data.iloc[idx, 2]
        mask = rle_decode(mask_rle, (image.shape[0], image.shape[1]))

        if self.transform:
            #### cut&origin ####
            # if random.random() >= 0.5:
            if idx < self.__len__()/2:
            ####################
                img2_idx = random.randint(0, len(self.data)-1)
                img2_path = self.data.iloc[img2_idx, 1]
                mask2_rle = self.data.iloc[img2_idx, 2]
                image2 = cv2.imread(img2_path)
                image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
                mask2 = rle_decode(mask2_rle, (image2.shape[0], image2.shape[1]))
                image, mask = cutmix(image, image2, mask, mask2)
                # cutmix_image = Image.fromarray(image)
                # plt.subplot(1, 2, 1)
                # plt.imshow(cutmix_image)
                # plt.axis('off')
                # plt.subplot(1, 2, 2)
                # plt.imshow(mask, cmap='gray')
                # plt.axis('off')
                # plt.show()
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask
    