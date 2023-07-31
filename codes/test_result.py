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
import matplotlib.pyplot as plt
from utils import rle_decode, rle_encode
# num = 0
test_df = pd.read_csv('./submit/submit_UNet_resnet34_cutmix&original_750_th0.5.csv')
test_df2 = pd.read_csv('./submit/submit_UNet_resnet34_cutmix&original_750.csv')
test_df3 = pd.read_csv('./submit/submit_UNet_resnet34_cutmix&original_750_th0.2.csv')
for i in range(50):
    rand_num = random.randint(0, len(test_df)-1)
    num = '{:05d}'.format(rand_num)
    print(num)
    # img = Image.open('./data/test_img/TEST_' + num + '.png')
    img = cv2.imread('./data/test_img/TEST_' + num + '.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask_rle = test_df.iloc[rand_num, 1]
    mask = rle_decode(mask_rle, (img.shape[0], img.shape[1]))
    img2 = cv2.imread('./data/test_img/TEST_' + num + '.png')
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    mask_rle2 = test_df2.iloc[rand_num, 1]
    mask2 = rle_decode(mask_rle2, (img2.shape[0], img2.shape[1]))
    img3 = cv2.imread('./data/test_img/TEST_' + num + '.png')
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    mask_rle3 = test_df3.iloc[rand_num, 1]
    mask3 = rle_decode(mask_rle3, (img3.shape[0], img3.shape[1]))
    plt.subplot(3, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.subplot(3, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    plt.subplot(3, 2, 3)
    plt.imshow(img2)
    plt.axis('off')
    plt.subplot(3, 2, 4)
    plt.imshow(mask2, cmap='gray')
    plt.axis('off')
    plt.subplot(3, 2, 5)
    plt.imshow(img3)
    plt.axis('off')
    plt.subplot(3, 2, 6)
    plt.imshow(mask3, cmap='gray')
    plt.axis('off')
    plt.show()