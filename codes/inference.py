import os
import cv2
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101

from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils import *
from dataloader import *
from model_loader import *

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform_test = A.Compose(
    [   
        A.Resize(224, 224),
        A.Normalize(),
        ToTensorV2(),
    ]
)
    
test_dataset = SatelliteDataset(csv_file='./data/test.csv', transform=transform_test, infer=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

#####################################################    
name_weights = 'UNet_resnet34_cs_260_lr'
#####################################################

net_weights = torch.load(f'./weights/{name_weights}.pth',
                          map_location={'cuda:0': 'cpu'})
model.load_state_dict(net_weights)
with torch.no_grad():
    model.eval()
    result = []
    for images in tqdm(test_dataloader):
        images = images.float().to('cuda:0')
        
        outputs = model(images)
        masks = torch.sigmoid(outputs).to('cuda:0').cpu().numpy()
        masks = np.squeeze(masks, axis=1)
        masks = (masks > 0.35).astype(np.uint8) # Threshold = 0.35
        
        for i in range(len(images)):
            mask_rle = rle_encode(masks[i])
            if mask_rle == '': # 예측된 건물 픽셀이 아예 없는 경우 -1
                result.append(-1)
            else:
                result.append(mask_rle)

submit = pd.read_csv('./submit/sample_submission.csv')
submit['mask_rle'] = result

submit.to_csv(f'./submit/submit_{name_weights}.csv', index=False) 