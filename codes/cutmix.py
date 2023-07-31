import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def rand_bbox(size, lam = 0.8):
    W = size[1] # W x H 
    H = size[0]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(cut_w//2, W - cut_w//2)
    cy = np.random.randint(cut_h//2, H - cut_h//2)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(origin_img, refer_img, origin_mask, refer_mask):
    size = origin_img.shape

    origin_np = np.array(origin_img)
    refer_np = np.array(refer_img)

    # bounding box 계산
    bbx1_origin, bby1_origin, bbx2_origin, bby2_origin = rand_bbox(size)
    bbx1_refer, bby1_refer, bbx2_refer, bby2_refer = rand_bbox(size)

    # origin 이미지의 bounding box 영역을 refer 이미지로 대체
    origin_np[bbx1_origin:bbx2_origin, bby1_origin:bby2_origin] = refer_np[bbx1_refer:bbx2_refer, bby1_refer:bby2_refer]
    origin_mask[bbx1_origin:bbx2_origin, bby1_origin:bby2_origin] = refer_mask[bbx1_refer:bbx2_refer, bby1_refer:bby2_refer]
    # Cutmix된 이미지 생성
    cutmix_image = Image.fromarray(origin_np)

    return origin_np, origin_mask


# 전역변수
# W = 1024
# H = 1024
# lam = 0.8


# origin = Image.open('./data/train_img/TRAIN_0000.png')
# refer = Image.open('./data/train_img/TRAIN_0001.png')

# # origin = origin.resize((W, H))
# # refer = refer.resize((W, H))

# cutmix_image = cutmix(origin, refer)

# plt.subplot(1, 3, 1)
# plt.imshow(origin)
# plt.axis('off')

# plt.subplot(1, 3, 2)
# plt.imshow(refer)
# plt.axis('off')

# plt.subplot(1, 3, 3)
# plt.imshow(cutmix_image)
# plt.axis('off')

# plt.show()