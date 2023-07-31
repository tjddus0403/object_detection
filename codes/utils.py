import numpy as np
import torch

# RLE 디코딩 함수
def rle_decode(mask_rle, shape):
    if isinstance(mask_rle, float) :
        return np.zeros(shape[0]*shape[1], dtype=np.int8).reshape(shape)
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

# RLE 인코딩 함수
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def dice(y_true, y_pred, epsilon=1e-7):
    y_true_flat = y_true  # 2차원 행렬을 1차원으로 변환
    y_pred_flat = y_pred  # 2차원 행렬을 1차원으로 변환

    intersection = torch.sum(y_true_flat * y_pred_flat)  # 1이 겹치는 개수의 합
    union = torch.sum(y_true_flat) + torch.sum(y_pred_flat) + epsilon  # 정답 행렬의 1의 개수 + 예측 행렬의 1의 개수 + 입실론

    dice = (2 * intersection + epsilon) / union  # Dice 계수 계산

    return dice