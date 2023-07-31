# import pandas as pd

# data = pd.read_csv('./data/train_train.csv')
# data1 = pd.read_csv('./data/train.csv')
# # print(data)
# for i in range(len(data)):
#     if isinstance(data['mask_rle'][i],float):
#         print(data['mask_rle'][i])
        
import tensorflow as tf

# GPU가 사용 가능한지 확인
if tf.test.gpu_device_name():
    print('GPU device found')
else:
    print("No GPU found")