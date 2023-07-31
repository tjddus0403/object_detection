from dataloader import *
from model import *
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torchcontrib.optim import SWA

# 사용 장치 선언
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 선언
model = UNet_ResNet34(num_classes=1)
mode = model.to(device)

# loss function과 optimizer 정의
# pos_weight = torch.tensor([2.0])
# pos_weight = pos_weight.to(device)+
# criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

# 학습률(lr) 스케줄러 설정
# scheduler = CosineAnnealingLR(optimizer, T_max=150, eta_min=0.00001)

# SWA optimizer를 생성합니다
# swa_start = 300
# swa_freq = 10
# swa_optimizer = SWA(optimizer, swa_start=swa_start, swa_freq=swa_freq)