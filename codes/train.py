from model_loader import *
import time
from tqdm import tqdm
from utils import dice
from torch.utils.tensorboard import SummaryWriter

############---조작할 부분---###############
start_epoch = 0              
end_epoch = 1200
model_name = 'UNet_resnet34_cs' # 모델 바뀌면 꼭 수정하기
freeze = False # transfer를 위해서 가중치 고정할 경우만 나두기
threshold = 0.35
alpha = 5
#########################################

# 사전 훈련된 가중치 가져오기
weight_file = f'./weights/{model_name}_{start_epoch}.pth'
if start_epoch != 0:
    net_weights = torch.load( weight_file, map_location={'cuda:0': 'cpu'})
    model.load_state_dict(net_weights)
    print('네트워크 설정 완료: 학습된 가중치를 로드했습니다')
    
print("사용 중인 장치:", device)

# TensorBoard 설정
current_time = time.strftime("%m-%d-%H:%M", time.localtime())
log_dir = f'./logs/{model_name}_lr/{current_time}'  # TensorBoard 로그 디렉토리 경로
writer = SummaryWriter(log_dir=log_dir)

# 네트워크가 어느 정도 고정되면, 고속화시킨다
torch.backends.cudnn.benchmark = True

logs = []

if freeze:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    
# train loop
for epoch in range(start_epoch, end_epoch):
    # 반복자의 카운터 설정
    iteration = 0
    val_iteration = 1
    epoch_train_loss = 0.0  # epoch의 손실합
    epoch_val_loss = 0.0  # epoch의 손실합
    epoch_train_dice_score = 0.0
    epoch_val_dice_score = 0.0
    total_samples = 0
    
    print('-----------------------------------------------------------------------------------')

    # 어차피 매 에폭마다 train, val 해야하므로 구분하는 코드를 없애고 아래처럼 train for문, val for문 두개로 나눴다
    model.train()  # 모델을 훈련모드로
    print('Epoch {}/{} (train)'.format(epoch+1, end_epoch))
    iterator_train = tqdm(dataloaders_dicts['train'])
    for images, masks in iterator_train:
        images = images.float().to(device)
        masks = masks.float().to(device)
        # if phase == 'train':
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks.unsqueeze(1))
        sigmoid_outputs = torch.sigmoid(outputs)  # 시그모이드 함수 적용하여 확률 값으로 변환
        thresholded_outputs = torch.where(sigmoid_outputs >= threshold, 1, 0)  # threshold 기준으로 값을 수정
        dice_score = dice(masks.unsqueeze(1), thresholded_outputs)
        epoch_train_dice_score += dice_score
        # loss += (1 - dice_score)
        epoch_train_loss += loss.item()
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), clip_value=2.0)
        optimizer.step()
        iterator_train.set_description('Loss: {:.4f} || Dice_score: {:.4f}'.format(
            loss.item(), dice_score))
        iteration += 1
    epoch_train_loss /= iteration
    
    epoch_train_dice_score /= iteration
    # epoch_val_dice_score /= val_iteration
    print('Epoch_TRAIN_avg_Loss:{:.4f} || Train_dice_score_avg:{:.4f}'.format(epoch_train_loss, epoch_train_dice_score))
    # TensorBoard에 기록
    writer.add_scalar('Loss/Train', epoch_train_loss, epoch)
    writer.add_scalar('Dice Score/Train', epoch_train_dice_score, epoch)
    
    # 로그를 저장
    log_epoch = {'epoch': epoch+1,
                    'train_loss': epoch_train_loss,
                    'train_dice_score': epoch_train_dice_score.item()}
    logs.append(log_epoch)
    df = pd.DataFrame(logs)
    df.to_csv(f"./log_csv/log_{model_name}_lr.csv", index = False)
        # else:
        
    if (epoch+1)%10== 0:
        # continue
        model.eval()   # 모델을 검증모드로
        print('Epoch {}/{} (valid)'.format(epoch+1, end_epoch))
        iterator_val = tqdm(dataloaders_dicts['val'])
        for images, masks in iterator_val:
            images = images.float().to(device)
            masks = masks.float().to(device)
            with torch.no_grad():
                outputs = model(images)
                loss = criterion(outputs, masks.unsqueeze(1))
                sigmoid_outputs = torch.sigmoid(outputs)  # 시그모이드 함수 적용하여 확률 값으로 변환
                thresholded_outputs = torch.where(sigmoid_outputs >= threshold, 1, 0)  # threshold 기준으로 값을 수정
                val_dice_score = dice(masks.unsqueeze(1), thresholded_outputs)
                epoch_val_dice_score += val_dice_score
                # loss += 1- val_dice_score
                epoch_val_loss += loss.item()
                iterator_val.set_description('Loss: {:.4f} || Dice_score: {:.4f}'.format(
                loss.item(), val_dice_score))
                val_iteration += 1
        epoch_val_loss /= val_iteration
        epoch_val_dice_score /= val_iteration
        print('Epoch_VAL_avg_Loss:{:.4f} || Val_dice_score_avg:{:.4f}'.format(epoch_val_loss, epoch_val_dice_score))
        writer.add_scalar('Loss/Val', epoch_val_loss, epoch)
        writer.add_scalar('Dice Score/Val', epoch_val_dice_score, epoch)
        log_epoch = {'epoch': epoch+1,
                    'val_loss': epoch_val_loss,
                    'val_dice_score' : epoch_val_dice_score.item()}
        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        df.to_csv(f"./log_csv/log_{model_name}.csv", index = False)
        
    # epoch의 phase 당 loss와 정답률
    # epoch_train_loss /= iteration
    
    # epoch_train_dice_score /= iteration
    # # epoch_val_dice_score /= val_iteration
    # print('Epoch_TRAIN_avg_Loss:{:.4f} || Train_dice_score_avg:{:.4f}'.format(epoch_train_loss, epoch_train_dice_score))
    # # TensorBoard에 기록
    # writer.add_scalar('Loss/Train', epoch_train_loss, epoch)
    # writer.add_scalar('Dice Score/Train', epoch_train_dice_score, epoch)
    
    # # 로그를 저장
    # log_epoch = {'epoch': epoch+1,
    #                 'train_loss': epoch_train_loss,
    #                 'train_dice_score': epoch_train_dice_score.item()}
    # logs.append(log_epoch)
    # df = pd.DataFrame(logs)
    # df.to_csv(f"./log_csv/log_{model_name}_lr.csv", index = False)
    
    # 학습률 스케줄링
    # scheduler.step() 

    # 10 epoch마다 가중치 저장
    if ((epoch+1)%10 == 0):
        torch.save(model.state_dict(), f'weights/{model_name}_' + str(epoch+1) + '_lr.pth')
        
#     # SWA를 적용할 때
#     if epoch > swa_start and (epoch - swa_start) % swa_freq == 0:
#         # SWA 모델을 업데이트합니다
#         swa_optimizer.update_swa()

# # SWA로 얻은 평균 가중치로 모델을 초기화합니다
# swa_optimizer.swap_swa_sgd()

# TensorBoard 사용 종료
writer.close()
