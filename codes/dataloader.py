from custom_dataset import *

transform = A.Compose(
    [   
        A.Rotate((-90,90)),
        A.RandomCrop(224, 224),
        # A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5, p=0.75),
        # A.Solarize(p=0.1), # 밝기 명암뒤집기 = 128을 넘을 수 없게, 넘었을 때 (256 - 원래 픽셀)로 바꿔버림.
        # A.Sharpen(p=0.5), # 날카로운 경계를 강조
        # A.RingingOvershoot(p=0.3),
        # A.Emboss(0.3), # 입체효과
        # A.GaussNoise(),
        # A.RandomFog(p=0.1), # 안개효과
        # A.RandomSunFlare(p=0.2), # 태양빛 반사효과
        # A.RandomShadow(p=0.1), # 그림자만드는 효과
        # A.Equalize(),
        # A.UnsharpMask(p=0.2), # 이미지 선명도 향상, 블러 처리된 부분 자동 복원
        A.Normalize(),
        ToTensorV2(),
    ]
)


transform_test = A.Compose(
    [   
        #A.Resize(224, 224),
        A.RandomCrop(224,224),
        A.Normalize(),
        ToTensorV2(),
    ]
)

####################################################################
split = True
train_val_rate = 0.2
####################################################################

if split:
    train = pd.read_csv("./data/train.csv")
    val = train.sample(int(len(train)*train_val_rate), random_state=1)
    train = train.drop(val.index)
    train = pd.concat([train, train], ignore_index=True)

    train.to_csv("./data/train_train.csv",index=False)
    val.to_csv("./data/train_valid.csv", index=False)


train_dataset = SatelliteDataset(csv_file='./data/train_train.csv', transform=transform)
val_dataset = SatelliteDataset(csv_file='./data/train_valid.csv', transform=transform_test)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

dataloaders_dicts = {'train':train_dataloader, 'val':val_dataloader}



# if split:
    # 전체적인 설명 
    # rate가 0.8이라 하면 데이터를 0.2 0.6 0.2 로 나누어 one, two, three 라고 저장하고
    # train, val 데이터가 매 에폭 마다 번갈아 가므로 아래처럼 even용, odd용으로 구분한다.
    # even_train = one+two, even_val = three
    # odd_train = two+three, odd_val = one
    # train에 double이 붙은 이유는 cutmix&original을 하기 위해서 이다.
    # double만 해주면 custom_dataset.py의 97 line에서 알아서 cutmix&original을 해준다.
    # 또한 맨 아래줄을 보면 dataloaders_dicts를 dataloaders_dicts_list로 바꾸었는데 
    # train.py에서 iterator 설정을 매 에폭 마다 다르게 하기 위함이다.

    # train = pd.read_csv("./data/train.csv")

#     train_size_first = int(len(train) * (1 - train_val_rate))
#     train_size_second = int(len(train) * train_val_rate)
    
#     train_one = train[:train_size_first]
#     train_two = train[train_size_first:train_size_second]
#     train_three = train[train_size_second:]
    
#     train_even_train = pd.concat([train_one, train_two], ignore_index=True)
#     train_even_train_double = pd.concat([train_even_train, train_even_train], ignore_index=True)
#     train_even_train_double.to_csv("./data/train_even_train_double.csv", index=False)
#     train_three.to_csv("./data/train_even_val.csv", index=False)
    
#     train_odd_train = pd.concat([train_two, train_three], ignore_index=True)
#     train_odd_train_double = pd.concat([train_odd_train, train_odd_train], ignore_index=True)
#     train_odd_train_double.to_csv("./data/train_odd_train_double.csv", index=False)
#     train_one.to_csv("./data/train_odd_val.csv", index=False)
   

# train_even_dataset = SatelliteDataset(csv_file='./data/train_even_train_double.csv', transform=transform)
# val_even_dataset = SatelliteDataset(csv_file='./data/train_even_val.csv', transform=transform_test)
# train_odd_dataset = SatelliteDataset(csv_file='./data/train_odd_train_double.csv', transform=transform)
# val_odd_dataset = SatelliteDataset(csv_file='./data/train_odd_val.csv', transform=transform_test)

# train_even_dataloader = DataLoader(train_even_dataset, batch_size=32, shuffle=True, num_workers=4)
# val_even_dataloader = DataLoader(val_even_dataset, batch_size=32, shuffle=False, num_workers=4)
# train_odd_dataloader = DataLoader(train_odd_dataset, batch_size=32, shuffle=True, num_workers=4)
# val_odd_dataloader = DataLoader(val_odd_dataset, batch_size=32, shuffle=False, num_workers=4)

# dataloaders_dicts_list = [{'train':train_even_dataloader, 'val':val_even_dataloader}, {'train':train_odd_dataloader, 'val':val_odd_dataloader}]