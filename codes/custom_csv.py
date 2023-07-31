import pandas as pd

def change_column_value(path):
    df = pd.read_csv('./data/test.csv')  # 데이터프레임을 './data.csv' 파일에서 읽어옴
    df['img_path'] = path + df['img_path'].apply(lambda x: x[1:])  # 해당 컬럼의 값을 변경하여 업데이트
    df.to_csv('./data/test.csv', index=False)  # 업데이트된 데이터프레임을 './data/data.csv' 파일에 저장
#change_column_value('./data')  

def train_change_column_value(path):
    df = pd.read_csv('./data/train.csv')  # 데이터프레임을 './data.csv' 파일에서 읽어옴
    df['img_path'] = path + df['img_path'].apply(lambda x: x[1:])  # 해당 컬럼의 값을 변경하여 업데이트
    df.to_csv('./data/train.csv', index=False)  # 업데이트된 데이터프레임을 './data/data.csv' 파일에 저장

train_change_column_value('./data') 