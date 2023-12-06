import pandas as pd # 데이터 조작과 분석
import numpy as np # 다차원 배열과 행렬 연산
import matplotlib.pyplot as plt # 데이터 시각화, 그래프 생성
import FinanceDataReader as fdr #금융 데이터 수집
from sklearn.preprocessing import MinMaxScaler #[0, 1] 범위로 스케일 조정
from sklearn.model_selection import train_test_split #데이터를 학습용과 테스트용으로 분할
import tensorflow as tf #딥러닝 및 기계 학습 모델을 구축
from tensorflow.keras.models import Sequential #각 레이어를 차례로 쌓아 간단한 신경망 모델 생성
from tensorflow.keras.layers import Dense #입력과 출력 간의 모든 뉴런이 서로 연결된 완전 연결 레이어를 생성
from tensorflow.keras.layers import LSTM #Long Short-Term Memory 시퀀스 데이터에서 장기 의존성을 효과적으로 모델링하는 데 사용되는 순환 신경망
from tensorflow.keras.layers import Conv1D #시계열 데이터나 텍스트와 같은 1차원 입력에 대한 특징 추출 및 패턴 학습
from tensorflow.keras.losses import Huber #휴버 손실(Huber loss) 계산
from tensorflow.keras.optimizers import Adam #신경망 학습 최적화 알고리즘(경사 하강법의 변형)
from tensorflow.keras.callbacks import EarlyStopping #머신러닝 모델의 훈련을 조기에 중단하는 기능을 제공
from tensorflow.keras.callbacks import ModelCheckpoint #훈련 중간에 모델의 성능이나 상태를 저장
import os.path #파일 및 디렉토리 경로 이용
from os import path #파일 및 디렉토리 경로를 조작
import warnings #경고관리 및 제어
warnings.filterwarnings('ignore') #실행중 경고 무시
plt.rcParams['font.family'] = 'NanumGothic' #그래프에 사용되는 글꼴 정의
#라이브러리 로드

maxepcoin = 50

while True: #항상 켜짐
    if(path.exists("check.i")): #서버에서 require.txt를 업데이트할때 1.check(반복flag)와 함께 생성
        os.remove('check.i') #반복flag 회수
        f=open('C:/person data/project/require.txt','r') #require.txt 파일 열기
        orgdata= f.readlines()
        STOCK_CODE = orgdata[0][0:-1] #첫번째줄, \n 제거: 주식코드
        epcoin = orgdata[1] #두번째줄: 학습횟수
        if(STOCK_CODE=='undefined' or epcoin=='undefined'): #서버 이상데이터 수신시, 상시예외데이터: monitor대상아님
            continue #이번 반복은 넘김

        print(STOCK_CODE, epcoin)  # monitor
        if(int(epcoin)<1 or int(epcoin)>maxepcoin):
            print('Invalid learning times')
            continue
        if(len(STOCK_CODE)!=4 and len(STOCK_CODE)!=6):
            print('Not stock code')
            continue
        if(path.exists("./public/"+STOCK_CODE+','+epcoin+'.png')):
            print('Already data') #monitor
            continue

        stock = fdr.DataReader(STOCK_CODE) #주식데이터 FinanceDataReader를 이용해 불러옴
        stock['Year'] = stock.index.year
        stock['Month'] = stock.index.month
        stock['Day'] = stock.index.day

        scaler = MinMaxScaler() #정규화(값을 0~1사이로 변경)
        # 스케일을 적용할 column을 정의
        scale_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        # 스케일 후 columns
        scaled = scaler.fit_transform(stock[scale_cols])

        df = pd.DataFrame(scaled, columns=scale_cols)

        x_train, x_test, y_train, y_test = train_test_split(
            df.drop('Close', 1),
            df['Close'],
            test_size=0.2,
            random_state=0,
            shuffle=False)

        def windowed_dataset(series, window_size, batch_size, shuffle):
            series = tf.expand_dims(series, axis=-1)
            ds = tf.data.Dataset.from_tensor_slices(series)
            ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
            ds = ds.flat_map(lambda w: w.batch(window_size + 1))
            if shuffle:
                ds = ds.shuffle(1000)
            ds = ds.map(lambda w: (w[:-1], w[-1]))
            return ds.batch(batch_size).prefetch(1)

        WINDOW_SIZE=20 #분석 분할크기
        BATCH_SIZE=32

        train_data = windowed_dataset(y_train, WINDOW_SIZE, BATCH_SIZE, True)
        test_data = windowed_dataset(y_test, WINDOW_SIZE, BATCH_SIZE, False)

        model = Sequential([
            # 1차원 feature map 생성
            Conv1D(filters=32, kernel_size=5,
                   padding="causal",
                   activation="relu",
                   input_shape=[WINDOW_SIZE, 1]),
            # LSTM
            LSTM(16, activation='tanh'),
            Dense(16, activation="relu"),
            Dense(1),
        ])

        loss = Huber()
        optimizer = Adam(0.0005)
        model.compile(loss=Huber(), optimizer=optimizer, metrics=['mse'])

        # earlystopping은 10번 epoch통안 val_loss 개선이 없다면 학습을 멈춥니다.
        earlystopping = EarlyStopping(monitor='val_loss', patience=10)
        # val_loss 기준 체크포인터도 생성합니다.
        filename = os.path.join('tmp', STOCK_CODE+','+epcoin+'.ckpt')

        checkpoint = ModelCheckpoint(filename,
                                     save_weights_only=True,
                                     save_best_only=True,
                                     monitor='val_loss',
                                     verbose=1)

        history = model.fit(train_data,
                            validation_data=(test_data),
                            epochs=int(epcoin),
                            callbacks=[checkpoint, earlystopping])

        model.load_weights(filename)
        pred = model.predict(test_data)

        #출력부
        plt.figure(figsize=(12, 9))
        plt.plot(np.asarray(y_test)[80:], label='실제값')
        plt.plot(pred, label='예측값')
        plt.legend()
        filesave = os.path.join('public',STOCK_CODE+','+epcoin+'.png')
        plt.savefig(filesave)
        print('finish '+STOCK_CODE+' '+epcoin) #monitor