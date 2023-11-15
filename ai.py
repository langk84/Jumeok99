import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import FinanceDataReader as fdr
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Lambda
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os.path
from os import path
warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'NanumGothic'

while True:
    if(path.exists("1.check")):
        f=open('C:/person data/project/require.txt','r')
        s3= f.readlines()
        print(s3[0][0:-1])
        print(s3[1])
        print('---')
        STOCK_CODE = s3[0][0:-1]
        epcoin = s3[1]
        if(STOCK_CODE=='undefined' or epcoin=='undefined'):
            os.remove('1.check')
            continue
        stock = fdr.DataReader(STOCK_CODE)
        stock['Year'] = stock.index.year
        stock['Month'] = stock.index.month
        stock['Day'] = stock.index.day

        scaler = MinMaxScaler()
        # 스케일을 적용할 column을 정의합니다.
        scale_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        # 스케일 후 columns
        scaled = scaler.fit_transform(stock[scale_cols])

        df = pd.DataFrame(scaled, columns=scale_cols)

        x_train, x_test, y_train, y_test = train_test_split(df.drop('Close', 1), df['Close'], test_size=0.2, random_state=0, shuffle=False)

        def windowed_dataset(series, window_size, batch_size, shuffle):
            series = tf.expand_dims(series, axis=-1)
            ds = tf.data.Dataset.from_tensor_slices(series)
            ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
            ds = ds.flat_map(lambda w: w.batch(window_size + 1))
            if shuffle:
                ds = ds.shuffle(1000)
            ds = ds.map(lambda w: (w[:-1], w[-1]))
            return ds.batch(batch_size).prefetch(1)

        WINDOW_SIZE=20
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

        plt.figure(figsize=(12, 9))
        plt.plot(np.asarray(y_test)[20:], label='actual')
        plt.plot(pred, label='prediction')
        plt.legend()
        filesave = os.path.join('public',STOCK_CODE+','+epcoin+'.png')
        plt.savefig(filesave)
        os.remove('1.check')
        print()
        print('finish'+STOCK_CODE+epcoin)