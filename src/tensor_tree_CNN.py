from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
import tensorflow as tf
import numpy as np


def read_trainx():
    data=np.load('../Defect_Data/4号树木x.npy')
    data=np.array(data,dtype='float')
    return data

def read_trainy():
    label=np.loadtxt('../Defect_Data/label4_onlydefect_20.txt',encoding='utf-8',dtype='int')
    label=np.array(label).reshape(-1)
    temp=np.zeros(shape=(5476,400),dtype='int')
    temp[:,0:400]=label
    return temp



def main():
    # 获取数据
    x_train=read_trainx()
    y_train=read_trainy()
    x_test=x_train[0:1000]
    y_test=y_train[0:1000]
    x_train=x_train[1000:]
    y_train=y_train[1000:]

    model = keras.Sequential()
    model.add(keras.Input(shape=(44)))  # 250x250 RGB images
    # model.add(layers.Convolution2D(  # 第一层卷积(28*28)
    #     filters=32,
    #     kernel_size=5,
    #     strides=1,
    #     padding='same',
    #     activation='relu'
    # ))
    # model.add(layers.MaxPooling2D(  # 第一层池化(14*14),相当于28除以2
    #     pool_size=2,
    #     strides=2,
    #     padding='same'
    # ))
    model.add(layers.Flatten()

    )  # 把池化层的输出扁平化为一维数据
    model.add(Dense(1440))  # 第一层全连接层
    model.add(Dense(720))  # 第一层全连接层
    model.add(Dense(400,activation=tf.sigmoid))  # 第二层全连接层
    model.summary()

    model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
    model.fit(x=x_train,y=y_train, batch_size=200, epochs=10)
    result = model.evaluate(x_test, y_test)
    print('TEST ACC:', result[1])

if __name__ == '__main__':
    main()
