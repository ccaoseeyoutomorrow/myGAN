from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
import tensorflow as tf
import numpy as np


def read_trainx(filname):
    data=np.load(filname)
    data=np.array(data,dtype='float')
    return data

def read_trainy(length,filename):
    label=np.loadtxt(filename,encoding='utf-8',dtype='int')
    label=np.array(label).reshape(-1)
    temp=np.zeros(shape=(length,400),dtype='int')
    temp[:,0:400]=label
    return temp

def read_testx():
    length=[]
    data1=np.load('../Defect_Data/1号树木x.npy')
    data1=np.array(data1,dtype='float')
    length.append(data1.shape[0])
    data2=np.load('../Defect_Data/2号树木x.npy')
    data2=np.array(data2,dtype='float')
    length.append(data2.shape[0])
    data3=np.load('../Defect_Data/3号树木x.npy')
    data3=np.array(data3,dtype='float')
    length.append(data3.shape[0])
    data=np.concatenate((data1,data2,data3))
    return data,length

def read_testy(length):
    label1=np.loadtxt('../Defect_Data/label1_onlydefect_20.txt',encoding='utf-8',dtype='int')
    label1=np.array(label1).reshape(-1)
    temp1=np.zeros(shape=(length[0],400),dtype='int')
    temp1[:,0:400]=label1
    label2=np.loadtxt('../Defect_Data/label1_onlydefect_20.txt',encoding='utf-8',dtype='int')
    label2=np.array(label2).reshape(-1)
    temp2=np.zeros(shape=(length[1],400),dtype='int')
    temp2[:,0:400]=label2
    label3=np.loadtxt('../Defect_Data/label1_onlydefect_20.txt',encoding='utf-8',dtype='int')
    label3=np.array(label3).reshape(-1)
    temp3=np.zeros(shape=(length[2],400),dtype='int')
    temp3[:,0:400]=label3
    Y=np.concatenate((temp1,temp2,temp3))
    return Y

def loadmnist():
    x_train1=read_trainx('../Defect_Data/3号树木x_median.npy')
    x_train2=read_trainx('../Defect_Data/4号树木x_median.npy')
    x_train=np.concatenate((x_train1,x_train2),axis=0)
    y_train1=read_trainy(x_train1.shape[0],'../Defect_Data/label3_onlydefect_20.txt')
    y_train2=read_trainy(x_train2.shape[0],'../Defect_Data/label4_onlydefect_20.txt')
    y_train=np.concatenate((y_train1,y_train2),axis=0)
    index = [i for i in range(len(x_train))]
    np.random.shuffle(index)
    x_train=x_train[index]
    y_train=y_train[index]
    x_test=x_train[0:100]
    y_test=y_train[0:100]
    x_train=x_train[100:]
    y_train=y_train[100:]
    return x_train,y_train,x_test,y_test


def main():
    # 获取数据
    x_train,y_train,x_test,y_test=loadmnist()



    #进一步处理数据
    # x_test=x_train[0:500]
    # y_test=y_train[0:500]
    # x_train=x_train[500:]
    # y_train=y_train[500:]

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
    model.add(Dense(1440,activation='softplus'))  # 第一层全连接层
    model.add(Dense(720,activation='relu'))  # 第一层全连接层
    config = model.get_config()  # 把model中的信息，solver.prototxt和train.prototxt信息提取出来
    print(config)
    # model = model.from_config(config)  # 还回去
    model.add(Dense(400,activation=tf.sigmoid))  # 第二层全连接层
    model.summary()

    model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
    model.fit(x=x_train,y=y_train, batch_size=200, epochs=2)
    result = model.evaluate(x_test, y_test)
    print('TEST ACC:', result[1])

if __name__ == '__main__':
    main()