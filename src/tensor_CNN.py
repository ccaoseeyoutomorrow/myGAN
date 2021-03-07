import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.utils import np_utils
from 手写CNN import loadmnist





def main():
    x_train, y_train, x_test, y_test = loadmnist()
    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
    y_train = np_utils.to_categorical(y_train, num_classes=10)
    y_test = np_utils.to_categorical(y_test, num_classes=10)

    model = keras.Sequential()
    model.add(keras.Input(shape=(28, 28, 1)))  # 250x250 RGB images
    model.add(layers.Convolution2D(  # 第一层卷积(28*28)
        filters=32,
        kernel_size=5,
        strides=1,
        padding='same',
        activation='relu'
    ))
    # model.add(layers.MaxPooling2D(  # 第一层池化(14*14),相当于28除以2
    #     pool_size=2,
    #     strides=2,
    #     padding='same'
    # ))
    model.add(layers.Flatten()

    )  # 把池化层的输出扁平化为一维数据
    model.add(Dense(32, activation='relu'))  # 第一层全连接层
    model.add(Dense(10, activation='softmax'))  # 第二层全连接层
    model.summary()

    model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
    model.fit(x=x_train,y=y_train, batch_size=200, epochs=10)
    result = model.evaluate(x_test, y_test)
    print('TEST ACC:', result[1])

if __name__ == '__main__':
    main()
