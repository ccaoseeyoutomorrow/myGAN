import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.layers import Dense, concatenate
from tensorflow.python.keras.utils import np_utils
from src.手写CNN import loadmnist

x_train, y_train, x_test, y_test = loadmnist()
x_train = x_train.reshape(-1, 784) / 255.0
x_test = x_test.reshape(-1, 784) / 255.0
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

x_train1=x_train[0:30000]
x_train2=x_train[30000:60000]
y_train1=y_train[0:30000]
y_train2=y_train[30000:60000]

x_test1=x_test[0:5000]
x_test2=x_test[5000:10000]
y_test1=y_test[0:5000]
y_test2=y_test[5000:10000]

# define two sets of inputs
inputA = Input(shape=(784,))
inputB = Input(shape=(784,))
# 训练的第一个分支
x = Dense(64, activation="relu")(inputA)
x = Dense(10, activation="relu")(x)
x = Model(inputs=inputA, outputs=x)
# 训练的第二个分支
y = Dense(64, activation="relu")(inputB)
y = Dense(32, activation="relu")(y)
y = Dense(10, activation="relu")(y)
y = Model(inputs=inputB, outputs=y)
# combine the output of the two branches
combined = concatenate([x.output, y.output])
# apply a FC layer and then a regression prediction on the
# combined outputs
z = Dense(64, activation="relu")(combined)
z = Dense(20, activation="linear")(z)
# our model will accept the inputs of the two branches and
# then output a single value
model = Model(inputs=[x.input, y.input], outputs=z)
model.summary()

y_input=concatenate([y_train1,y_train2])
Y_test=concatenate([y_test1,y_test2])

model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
model.fit(x=[x_train1,x_train2],y=y_input, batch_size=200, epochs=10)
model.evaluate([x_test1,x_test2], Y_test)