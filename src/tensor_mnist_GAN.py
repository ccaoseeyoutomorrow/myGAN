import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.layers import Dense, concatenate
from tensorflow.python.keras.utils import np_utils
from 手写CNN import loadmnist





def get_generator(noise_img, n_units, out_dim, reuse=False, alpha=0.01):
    """
    生成器
    noise_img: 生成器的输入
    n_units: 隐层单元个数
    out_dim: 生成器输出tensor的size，这里应该为32*32=784
    alpha: leaky ReLU系数
    """
    # hidden layer
    hidden1 = Dense(n_units)(noise_img)
    # leaky ReLU
    hidden1 = tf.maximum(alpha * hidden1, hidden1)
    # dropout
    hidden1 = tf.nn.dropout(hidden1, rate=0.2)
    # logits & outputs
    logits = Dense(out_dim)(hidden1)
    outputs = tf.tanh(logits)
    return logits, outputs


def get_discriminator(img, n_units, reuse=False, alpha=0.01):
    """
    判别器
    n_units: 隐层结点数量
    alpha: Leaky ReLU系数
    """
    # hidden layer
    hidden1 = Dense(n_units)(img)
    hidden1 = tf.maximum(alpha * hidden1, hidden1)
    # logits & outputs
    logits = Dense(1)(hidden1)
    outputs = tf.sigmoid(logits)
    return logits, outputs


# 定义参数
x_train, y_train, x_test, y_test = loadmnist()
x_train = x_train.reshape(-1, 28, 28) / 255.0
# 真实图像的size
img_size = x_train[0].shape[0]
# 传入给generator的噪声size
noise_size = 100
# 生成器隐层参数
g_units = 128
# 判别器隐层参数
d_units = 128
# leaky ReLU的参数
alpha = 0.01
# learning_rate
learning_rate = 0.001
# label smoothing
smooth = 0.1


img = x_train.images[50]
plt.imshow(img.reshape((28, 28)), cmap='Greys_r')


# define two sets of inputs
inputA = Input(shape=(784,))
inputB = Input(shape=(128,))

# 训练的第一个分支
# hidden layer
G = Dense(784, activation="relu")(inputA)
# leaky ReLU
G = tf.maximum(alpha * G, G)
# dropout
G = tf.nn.dropout(G, rate=0.2)
# logits & outputs
G_logits= Dense(128, activation="relu")(G)
G_outputs = tf.tanh(G_logits)
G = Model(inputs=inputA, outputs=G_outputs)

# 训练的第二个分支
D = Dense(64, activation="relu")(inputB)
D = tf.maximum(alpha * D, D)
# logits & outputs
D_logits = Dense(1)(D)
D_outputs = tf.sigmoid(D_logits)
D = Model(inputs=inputB, outputs=D_outputs)


# discriminator的loss
# 识别真实图片
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                                     labels=tf.ones_like(d_logits_real)) * (1 - smooth))
# 识别生成的图片
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                     labels=tf.zeros_like(d_logits_fake)))
# 总体loss
d_loss = tf.add(d_loss_real, d_loss_fake)

# generator的loss
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                labels=tf.ones_like(d_logits_fake)) * (1 - smooth))
# combine the output of the two branches
combined = concatenate([G.output, D.output])
# apply a FC layer and then a regression prediction on the
# combined outputs
z = Dense(2, activation="relu")(combined)
z = Dense(1, activation="linear")(z)
# our model will accept the inputs of the two branches and
# then output a single value
model = Model(inputs=[x.input, y.input], outputs=z)