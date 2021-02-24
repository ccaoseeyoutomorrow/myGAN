import numpy as np
import tensorflow as tf
import struct
import matplotlib.pyplot as plt


class layer():
    # 定义卷积网络中的层
    L = 0
    W = 0
    H = 0
    b = np.zeros(shape=(30, 30, 5), dtype='double')
    delta = np.zeros(shape=(30, 30, 5), dtype='double')

    def __init__(self):
        self.m = np.random.rand(30, 30, 5)


class fcnn_layer():
    # 定义全连接网络中的层
    length = 0
    m = np.zeros(shape=(1000), dtype='double')
    b = np.zeros(shape=(1000), dtype='double')
    delta = np.zeros(shape=(1000), dtype='double')

    def __init__(self):
        self.w = np.random.rand(20, 1000)


class Network():
    # 定义CNN
    Input_layer = layer()
    conv_layer1 = layer()
    pool_layer1r = layer()
    filter1 = []
    for i in range(5):
        filter1.apppend(layer())
    fcnn_input = fcnn_layer()
    fcnn_w = fcnn_layer()
    fcnn_outpot = fcnn_layer()


CNN = Network()


def get_images(buf, n):
    """
    读取前n张图片
    :param buf:
    :param n:
    :return:
    """
    im = []
    index = struct.calcsize('>IIII')
    for i in range(n):
        temp = struct.unpack_from('>784B', buf, index)
        im.append(np.reshape(temp, (28, 28)))
        index += struct.calcsize('>784B')
    return im


def get_labels(buf, n):
    '''
    读取前n个标签
    '''
    l = []
    index = struct.calcsize('>II')
    for i in range(n):
        temp = struct.unpack_from('>1B', buf, index)
        l.append(temp[0])
        index += struct.calcsize('>1B')
    return l


def preprocess():
    # 训练集文件
    train_images_idx3_ubyte_file = '../MNIST_data/train-images.idx3-ubyte'
    # 训练集标签文件
    train_labels_idx1_ubyte_file = '../MNIST_data/train-labels.idx1-ubyte'
    # 测试集文件
    test_images_idx3_ubyte_file = '../MNIST_data/t10k-images.idx3-ubyte'
    # 测试集标签文件
    test_labels_idx1_ubyte_file = '../MNIST_data/t10k-labels.idx1-ubyte'
    with open(train_images_idx3_ubyte_file, 'rb') as f:
        train_image = f.read()
    with open(train_labels_idx1_ubyte_file, 'rb') as f:
        train_labels = f.read()
    with open(test_images_idx3_ubyte_file, 'rb') as f:
        test_images = f.read()
    with open(test_labels_idx1_ubyte_file, 'rb') as f:
        test_labels = f.read()
    return train_image, train_labels, test_images, test_labels


'''
读取
'''
image, label, test_img, test_label = preprocess()
n = 16
train_img = get_images(image, n)
train_label = get_labels(label, n)

'''
显示
'''
for i in range(n):
    plt.subplot(4, 4, 1 + i)
    title = u"label:" + str(train_label[i])
    plt.title(title)
    plt.imshow(train_img[i], cmap='gray')
# plt.show()
