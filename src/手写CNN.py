import numpy as np
import struct
import scipy.io as sio


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


def relu(input_X):
    """
    Arguments:
        input_X -- a numpy array
    Return :
        A: a numpy array. let each elements in array all greater or equal 0
    """

    A = np.where(input_X < 0, 0, input_X)
    return A


def softmax(input_X):
    """
    Arguments:
        input_X -- a numpy array
    Return :
        A: a numpy array same shape with input_X
    """
    exp_a = np.exp(input_X)
    sum_exp_a = np.sum(exp_a, axis=1)
    sum_exp_a = sum_exp_a.reshape(input_X.shape[0], -1)
    ret = exp_a / sum_exp_a
    # print(ret)
    return ret


class Convolution:
    def __init__(self, W, fb, stride=1, pad=0):
        """
        步骤：卷积-relu-池化-全连接-relu-全连接-softmax
        W-- 滤波器权重，shape为(FN,NC,FH,FW),FN 为滤波器的个数
        fb -- 滤波器的偏置，shape 为(1,FN)
        stride -- 步长
        pad -- 填充个数
        """
        self.conv = None
        self.pool1 = None  # 池化

        self.pool2 = None
        self.fnc = None
        self.W = W
        self.fb = fb
        self.stride = stride
        self.pad = pad

    def single_conv(A_slide, W):  # 请注意：这里的A_slide是A_prev的局部，也就是和滤波器重叠的部分
        """
        这个函数是对于一个滤波器组而言的
        """
        s = np.multiply(A_slide, W)  # 卷积的步骤1：A_slide和滤波器对应元素相乘
        Z = np.sum(s)  # 卷积步骤2：将s矩阵的所有元素相加
        return Z  # 说明：这里返回的是一个数，到时候是要赋值给新矩阵的


def loadmnist():
    # 训练集文件
    img_train_path = '../MNIST_data/train-images.idx3-ubyte'
    # 训练集标签文件
    label_train_path = '../MNIST_data/train-labels.idx1-ubyte'
    # 测试集文件
    img_test_path = '../MNIST_data/t10k-images.idx3-ubyte'
    # 测试集标签文件
    label_test_path = '../MNIST_data/t10k-labels.idx1-ubyte'

    with open(label_test_path, 'rb') as file:
        test_labels = np.frombuffer(file.read(), dtype=np.uint8, offset=8)
        m_t10k = len(test_labels)
        test_labels = np.asarray(test_labels).reshape(m_t10k, 1)

    with open(label_train_path, 'rb') as file:
        train_labels = np.frombuffer(file.read(), dtype=np.uint8, offset=8)
        m_train = len(train_labels)
        train_labels = np.asarray(train_labels).reshape(m_train, 1)

    with open(img_test_path, 'rb') as file:
        test_images = np.frombuffer(file.read(), dtype=np.uint8, offset=16)
        test_images = np.asarray(test_images).reshape(m_t10k, 28,28)

    with open(img_train_path, 'rb') as file:
        train_image = np.frombuffer(file.read(), dtype=np.uint8, offset=16)
        train_image = np.asarray(train_image).reshape(m_train, 28,28)

    return train_image, train_labels, test_images, test_labels


def forward_propagation():
    """
    前向传播
    """
    pass


def back_propagation():
    """
    反向传播
    """
    pass


def train(iter):
    for i in range(iter):
        err = 0
        for j in range(60000):  # 训练样本数
            forward_propagation(i, 0)
            back_propagation()
        print("step:", i, "loss:%.5f\n", 1.0 * err / 60000)  # 每次记录一遍数据集的平均误差



