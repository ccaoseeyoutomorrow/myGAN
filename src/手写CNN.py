import numpy as np
import struct
import scipy.io as sio
from tensorflow.python.keras.utils import np_utils


class Dense_layer():
    def __init__(self, m,n, num):
        """
        仿照tensorflow创建一个Dense层
        对于尺寸为[m, n]的二维张量input，
        会生成：尺寸为[n, k]的权矩阵kernel，
        和尺寸为[m, k]的偏移项bias。
        内部的计算过程为y = input * kernel + bias，
        输出值y的维度为[m, k]。
        :param m: 输入矩阵行数
        :param n: 输入矩阵列数
        :param input: 输入数据
        :param num: 输出的尺寸大小
        """
        self.kernel = np.random.rand(n, num) / (n * num)
        self.bias = np.zeros(shape=(m, num))
        self.num = num

    def forward(self,input):
        """
        根据input的batch一起进行推测
        :param input:
        :return:
        """
        self.input=input
        y = np.dot(input, self.kernel) + self.bias
        return y

    def predict(self,input):
        kerneltemp=np.mean(self.kernel)


    def backward(self, dz, learning_rate):
        """
        dz-- 前面的导数
        """
        #         print("Affine backward")
        #         print(self.X.shape)
        #         print(dz.shape)
        #         print(self.W.shape)

        # assert (dz.shape == self.out_shape)
        m = self.input.shape[0]

        self.dW = np.dot(self.input.T, dz) / m
        self.db = np.sum(dz, axis=0, keepdims=True) / m
        # assert (self.dW.shape == self.W.shape)
        # assert (self.db.shape == self.b.shape)

        dx = np.dot(dz, self.kernel.T)
        # assert (dx.shape == self.X.shape)

        dx = dx.reshape(self.input.shape)  # 保持与之前的x一样的shape

        # 更新W和b
        self.kernel = self.kernel - learning_rate * self.dW
        self.bias = self.bias - learning_rate * self.db

        return dx


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, input):
        self.mask = input <= 0
        out = input
        out[self.mask] = 0
        return out

    def backward(self, output):
        temp = output
        temp[self.mask] = 0
        return temp


class SoftMax:
    def __init__(self):
        self.y_hat = None

    def forward(self, X):
        self.y_hat = self.softmax(X)
        return self.y_hat

    def backward(self, labels):
        dx = (self.y_hat - labels)
        return dx

    def softmax(self,input_X):
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
    img_train_path = 'C:/Users\Administrator/Desktop/GAN/MNIST_data/train-images.idx3-ubyte'
    # 训练集标签文件
    label_train_path = 'C:/Users\Administrator/Desktop/GAN/MNIST_data/train-labels.idx1-ubyte'
    # 测试集文件
    img_test_path = 'C:/Users\Administrator/Desktop/GAN/MNIST_data/t10k-images.idx3-ubyte'
    # 测试集标签文件
    label_test_path = 'C:/Users\Administrator/Desktop/GAN/MNIST_data/t10k-labels.idx1-ubyte'

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
        test_images = np.asarray(test_images).reshape(m_t10k, 28, 28)

    with open(img_train_path, 'rb') as file:
        train_image = np.frombuffer(file.read(), dtype=np.uint8, offset=16)
        train_image = np.asarray(train_image).reshape(m_train, 28, 28)

    return train_image, train_labels, test_images, test_labels

def cross_entropy_error(labels,logits):
    """
    计算误差
    :param labels:
    :param logits:
    :return:
    """
    return -np.sum(labels*np.log(logits))

class mymodel:
    def __init__(self,batch):
        # 获取数据
        x_train, y_train, x_test, y_test = loadmnist()
        x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
        y_train = np_utils.to_categorical(y_train, num_classes=10)
        y_test = np_utils.to_categorical(y_test, num_classes=10)

        # 构建网络
        x_train = x_train.reshape(-1, 28 * 28)
        self.batch=batch
        self.learning_rate = 1
        self.inputx = x_train
        self.inputy = y_train
        self.x_test=x_test.reshape(-1, 28 * 28)
        self.y_test = y_test
        self.dense1 = Dense_layer(batch,self.inputx.shape[1], 32)
        self.relu1 = Relu()
        self.dense2 = Dense_layer(batch,32, 10)
        self.softmax = SoftMax()

    def forward_propagation(self,X):
        """
        前向传播
        """
        test=self.dense1.predict(X)
        outcome = self.dense1.forward(X)
        outcome = self.relu1.forward(outcome)
        outcome = self.dense2.forward(outcome)
        outcome = self.softmax.forward(outcome)
        return outcome

    def preict(self,X):
        pass

    def back_propagation(self, Y):
        """
        反向传播
        """
        outcome = self.softmax.backward(Y)
        outcome = self.dense2.backward(outcome,self.learning_rate)
        outcome = self.relu1.backward(outcome)
        outcome = self.dense1.backward(outcome, self.learning_rate)
        return outcome

    def train(self):
        iter=int(self.inputx.shape[0]/self.batch)
        lost=0
        for i in range(iter):
            inputx=self.inputx[i*self.batch:(i+1)*self.batch]
            inputy=self.inputy[i*self.batch:(i+1)*self.batch]
            outcome=self.forward_propagation(inputx)
            losttemp=cross_entropy_error(inputy,outcome) #计算损失函数
            lost+=losttemp
            outcome = self.back_propagation(inputy)
        print("lost:",lost)

    def predict(self):
        iter = int(self.x_test.shape[0] / self.batch)
        accuracy=0
        for i in range(iter):
            inputx=self.x_test[i*self.batch:(i+1)*self.batch]
            inputy=self.y_test[i*self.batch:(i+1)*self.batch]
            outcome = self.forward_propagation(inputx)
            one_hot = np.zeros_like(outcome)
            one_hot[range(inputx.shape[0]), np.argmax(outcome, axis=1)] = 1
            accuracy += np.sum(np.argmax(one_hot, axis=1) == np.argmax(inputy, axis=1))
        accuracy=accuracy/self.x_test.shape[0]
        return accuracy


def main():
    Model = mymodel(200)
    for i in range(10):
        Model.train()
        print(Model.predict())
    # for i in range(5):
    #     err = 0
    #     for j in range(60000):  # 训练样本数
    #         outcome = Model.forward_propagation()
    #         forward_propagation(x_train, y_train)
    #         back_propagation()
    #     print("step:", i, "loss:%.5f\n", 1.0 * err / 60000)  # 每次记录一遍数据集的平均误差


if __name__ == '__main__':
    main()
