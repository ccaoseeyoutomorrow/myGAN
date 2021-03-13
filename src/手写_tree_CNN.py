import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def read_trainx():
    data=np.load('../Defect_Data/4号树木x.npy')
    data=np.array(data,dtype='float')
    return data

def read_trainy():
    label=np.loadtxt('../Defect_Data/label4_20.txt',encoding='utf-8',dtype='int')
    label=np.array(label).reshape(-1)
    temp=np.zeros(shape=(5476,400),dtype='int')
    temp[:,0:400]=label
    return temp

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

class generate_img:
    def __init__(self):
        self.y = None
        self.thre1=0
        self.thre2=0

    def forward(self, X):
        self.x=X
        Xt=X.reshape(X.shape[0]*X.shape[1])
        mint=np.min(X)
        maxt=np.max(X)
        mm=maxt-mint
        if self.thre1==self.thre2:
            thre1=mint+mm/4 #空白面积的分隔符
            thre2=maxt-mm/10 #缺陷面积的分隔符
        xiabiao0=np.where(Xt<=self.thre1)[0] #找出在面积外的下标，为空白
        # xiabiao1=np.where(X>thre1 and X<thre2)[0] #找出在面积内的下标
        xiabiao2 = np.where(Xt>=self.thre2)[0]#找出在面积内比最大值小22的下标号，即为缺陷
        Xt[:]=1 #正常区域
        Xt[xiabiao0]=0 #空白
        Xt[xiabiao2]=2 #缺陷
        self.y=Xt.reshape(X.shape[0],X.shape[1])
        return self.y

    def backward(self, labels):
        m = self.x.shape[0]
        dz=self.y - labels
        self.dth1 = np.sum(dz, axis=0, keepdims=True) / m

        # 更新W和b
        self.thre1 = self.thre1 - 0.1 * self.dth1
        dx = (self.y - labels)
        return dx

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
        x_train=read_trainx()
        y_train=read_trainy()
        x_test=x_train[0:1000]
        y_test=y_train[0:1000]
        x_train=x_train[1000:]
        y_train=y_train[1000:]

        # 基础参数
        self.batch=batch
        self.learning_rate = 3
        self.inputx = x_train
        self.inputy = y_train
        self.x_test=x_test
        self.y_test = y_test

        #构建网络
        self.dense1 = Dense_layer(batch,self.inputx.shape[1], 1000)
        self.relu1 = Relu()
        self.dense2 = Dense_layer(batch,1000, 400)
        self.Gimg = generate_img()

    def forward_propagation(self,X):
        """
        前向传播
        """
        test=self.dense1.predict(X)
        outcome = self.dense1.forward(X)
        outcome = self.relu1.forward(outcome)
        outcome = self.dense2.forward(outcome)
        outcome = self.Gimg.forward(outcome)
        return outcome

    def preict(self,X):
        pass

    def back_propagation(self, Y):
        """
        反向传播
        """
        outcome = self.Gimg.backward(Y)
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
            temp=np.where(outcome!=inputy)[0]
            sum=temp.shape[0]
            accuracy+=sum/inputy[0].shape[0]
        accuracy=accuracy/self.x_test.shape[0]
        return accuracy

    def show_one(self):
        """
        显示预测的一张图
        """
        inputx=self.x_test[0*self.batch:(0+1)*self.batch]
        outcome = self.forward_propagation(inputx)
        a=outcome[0].reshape(20,20)
        # 红-黄-绿 无渐变
        cdict = {'red': ((0.0, 0.0, 1.0),
                         (0.3, 1.0, 0.0),
                         (0.6, 0.0, 1.0),
                         (1, 1.0, 1.0)),

                 'green': ((0.0, 0.0, 1.0),
                           (0.3, 1.0, 1.0),
                           (0.6, 1.0, 0.0),
                           (1, 0.0, 0.0)),

                 'blue': ((0.0, 0.0, 1.0),
                          (0.3, 1.0, 0.0),
                          (0.6, 0.0, 0.0),
                          (1, 0.0, 0.0)),
                 }
        cmap_name = 'defect'
        fig, axs = plt.subplots(figsize=(15, 15))
        blue_red1 = LinearSegmentedColormap(cmap_name, cdict)
        plt.register_cmap(cmap=blue_red1)
        im1 = axs.imshow(a, cmap=blue_red1)
        fig.colorbar(im1, ax=axs)  # 在图旁边加上颜色bar
        plt.show()

def main():
    Model = mymodel(200)
    for i in range(5):
        Model.train()
        print(Model.predict())
    Model.show_one()
    # for i in range(5):
    #     err = 0
    #     for j in range(60000):  # 训练样本数
    #         outcome = Model.forward_propagation()
    #         forward_propagation(x_train, y_train)
    #         back_propagation()
    #     print("step:", i, "loss:%.5f\n", 1.0 * err / 60000)  # 每次记录一遍数据集的平均误差


if __name__ == '__main__':
    main()
