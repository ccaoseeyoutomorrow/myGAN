import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # for pretty plots
from scipy.stats import norm

"""
    生成基于size的，统一分布的数据(size,1),用来加载生成数据的噪点
"""
class GenDataLoader():
    def __init__(self, size = 200, low = -1, high = 1):
        self.size = size
        self.low = low
        self.high = high

    def next_batch(self):
        z = np.random.uniform(self.low, self.high, [self.size, 1])
        # z = np.linspace(-5.0, 5.0, self.size) + np.random.random(self.size) * 0.01  # sample noise prior
        # z = z.reshape([self.size, 1])
        return z

"""
    生成基于mu,sigma,size的正态分布数据(size,1),用来加载真实数据（正态分布）的
"""
class RealDataLoader():
    def __init__(self, size = 200, mu = -1, sigma = 1):
        self.size = size
        self.mu = mu
        self.sigma = sigma

    def next_batch(self):
        data = np.random.normal(self.mu, self.sigma, [self.size ,1])  #(batch_size, size)
        data.sort()
        return data


def momentum_optimizer(loss,var_list):
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        0.01,                # Base learning rate.
        batch,  # Current index into the dataset.
        epoch// 4,          # Decay step - this decays 4 times throughout training process.
        0.95,                # Decay rate.
        staircase=True)
    #optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=batch,var_list=var_list)
    optimizer=tf.train.MomentumOptimizer(learning_rate,0.6).minimize(loss,global_step=batch,var_list=var_list)
    return optimizer

class Generator():
    def __init__(self, inputs, input_size = 1, hidden_size = 6, output_size = 1):
        with tf.variable_scope("generator"):
            weight1 = weight_variable(shape=[input_size, hidden_size], name="weight1") #(size, 100)
            bias1 = bias_variable(shape=[1, hidden_size], name="bias1") #(1, 100)
            weight2 = weight_variable(shape=[hidden_size, hidden_size], name="weight2")
            bias2 = bias_variable(shape=[1, hidden_size], name="bias2")
            weight3 = weight_variable(shape=[hidden_size, output_size], name="weight3")
            bias3 = bias_variable(shape=[1, output_size], name="bias3")
            frac1 = tf.nn.tanh(tf.matmul(inputs, weight1) + bias1, name="frac1")   #(batch_size, 100)
            frac2 = tf.nn.tanh(tf.matmul(frac1, weight2) + bias2, name="frac2")
            frac3 = tf.nn.tanh(tf.matmul(frac2, weight3) + bias3, name="frac3")
            self.frac = frac3
            self.var_list = [weight1, bias1, weight2, bias2, weight3, bias3]
            # self.frac, self.var_list = mlp(inputs, 1)
            self.frac = tf.multiply(self.frac, 5)
    def get_param(self):
        return self.frac, self.var_list