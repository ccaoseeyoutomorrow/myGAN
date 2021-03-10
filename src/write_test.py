import numpy as np
import struct
import scipy.io as sio
import numpy as np
import scipy.io as io
import math, os, os.path
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import LinearSegmentedColormap
import time
from numpy.linalg import lstsq


# 读取txt文件数据
def readfile(filename):
    data_list = []
    with open(filename, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()  # 整行读取数据
            if not lines:
                break
            if lines != '\n':
                temp = round(float(lines), 4)
                data_list.append(temp)  # 添加新读取的数据
    data_list = np.array(data_list)  # 将数据从list类型转换为array类型。
    return data_list


def write_x():
    """
    编写4号树木的x轴输入信息
    """
    ospath = '../Data2/实验室4号树木/'
    ospathnames = [name for name in os.listdir(ospath)
                   if os.path.isdir(os.path.join(ospath, name))]
    for i in range(len(ospathnames)):
        ospathnames[i] = ospath + ospathnames[i] + '/'
    outcome = []
    for osname in ospathnames:
        filenames = [name for name in os.listdir(osname)
                     if os.path.isfile(os.path.join(osname, name)) and
                     name.endswith('树莓派.txt')]
        for i in range(len(filenames)):
            filenames[i] = osname + filenames[i]
        locat_filename = osname + 'location.txt'
        locat_list = [[] for i in range(8)]
        with open(locat_filename, 'r', encoding='utf-8') as file_to_read:
            for i in range(8):
                lines = file_to_read.readline()  # 整行读取数据
                nums = re.split(',|\n', lines)
                locat_list[i].append(nums[0])  # 添加新读取的数据
                locat_list[i].append(nums[1])  # 添加新读取的数据
        locat_list = np.array(locat_list).reshape(16)  # 将数据从list类型转换为array类型。

        for i in range(25):
            data_list = readfile(filenames[i])
            data_list = np.concatenate((data_list, locat_list), axis=0)
            outcome.append(data_list)
    outcome = np.array(outcome).reshape(-1, 44)
    np.save('../Defect_Data/4号树木x_8x25.npy', outcome)

def write_y():
    """
    这个函数不会被调用，用于编写y轴信息
    """
    label=np.loadtxt('../Defect_Data/label4.txt',encoding='utf-8',dtype='int')
    label=np.array(label).reshape(-1)
    temp=np.zeros(shape=(5476,10000),dtype='int')
    temp[:,0:10000]=label
    pass

def write_test_x():
    """
    编写1-3号树木的输入x信息，秩序改写osname和np.save的文件名
    """
    osname = '../Data/实验室3号树木/'
    filenames = [name for name in os.listdir(osname)
                 if os.path.isfile(os.path.join(osname, name)) and
                 (name.startswith('树莓派') or
                  name.startswith('20') or
                  name.startswith('手按'))]
    for i in range(len(filenames)):
        filenames[i] = osname + filenames[i]
    locat_filename = osname + 'location.txt'
    locat_list = [[] for i in range(8)]
    with open(locat_filename, 'r', encoding='utf-8') as file_to_read:
        for i in range(8):
            lines = file_to_read.readline()  # 整行读取数据
            nums = re.split(',|\n', lines)
            locat_list[i].append(nums[0])  # 添加新读取的数据
            locat_list[i].append(nums[1])  # 添加新读取的数据
    locat_list = np.array(locat_list).reshape(16)  # 将数据从list类型转换为array类型。
    outcome=[]
    for i in range(len(filenames)):
        data_list = readfile(filenames[i])
        data_list = np.concatenate((data_list, locat_list), axis=0)
        outcome.append(data_list)
    outcome = np.array(outcome).reshape(-1, 44)
    np.save('../Defect_Data/3号树木x.npy', outcome)
    pass

write_x()