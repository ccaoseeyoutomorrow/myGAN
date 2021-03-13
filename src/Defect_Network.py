# 速度校准+神经网络缺陷复现
# 缺陷复现
# 设置时间段为判断依据
import math, os, os.path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import LinearSegmentedColormap
import time
from numpy.linalg import lstsq
import re

Cell_Number = 20
NodeA_num = 8  # 节点数量
NodeB_num = 8
Tree_Radius = 25  # 树木最长直径 单位：cm
PI = 3.141592654


class Node():  # 存放传感器位置
    def __init__(self, x=0, y=0):
        """
        传感器位置类
        :param x:
        :param y:
        """
        self.x = x
        self.y = y


def Node_update(Node_location):
    """
    创建传感器类list
    :param Node_location:
    :return:
    """
    Node_list = []
    for i in Node_location:
        Node_list.append(Node(i[0], i[1]))
    return Node_list  # 返回存放Node位置的list


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


# 网格类
class Cell():
    def __init__(self, radiusA, radiusB):
        self.X = np.zeros(shape=(Cell_Number, Cell_Number))
        self.Y = np.zeros(shape=(Cell_Number, Cell_Number))
        self.inner = np.zeros(shape=(Cell_Number, Cell_Number,1), dtype='int')  # 是否在园内的标号
        D_number=int((NodeA_num-1)*NodeA_num/2)
        self.D=np.zeros(shape=(Cell_Number, Cell_Number,D_number))

        cell_length = (max(radiusA, radiusB) + 1) * 2 / Cell_Number
        for i in range(Cell_Number):
            self.X[:, i] = (i - Cell_Number / 2) * cell_length
            self.Y[i, :] = -(i - Cell_Number / 2) * cell_length

        for i in range(Cell_Number):
            for j in range(Cell_Number):
                # 判断点是否在椭圆内
                if Ellipse_distance(0, 0, self.X[i][j], self.Y[i][j], radiusA, radiusB):
                    self.inner[i][j] = 1


    def re_label(self, yuzhi):
        """
        根据阈值生成label
        :param yuzhi:
        """
        temp = np.zeros(shape=(Cell_Number, Cell_Number))
        for i in range(Cell_Number):
            for j in range(Cell_Number):
                if self.inner[i][j] == False:
                    continue
                if self.V[i][j] <= yuzhi:
                    temp[i][j] = 2
                else:
                    temp[i][j] = 1
        return temp


    # 根据原射线对小格子进行速度估计
    def update_D(self,Node_list_A, Node_list_B):
        """
        :param V_list: 速度矩阵
        :param Node_list_A: A列传感器位置集合
        :param Node_list_B: B列传感器位置集合
        :param dis: cell距离线的最短距离
        :param sort: 优先选择方式，0：最快，1：最慢，2：距离最近，3：直线的长度最短，其他：平均值
        """
        for i in range(Cell_Number):
            for j in range(Cell_Number):
                # 判断点是否在园内
                CX = self.X[i][j]
                CY = self.Y[i][j]
                count = 0
                for n in range(NodeA_num):
                    for m in range(n + 1, NodeB_num):
                        self.D[i][j][count] = pl_distance(CX, CY, Node_list_A[n].x, Node_list_A[n].y, Node_list_B[m].x,
                                              Node_list_B[m].y)
                        count+=1


    def rect_tangle_distance(self, X, Y):
        """
        判断点Cell是否在四边形内
        :param X:
        :param Y:
        :return:
        """
        temp = False
        temp1 = -(6.8 / 2.4) * (X - 3.4)
        temp2 = (6.8 / 0.9) * (X - 8.3)
        if (Y >= temp1 and Y >= temp2 and Y >= 0 and Y <= 6.8):
            temp = True
        return temp


def pl_distance(px, py, ax, ay, bx, by):
    """
    计算点P到点A，点B形成的直线的距离
    :param px:
    :param py:
    :param ax:
    :param ay:
    :param bx:
    :param by:
    :return:
    """
    # 对于两点坐标为同一点时,返回点与点的距离
    if ax == bx and ay == by:
        point_array = np.array((px, py))
        point1_array = np.array((ax, ay))
        return np.linalg.norm(point_array - point1_array)
    # 计算直线的三个参数
    A = by - ay
    B = ax - bx
    C = (ay - by) * ax + (bx - ax) * ay
    # 根据点到直线的距离公式计算距离
    distance = np.abs(A * px + B * py + C) / (np.sqrt(A ** 2 + B ** 2))
    return distance


def Ellipse_distance(Circle_X, Circle_Y, Cell_X, Cell_Y, a, b):
    """
    判断点Cell是否在椭圆内
    :param Circle_X：椭圆圆心X坐标
    :param Circle_Y：椭圆圆心Y坐标
    :param Cell_X：点Cell的X坐标
    :param Cell_Y：点Cell的Y坐标
    :param a：椭圆长轴
    :param b：椭圆短轴
    """
    if a == 0 or b == 0:
        return 0
    dis = (Circle_X - Cell_X) * (Circle_X - Cell_X) / (a * a) + (Circle_Y - Cell_Y) * (Circle_Y - Cell_Y) / (b * b)
    if (dis <= 1):
        return 1
    else:
        return 0

# 超声波射线类，存超声波射线的传播时间、传播距离、速度等
class Ultrasonic_Line():
    def __init__(self, Node_list_R, Node_list_T, time, agflag):
        self.C = np.zeros(shape=(NodeA_num, NodeB_num))  # 类似离心率
        self.B = np.zeros(shape=(NodeA_num, NodeB_num))  # 短轴
        self.Time_list = np.zeros(shape=(NodeA_num, NodeB_num))
        self.Distance_list = np.zeros(shape=(NodeA_num, NodeB_num))  # 距离，也是长轴
        self.Speed_list = np.zeros(shape=(NodeA_num, NodeB_num))
        last_network_number=int(((NodeA_num-1)*NodeA_num)/2)
        self.Speed_list2=np.zeros(shape=last_network_number)
        self.bias = np.zeros(shape=(NodeA_num, NodeB_num))  # 偏置：1-β*β
        self.timebias = np.zeros(shape=(NodeA_num, 2))

        for i in range(NodeA_num):  # 距离list赋值
            for j in range(NodeB_num):
                if i != j:
                    temp1 = 2.4  # 传感器之间的误差
                    temp2 = distance(Node_list_R[i], Node_list_T[j])  # 传感器之间的距离
                    # 给距离赋值，单位：厘米
                    self.Distance_list[i][j] = math.sqrt(temp1 * temp1 + temp2 * temp2)
                else:
                    self.Distance_list[i][j] = 0

        # 时间list赋值
        data_list = time
        count = 0

        for i in range(NodeA_num):
            for j in range(i + 1, NodeB_num):
                if i != j:
                    # 给时间赋值，单位：毫秒
                    self.Time_list[i][j] = data_list[count]
                    self.Time_list[j][i] = data_list[count]
                else:
                    self.Time_list[j][i] = 0
                count += 1

        self.Speed_list = np.divide(self.Distance_list, self.Time_list)  # 速度list赋值
        for i in range(len(self.Speed_list)):
            for j in range(len(self.Speed_list)):
                temp = min(abs(i + NodeA_num - j), abs(i - j))
                biospi = (90 - temp * 22.5) * math.pi / 180  # 90-temp*22.5°为圆周角度数，角度转化为弧度1°=π/180
                self.bias[i][j] = 1 - 0.2 * biospi * biospi
        self.Speed_list = np.divide(self.Speed_list, self.bias)

        # 根据无损区域的速度，来更新时间误差
        if agflag == 1:
            self.Shen_updateV(time)

        # 速度归一化操作
        # 找出比thres大的下标号
        temp = np.where(self.Speed_list.reshape(NodeA_num * NodeB_num) < 1000)[0]
        # 找出最大/小速度的下标号
        maxlabel = temp[self.Speed_list.reshape(NodeA_num * NodeB_num)[temp].argsort()[-1]]
        mixlabel = temp[self.Speed_list.reshape(NodeA_num * NodeB_num)[temp].argsort()[0]]
        maxspeed = self.Speed_list.reshape(NodeA_num * NodeB_num)[maxlabel]
        minspeed = self.Speed_list.reshape(NodeA_num * NodeB_num)[mixlabel]
        mm = maxspeed - minspeed
        count = 0
        count2=0
        for i in range(NodeA_num):
            for j in range(i + 1, NodeB_num):
                if i != j:
                    # 给时间赋值，单位：毫秒
                    self.Speed_list[i][j] = (self.Speed_list[i][j] - minspeed) / mm
                    self.Speed_list[j][i] = (self.Speed_list[j][i] - minspeed) / mm
                    self.Speed_list2[count2]=(self.Speed_list[i][j] - minspeed) / mm
                    count2+=1
                else:
                    self.Time_list[j][i] = 0
                count += 1



    def Shen_updateV(self, time):
        """
        速度误差校正
        :param file_time_name:
        """
        # 时间list赋值
        data_list = time
        # 计算时间补偿
        speed_sort = np.zeros(shape=(28), dtype='int')
        count = 0
        for i in range(8):
            for j in range(i + 1, 8):
                speed_sort[count] = i * 8 + j
                count += 1
        sorted_speed = speed_sort[self.Speed_list.reshape(8 * 8)[speed_sort].argsort()]
        count_num = 14
        count_array = np.zeros(shape=(count_num, 3), dtype='int')
        count_array1 = [[0, 0, 7], [1, 0, 1], [2, 2, 3], [3, 3, 4], [4, 4, 5], [5, 5, 6], [6, 6, 7]]
        for i in range(count_num):
            count_array[i] = [i, int(sorted_speed[-i - 2] / 8), sorted_speed[-i - 2] % 8]
        # count_array=np.array([[0,0,7],[1,0,1],[2,2,3],[3,3,4],[4,4,5],[5,5,6],[6,6,7],[7,0,2],
        #                [8,1,3],[9,2,4],[10,3,5],[11,4,6],[12,5,7],[13,0,6]],dtype='int')
        count_array = np.vstack((count_array1, count_array))
        for i in range(count_array.shape[0]):
            count_array[i][0] = i
        arrayl = count_array.shape[0]
        A = np.zeros(shape=(arrayl, 14))  # 构造系数矩阵 A
        B = np.zeros(shape=arrayl).T  # 构造转置矩阵 b （这里必须为列向量）
        de0 = 1  # 默认接收偏置标号
        de1 = 4  # 默认发送偏置标号
        del0 = 1  # 默认接收传感器位置
        del1 = 2  # 默认发送传感器位置l
        dij = self.Distance_list[del0][del1]
        bij = self.bias[del0][del1]
        tij = self.Time_list[del0][del1]
        for c, a, b in count_array:
            # 接收传感器标号
            if a == 0:
                n = 0
            else:
                n = a * 2 - 1
                # 发送传感器标号
            if b == 7:
                m = 7 * 2 - 1
            else:
                m = b * 2
            # ti=t[de0] 要求的系数
            # tj=t[de1] 要求的系数
            dnm = self.Distance_list[a][b]
            bnm = self.bias[a][b]
            tnm = self.Time_list[a][b]
            # tn=t[n] 要求的系数
            # tm=t[m] 要求的系数
            #   d[n][m]*bias[i][j]*t[i]+d[n][m]*bias[i][j]*t[j]-
            #   d[i][j]*bias[n][m]*t[n]-d[i][j]*bias[n][m]*t[m]==
            #   d[n][m]*t[i][j]*bias[i][j]-d[i][j]*bias[n][m]*t[n][m]
            A[c][de0] += dnm * bij
            A[c][de1] += dnm * bij
            A[c][n] -= dij * bnm
            A[c][m] -= dij * bnm
            B[c] = dnm * tij * bij - dij * bnm * tnm
        r = lstsq(A, B, rcond=None)  # 调用 solve 函数求解
        # print(r[2])
        for i in range(14):
            if i == 13:
                n1 = 7
                n2 = 1
            elif i == 0:
                n1 = 0
                n2 = 0
            else:
                n1 = int((i + 1) / 2)
                n2 = (i + 1) % 2
            self.timebias[n1][n2] = r[0][i]

        # 对原始时间加上偏置，重新计算
        count = 0
        for i in range(NodeA_num):
            for j in range(i + 1, NodeB_num):
                if i != j:
                    # 给时间赋值，单位：毫秒
                    self.Time_list[i][j] = data_list[count] - self.timebias[i][0] - self.timebias[j][1]
                    self.Time_list[j][i] = data_list[count] - self.timebias[i][0] - self.timebias[j][1]
                else:
                    self.Time_list[j][i] = 0
                count += 1
        self.Speed_list = np.divide(self.Distance_list, self.Time_list)  # 速度list赋值
        self.Speed_list = np.divide(self.Speed_list, self.bias)


def find_minN(nplist, num, xnum, ynum, thres):
    """
    找出nplist中比thres大的，第num个小的数值，xnum、ynum为list长宽
    例子
    temp=find_minN(self.Speed_list,2,8,8,1)
    :param nplist:
    :param num:
    :param xnum:
    :param ynum:
    :param thres:
    :return:
    """
    relist = nplist.reshape(xnum * ynum)
    # 找出比thres大的下标号
    temp = np.where(relist > thres)[0]
    # 找出下标号
    temp = temp[relist[temp].argsort()[num]]
    return nplist[int(temp / xnum)][temp % xnum]


def find_yuzhi(nplist, num, xnum, ynum):
    """
    找出并返回nplist中第num个小的数值
    :param nplist:
    :param num:
    :param xnum:
    :param ynum:
    :return:
    """
    relist = nplist.reshape(xnum * ynum)
    # 找出比thres大的下标号
    temp = np.where(relist < 10000)[0]
    # 找出下标号
    temp = temp[relist[temp].argsort()[num]]
    return nplist[int(temp / xnum)][temp % xnum]


# 计算点X到点Y的距离
def distance(X, Y):
    return math.sqrt((X.x - Y.x) * (X.x - Y.x) + (X.y - Y.y) * (X.y - Y.y))


def show_plt(list_v, yuzhi, cell_inner):
    """
    根据list_v和阈值，显示缺陷图像
    :param list_v:
    :param yuzhi:
    :param cell_inner:
    """
    fig, ax = plt.subplots()  # 更新
    fig.suptitle(time.strftime("%m%d%H%M%S", time.localtime()) + 'show_plt')
    x = []
    y = []
    for i in range(Cell_Number):
        for j in range(Cell_Number):
            if (list_v[i][j] <= 10):
                x.append(j)
                y.append(Cell_Number - i)
    ax.plot(x, y, 'wo')
    x = []
    y = []
    for i in range(Cell_Number):
        for j in range(Cell_Number):
            if cell_inner[i][j]:
                x.append(j)
                y.append(Cell_Number - i)
    ax.plot(x, y, 'go')
    x = []
    y = []
    for i in range(Cell_Number):
        for j in range(Cell_Number):
            if (list_v[i][j] <= yuzhi and list_v[i][j] > 10):
                x.append(j)
                y.append(Cell_Number - i)
    ax.plot(x, y, 'ro')
    plt.show()


def show_heatmap(list_v, agflag):
    """
    显示list_v的热力图
    :param list_v:
    :param agflag:
    """
    red_thre = 0.15
    yellow_thre = red_thre * 1.5
    # 红-黄-绿 无渐变
    cdict1 = {'red': ((0.0, 1.0, 1.0),
                      (0.01, 1.0, 1.0),
                      (yellow_thre, 1.0, 0.0),
                      (0.99, 0.0, 1.0),
                      (1, 1.0, 1.0)),

              'green': ((0.0, 1.0, 1.0),
                        (0.01, 1.0, 0.0),
                        (red_thre, 0.0, 1.0),
                        (1, 1.0, 1.0)),

              'blue': ((0.0, 1.0, 1.0),
                       (0.01, 1.0, 0.0),
                       (0.99, 0.0, 1.0),
                       (1, 1.0, 1.0)),
              }
    # 红-黄-绿 带渐变
    cdict = {'red': ((0.0, 1.0, 1.0),
                     (0.01, 1.0, 1.0),
                     (yellow_thre, 1.0, 1.0),
                     (0.75, 0.0, 0.0),
                     (1.0, 1.0, 0.0)),

             'green': ((0.0, 1.0, 1.0),
                       (0.01, 1.0, 0.0),
                       (red_thre, 1.0, 1.0),
                       (1.0, 1.0, 0.0)),

             'blue': ((0.0, 1.0, 1.0),
                      (0.01, 1.0, 0.0),
                      (0.5, 0.0, 0.0),
                      (0.75, 0.0, 0.0),
                      (1.0, 1.0, 0.0)),
             }
    cmap_name = 'my_list'
    fig, axs = plt.subplots(figsize=(15, 15))
    if agflag == 1:
        fig.suptitle('revised' + 'heatmap', size=50)
    else:
        fig.suptitle('heatmap', size=50)
    blue_red1 = LinearSegmentedColormap(cmap_name, cdict)
    plt.register_cmap(cmap=blue_red1)
    im1 = axs.imshow(list_v, cmap=blue_red1)
    fig.colorbar(im1, ax=axs)  # 在图旁边加上颜色bar
    plt.show()


def ultra_ray(Speed_list, Node_list_A, Node_list_B, yuzhi):
    """
    显示射线图
    :param Speed_list:
    :param Node_list_A:
    :param Node_list_B:
    :param yuzhi:
    """
    fig, ax = plt.subplots()  # 更新
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    for i in range(NodeA_num):
        for j in range(NodeB_num):
            if (Speed_list[i][j] <= yuzhi):
                color = 'r'
            else:
                color = 'g'
            ax.plot([Node_list_A[i].x, Node_list_B[j].x], [Node_list_A[i].y, Node_list_B[j].y], color=color)
    fig.suptitle(time.strftime("%m%d%H%M%S", time.localtime()) + 'ultra_ray')
    plt.show()


def with_no(numA, numB):
    """
    判断numA和numB是否同号
    :param numA:
    :param numB:
    :return:
    """
    return numA * numB >= 0


def _calulate_corss_lines(line0_pos0, line0_pos1, line1_pos0, line1_pos1):
    """
    求两条直线直接的交点
    :param line0_pos0: 第一条直接的第一个点的坐标
    :param line0_pos1: 第一条直接的第二个点的坐标
    :param line1_pos0: 第二条直接的第一个点的坐标
    :param line1_pos1: 第二条直接的第二个点的坐标
    """
    # x = (b0*c1 – b1*c0)/D
    # y = (a1*c0 – a0*c1)/D
    # D = a0*b1 – a1*b0， (D为0时，表示两直线重合)
    line0_a = line0_pos0[1] - line0_pos1[1]
    line0_b = line0_pos1[0] - line0_pos0[0]
    line0_c = line0_pos0[0] * line0_pos1[1] - line0_pos1[0] * line0_pos0[1]
    line1_a = line1_pos0[1] - line1_pos1[1]
    line1_b = line1_pos1[0] - line1_pos0[0]
    line1_c = line1_pos0[0] * line1_pos1[1] - line1_pos1[0] * line1_pos0[1]
    d = line0_a * line1_b - line1_a * line0_b
    if d == 0:
        # 重合的边线没有交点
        return None
    x = (line0_b * line1_c - line1_b * line0_c) * 1.0 / d
    y = (line0_c * line1_a - line1_c * line0_a) * 1.0 / d
    return x, y


def read_show(filename):
    """
    读取缺陷文件txt并显示
    :param filename:
    """
    a = np.loadtxt(filename, dtype=int)  # 最普通的loadtxt
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


def compareAB(list_label, list_test):
    """
    定量算法的计算能力
    :param list_label:标签
    :param list_test: 算法生成的图像
    :return:
    """
    area_count = 0  # 横截面总数量
    defect_count = 0  # 缺陷总数量
    TP = 0  # TP-表示该区域被正确地重建为缺陷；
    FN = 0  # FN-表示该区域被错误地重建为正常木材；
    FP = 0  # FP-表示该区域被错误地重建为缺陷；
    TN = 0  # TN-则表示该区域被正确地重建为正常木材
    for i in range(list_label.shape[0]):
        for j in range(list_label.shape[1]):
            if list_label[i][j] == 2 or list_label[i][j] == 1:
                area_count += 1
            if list_label[i][j] == 2:
                defect_count += 1
                if list_test[i][j] == 2:
                    TP += 1
                else:
                    FN += 1
            elif list_test[i][j] == 2:
                FP += 1
            elif list_test[i][j] == 1:
                TN += 1
    # 准确度(Accuracy)=(TP+TN)/(TP+TN+FP+FN)
    # 精度(Precision)=TP/(TP+FP)
    # 查全率(Recall)=TP/(TP+FN)
    # Accuracy=(TP+TN)/(TP+TN+FP+FN)#正确的重建除以总面积
    # Precision=TP/(TP+FP)#正确的缺陷重建除以算法总缺陷重建
    # Recall=TP/(TP+FN)#正确的缺陷重建除以真是总缺陷面积
    # print("总共面积单元：",area_count,
    #       "\n缺陷面积单元：",defect_count,
    #       "\n计算正确的缺陷面积单元：",TP,
    #       "\n未计算的缺陷面积单元：",FN,
    #       "\n计算错误的缺陷面积单元：",FP,
    #       "\n准确度(Accuracy)：",Accuracy,
    #       "\n精度(Precision)：",Precision,
    #       "\n查全率(Recall)：",Recall)
    return area_count, defect_count, TP, FN, FP, TN


def read_txt(filename):
    """
    读取文件并存储会int类型返回
    :param filename:
    :return:
    """
    a = np.loadtxt(filename, dtype=int)  # 最普通的loadtxt
    return a


def get_locationbyradius(a, b):
    """
    返回由a，b生成的list位置
    :param a: 椭圆长轴长度
    :param b: 椭圆短轴长度
    """
    locat_list = [[] for i in range(NodeA_num)]
    for i in range(NodeA_num):

        θ = i * 360 / NodeA_num - 90
        if θ == -90 or θ == 90:
            xtemp = 0
            ytemp = b
        elif θ == 0 or θ == 180:
            xtemp = a
            ytemp = 0
        else:
            xtemp = math.sqrt((a * a * b * b) / (b * b + a * a * math.tan(θ) * math.tan(θ)))
            ytemp = xtemp * math.tan(θ)
        locat_list[i].append(xtemp)
        locat_list[i].append(ytemp)
    locat_list = np.array(locat_list, dtype='float').reshape(8, 2)  # 将数据从list类型转换为array类型。
    return locat_list


def get_speed_ray(file_time_name, file_locat_name):
    agflag = 1
    # radius = 12  # 检测树木传感器的位置半径
    locat_list = [[] for i in range(8)]
    with open(file_locat_name, 'r', encoding='utf-8') as file_to_read:
        for i in range(8):
            lines = file_to_read.readline()  # 整行读取数据
            nums = re.split(',|\n', lines)
            locat_list[i].append(nums[0])  # 添加新读取的数据
            locat_list[i].append(nums[1])  # 添加新读取的数据
    locat_list = np.array(locat_list, dtype='float').reshape(8, 2)  # 将数据从list类型转换为array类型。
    # 根据radius和Node的数量自动计算出坐标赋值
    # locat_list=get_locationbyradius(radiusa,radiusb)
    # 根据位置初始化class
    Node_list_A = Node_update(locat_list)
    Node_list_B = Node_update(locat_list)
    Ultra_Line = Ultrasonic_Line(Node_list_A, Node_list_B, file_time_name, agflag)
    return Ultra_Line


def get_speed_raybynpy(filename, labelname):
    data = np.load(filename)
    data = np.array(data, dtype='float')
    last_network_number=int(((NodeA_num-1)*NodeA_num)/2*2)+1
    data_28x28=np.zeros(shape=(data.shape[0],Cell_Number,Cell_Number,last_network_number))
    for i in range(data.shape[0]):
        locat_list = np.array(data[i][28:44], dtype='float').reshape(8, 2)  # 将数据从list类型转换为array类型。
        # 根据位置初始化class
        Node_list = Node_update(locat_list)
        Ultra_Line = Ultrasonic_Line(Node_list, Node_list, np.array(data[i][0:28], dtype='float'), 1)
        cell_100 = Cell(10.3, 11.6)
        cell_100.update_D( Node_list, Node_list)  # 更新小格子距离每个射线的距离
        temp1=cell_100.D
        temp2=np.tile(Ultra_Line.Speed_list2,(Cell_Number,Cell_Number)).reshape(Cell_Number,Cell_Number,28)

        temp3=cell_100.inner
        temp=np.concatenate((temp1,temp2,temp3),axis=2)
        data_28x28[i][:]=temp
    return data_28x28


if __name__ == '__main__':
    npy_name = '../Defect_Data/4号树木x_median.npy'
    label_name = '../Defect_Data/label4_20.txt'
    data_28x28=get_speed_raybynpy(npy_name,label_name)
    np.save('../Defect_Data/4号树木x_GAN.npy', data_28x28)
