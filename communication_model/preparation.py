# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time      :   2021/4/13 14:10
# @Author    :   Chasion
# Description:   用来生成数据
import numpy as np
from Data.simulation import Simulation
from sklearn.metrics.pairwise import euclidean_distances
import pickle as plk
C = 3.0*100000000


class Radio:
    def __init__(self):
        """
        电台模型：
        参数：
            电台所在坦克id
            坦克坐标


            波长：     7.5m
            中心频率：  40MHZ
            带宽：     60MHZ
            发送功率：  15w
            信噪比阈值：30dB
            接受信号阈值：4e-7w26.2
            噪声系数：  5dB
            天线增益：  2dB

        """
        super(Radio, self).__init__()
        # 10, 'circle', 100, 100, 200
        self.point = Simulation(1000, 'circle', 100, 100, 200)
        self.point.plot_inter_and_outer_point()
        self.num = len(self.point.inter_position)
        self.f = [10, 20, 30, 40, 50, 60, 70, 80, 110, 150, 200, 230, 270, 350, 400]
        self.bw = [5, 5, 5, 5, 5, 5, 5, 5, 20, 20, 20, 30, 30, 30, 40]
        self.rpt = 4e-7
        self.sp = [0, 0, 0, 5, 5, 5, 5, 5, 10, 10, 10, 15, 15, 20, 20]
        self.dis = []

    def cal_dis(self):
        for i in range(self.num):
            temp = euclidean_distances(self.point.inter_position, [self.point.inter_position[i]])
            self.dis.append(temp.copy())

    def remove_idx(self, arr1, arr2):
        """
        去掉arr1移除arr2中的元素
        :param arr1:
        :param arr2:
        :return:
        """
        arr1 = arr1.tolist()
        arr2 = arr2.tolist()
        idx = [x for x in arr1 if x not in arr2]
        return np.array(idx)

    def remove_list_idx(self, list1, list2):
        "从list2中移除list1的元素"
        idx = [x for x in list2 if x not in list1]
        return np.array(idx)

    def jammer(self, bw):
        """
        干扰机的模拟
        参数：
            干扰机频率：   50MHZ
            带宽：        40MHZ
            发送功率：     10w
            天线增益：     10dB
        :return:
        """
        jmf = [40, 80, 150, 200, 260, 320, 400]
        jmbw = [30, 30, 40, 40,  50, 60, 80]
        jmsp = [22, 30, 30, 30, 30, 30, 30]
        # jmdb = 10
        jmf_idx = np.random.randint(len(jmf), size=(len(self.point.outer_position), ))
        jmf_np = np.array(jmf)[jmf_idx]
        jmbw_np = np.array(jmbw)[jmf_idx]
        jmsp_np = np.array(jmsp)[jmf_idx]
        # jmdb_np = np.array([jmdb] * len(self.point.outer_position))
        jmrp_list = []
        jm_set = []
        for i in range(self.num):
            temp = euclidean_distances(self.point.outer_position, [self.point.inter_position[i]]).reshape(-1)
            # dis_jammers.append(temp.copy())
            # 计算干扰机传输损耗
            jmpl = 32.4 + 20 * np.log10(jmf_np) + 20 * np.log10(temp) - 14
            # 计算干扰的发射功率，dBm值
            jmsp_dBm = 10 * np.log10(jmsp_np * 1000)
            # 计算接收功率，单位dBm
            jmrp = jmsp_dBm - jmpl + 12
            # 计算干扰频带覆盖率，带宽相除即可
            repeat_bw = jmbw_np / bw[i]
            # 超过1的值改为1
            idx = np.where(repeat_bw >= 1)
            repeat_bw[idx[0]] = 1
            # print(repeat_bw)
            pj = jmrp * repeat_bw
            jm_idx = np.where(pj > -33.98)
            jm_set.append(jm_idx[0].tolist().copy())
            jmrp_list.append(pj.tolist().copy())
        # print(jm_set)
        return jmrp_list, jm_set

    def receive_power(self, f, sp, bw):
        """
        计算路径损耗
        :return:
        """
        self.cal_dis()
        # 接收功率为0
        p_idx = np.where(sp == 0)
        rp_list = []
        index_set = []
        # 用来取接收功率的索引
        success = []
        communication = []
        # 用来表示邻接关系的索引，是顶点的实际值
        hyper_idx = []
        for i in range(self.num):
            # 去掉自己再计算
            # 要考虑同信道问题，在同一个信道传输才会产生干扰，而且发送功率不为0
            new_f = np.delete(f, i)
            new_dis = np.delete(self.dis[i], i)
            # 如何判断是否在同一个信道？
            idx = np.where(new_f == f[i])
            # 如果同信道的设备发射功率都为0，那么就不受到干扰，不需要计算任何的干扰值
            new_idx = self.remove_idx(idx[0], p_idx[0])
            index_set.append(new_idx.tolist().copy())
            if new_idx.shape[0] > 0:
                # 计算路径损失，单位dB
                pl = 32.4 + 20 * np.log10(new_f[new_idx]) + 20 * np.log10(new_dis[new_idx]) - 4
                # pr = sp[new_idx] / np.power(10, 0.1*pl)
                # 计算发送功率的dBm
                sp_dBm = 10 * np.log10(sp[new_idx] * 1000)
                # 计算接收功率，单位dBm
                rp = sp_dBm - pl + 4
                rp_list.append(rp.tolist().copy())

                success_idx = np.where(rp > -36.98)
                communication_idx = np.nanargmax(rp)
                hyper_idx.append(new_idx[success_idx[0]].tolist().copy())
                success.append(success_idx[0].tolist().copy())
                # success.append([i for i in range(new_idx.shape[0])])
                communication.append([communication_idx])
                # print(success_idx[0])
                # print(communication_idx)
                # pl_list.append(pl.tolist().copy())
                # print('f', new_f[new_idx])
                # print('dis', new_dis[new_idx])
                # print('sp', sp[new_idx])
                # print('pl', pl)
                # print('sp_dbm', sp_dBm)
                # print('rp', rp)

            else:
                # 这里添加0项，单位不是dBm
                rp_list.append([0])
                # 如果没有收到其他电台的信号，则不受干扰
                communication.append([])
                success.append([])
                hyper_idx.append([i])

        # 计算噪声功率
        KTB = -114 + 10 * np.log10(bw)
        pb = np.power(10, (KTB+5)/10) * 0.001
        # print(pb)
        # pb = 10 * np.log10(pb*1000)
        # print(pb)
        # print(success)
        # print(communication)
        # print(len(hyper_idx))
        # print(hyper_idx)
        for i in range(self.num):
            hyper_idx[i].insert(0, i)
        # print(hyper_idx)
        return rp_list, pb, success, communication, hyper_idx

    def SNR(self, communication, success, rp, jm_set, jmrp, pb):
        # 先将dBm转换为w进行计算
        snr = []
        label = []
        # print(success)
        # print(communication)
        for i in range(self.num):
            rp_np = np.array(rp[i])
            rp_w = np.power(10, 0.1 * rp_np) * 0.001
            jmrp_np = np.array(jmrp[i])
            jmrp_w = np.power(10, 0.1 * jmrp_np) * 0.001
            point_i = rp_w[communication[i]]
            if point_i.shape[0] == 0:
                label.append(0)
            else:
                remove_i_idx = self.remove_list_idx(communication[i], success[i])

                if remove_i_idx.shape[0] == 0:
                    point_y = [0]
                else:
                    point_y = rp_w[remove_i_idx]
                jammers = jmrp_w[jm_set[i]]
                # print('-' * 20)
                # print(point_i[0])
                # print(np.sum(point_y))
                # print(np.sum(jammers))
                # print(point_i[0] / (np.sum(point_y) + np.sum(jammers) + pb[i]))
                snr = 10 * np.log10(point_i[0] / (np.sum(point_y) + np.sum(jammers) + pb[i]))
                if snr >= 30:
                    label.append(0)
                else:
                    label.append(1)

        label = np.array(label)
        count = np.where(label == 1)
        print(len(count[0])/500)
        return label

    def trans2bin(self, num):
        num = int(num * 1000) / 1000
        # print(num)
        integer_num = int(num)
        bin_1 = bin(integer_num)
        fra_num = num - integer_num
        bins = []
        for i in range(3):
            fra_num = fra_num * 2
            bins.append(1 if fra_num >= 1 else 0)
            fra_num = fra_num - int(fra_num)

        bin_list = list(bin_1[2:])
        bin_list = [int(x) for x in bin_list]
        fill_list = [0] * (11 - len(bin_list))
        bin_list = fill_list + bin_list

        bin_list = bin_list + bins
        return bin_list

    def one_hot(self, data):
        """
        one-hot编码
        :param data:
        :return:
        """

        N = data.shape[0]
        max_idx = int(np.max(data))
        # data = data.tolist()
        one_hot = np.zeros(shape=(N, max_idx+1))
        for i in range(N):
            one_hot[i][int(data[i])] = 1
        return one_hot

    def dec2bin(self):
        """
        将十进制数转换为二进制
        :return:
        """
        position = self.point.inter_position
        position = position.tolist()
        p_one_hot = []
        # print(position)
        for i in range(self.num):
            p_one_hot.append(self.trans2bin(position[i][0]) + self.trans2bin(position[i][1]))
        for i in range(self.num):
            p_one_hot[i] = [int(x) for x in p_one_hot[i]]
        # print(len(p_one_hot[0]))
        return np.array(p_one_hot)

    def feature(self, index_f, bw_index, sp_index, jm_set):
        """
        特征构造部分
        :return:
        """
        f_one_hot = self.one_hot(index_f)
        bw_one_hot = self.one_hot(bw_index)
        sp_one_hot = self.one_hot(sp_index)
        pos_one_hot = self.dec2bin()
        jm_one_hot = []
        for i in range(self.num):
            if len(jm_set) == 0:
                jm_one_hot.append([0, 0, 0, 0])
            else:
                idx = np.zeros(shape=(4, ), dtype='int32')
                idx[jm_set[i]] = 1
                jm_one_hot.append(idx.tolist().copy())
        feats = np.hstack([f_one_hot, bw_one_hot, sp_one_hot, pos_one_hot, jm_one_hot])
        # (100, 52)
        print(feats.shape)
        return feats

    def get_index(self, aim, value):
        """
        坐标转换
        :param aim:
        :param value:
        :return:
        """
        inx = np.zeros(shape=(value.shape[0], ))
        for i in range(len(aim)):
            new = np.where(value == aim[i])
            inx[new[0]] = i
        return inx

    def write_files(self, content, path):
        """
        序列化数据
        :param path:
        :return:
        """
        with open(path, 'wb') as f:
            plk.dump(content, f)

    def parameters(self):
        index_f = np.random.randint(0, len(self.f), size=(self.num, ))
        f = np.array(self.f)[index_f]
        bw = np.array(self.bw)[index_f]
        bw_idx = self.get_index([5, 20, 30, 40], bw)
        # index_sp = np.random.randint(0, len(self.sp), size=(self.num, ))
        sp = np.array(self.sp)[index_f]
        sp_idx = self.get_index([0, 5, 10, 15, 20], sp)
        rpt = np.array([self.rpt]*self.num)
        radio = np.stack((f, bw, rpt, sp), axis=0).transpose()

        rp, pb, success, communication, edge_list = self.receive_power(f, sp, bw)
        # print(rp)
        jmrp, jm_set = self.jammer(bw)
        # print(jmrp)
        # print(index_set)
        label = self.SNR(communication, success, rp, jm_set, jmrp, pb)
        feats = self.feature(index_f, bw_idx, sp_idx, jm_set)
        names_pre = ['prediction_label', 'prediction_feats', 'prediction_edge_list']
        names_tra = ['train_label', 'train_feats', 'train_edge_list']
        ob = [label, feats, edge_list]
        for i in range(len(names_tra)):
            self.write_files(ob[i], 'Data/15/{}.content'.format(names_pre[i]))

        # self.write_files(label, 'Data/prediction_label.content')
        # self.write_files(feats, 'Data/prediction_feats.content')
        # self.write_files(edge_list, 'Data/prediction_edge_list.content')

        # return radio


if __name__ == '__main__':
    r = Radio()
    r.parameters()
    np.random.seed(100)
