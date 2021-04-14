# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time      :   2021/4/13 14:10
# @Author    :   Chasion
# Description:   用来生成数据
import numpy as np
from Data.simulation import Simulation
from sklearn.metrics.pairwise import euclidean_distances
C = 3.0*100000000


class Radio:
    def __init__(self, num):
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
            接受信号阈值：4e-7w
            噪声系数：  5dB
            天线增益：  2dB

        """
        super(Radio, self).__init__()
        self.point = Simulation(10, 'circle', 20, 20, 100)
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
        jmf = [60, 150, 260, 400]
        jmbw = [30, 40, 50, 60]
        jmsp = [10, 10, 15, 20]
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

        # 计算噪声功率
        KTB = -114 + 10 * np.log10(bw)
        pb = np.power(10, (KTB+5)/10) * 0.001
        # print(pb)
        # pb = 10 * np.log10(pb*1000)
        # print(pb)
        # print(success)
        # print(communication)
        return rp_list, pb, success, communication

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

        # label = np.array(label)
        # count = np.where(label == 1)
        # print(len(count[0]))
        return label

    def parameters(self):
        index_f = np.random.randint(0, len(self.f), size=(self.num, ))
        f = np.array(self.f)[index_f]
        bw = np.array(self.bw)[index_f]
        # index_sp = np.random.randint(0, len(self.sp), size=(self.num, ))
        sp = np.array(self.sp)[index_f]
        rpt = np.array([self.rpt]*self.num)
        radio = np.stack((f, bw, rpt, sp), axis=0).transpose()

        rp, pb, success, communication = self.receive_power(f, sp, bw)
        # print(rp)
        jmrp, jm_set = self.jammer(bw)
        # print(jmrp)
        # print(index_set)
        label = self.SNR(communication, success, rp, jm_set, jmrp, pb)
        return radio


if __name__ == '__main__':
    r = Radio(80)
    r.parameters()
    np.random.seed(100)
