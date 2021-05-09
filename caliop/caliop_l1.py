from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题

import os
import sys
import numpy as np
import numba
from numba import jit
from matplotlib import pyplot as plt
import matplotlib as mpl
import cv2
from pyhdf.SD import SD, SDC
from pyhdf.error import *


def generate_altitude():
    alt = []
    alt.extend(-2 + 0.3 * np.arange(0, 583-578, 1) + 0.15)
    alt.extend(-0.5 + 0.03 * np.arange(0, 578-288, 1) + 0.015)
    alt.extend(8.3 + 0.06 * np.arange(0, 288-88, 1) + 0.03)
    alt.extend(20.2 + 0.18 * np.arange(0, 88-33, 1) + 0.09)
    alt.extend(30.1 + 0.3 * np.arange(0, 33, 1) + 0.15)
    alt = np.array(alt)
    alt = np.around(alt, decimals=2)
    alt = alt[::-1]
    return alt


class CaliopL1:
    '''
    该类用来打开CALIOP L1b v4.0数据，
    '''
    def __init__(self):
        pass

    @staticmethod
    def getByName(filename, dataname):
        try:
            dt = SD(filename)
            # SDS Reading(TAI时间，经纬度，分类数据)
            xxx_sds = dt.select(dataname)
        except:
            print(f'VFM READ ERROR:[filename]')
            print("Unexpected error:", sys.exc_info()[0])
            return None
        return xxx_sds.get()

    @staticmethod
    def getTAB(filename):
        return CaliopL1.getByName(filename, 'Total_Attenuated_Backscatter_532')

    @staticmethod
    def getKeyName(filename):
        '''
        key name 是一个字符串，使用该字符串可以区分CALIOP不同产品是否为同一数据
        :param filename:
        :return:
        '''
        path, filename = os.path.split(filename)
        return filename[26:47]

if __name__ == "__main__":
    pass
