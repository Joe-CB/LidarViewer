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

def getTimeByFilename(vfm_filename):
    vfm_filepath, vfm_filename = os.path.split(vfm_filename)
    vfm_shotname, vfm_extension = os.path.splitext(vfm_filename)
    vfm_shotname = vfm_shotname.split('.')
    data = vfm_shotname[1][:10]
    data = data.split('-')
    return [int(idx) for idx in data]


def is_vfm(vfm_filename):
    # 打开文件
    filename, type = os.path.splitext(vfm_filename)
    if type != '.hdf':
        return False
    return True


def get_utc_time(hdf_url):
    begin = 0
    end = 0

    try:
        vfm_file = SD(hdf_url)
    except HDF4Error:
        return -1, -1
    else:
        sds_utc_time = vfm_file.select('Profile_UTC_Time')
        return sds_utc_time[0].min(), sds_utc_time[-1].max()


def get_sci_time(hdf_url):
    begin = 0
    end = 0

    try:
        vfm_file = SD(hdf_url)
    except HDF4Error:
        return -1, -1
    else:
        sds_sci_time = vfm_file.select('Profile_Time')
        return sds_sci_time[0].min(), sds_sci_time[-1].max()

def get_sci_array(hdf_filename):
    try:
        vfm_file = SD(hdf_filename)
    except HDF4Error:
        return -1, -1
    else:
        sds_sci_time = vfm_file.select('Profile_Time')
        return sds_sci_time.get()

def get_vfm_coor(vfm_filename):
    # 1. p_fileAddress：vfm数据的文件地址
    # 2. p_range_box 经纬度范围
    try:
        dt = SD(vfm_filename)
        # SDS Reading(经纬度，分类数据)
        latSds = dt.select('ssLatitude')
        lonSds = dt.select('ssLongitude')
        latitude = latSds.get()
        longitude = lonSds.get()
        return longitude, latitude
    except HDF4Error:
        print('VFM READ ERROR:')
        print(vfm_filename)
        print("Unexpected error:", sys.exc_info()[0])
        return None, None


def generate_altitude():
    first_range = np.arange(0, 290) * 0.03 - 0.5 + 0.015
    second_range = np.arange(0, 200) * 0.06 + 8.2 + 0.03
    third_range = np.arange(0, 55) * 0.18 + 20.2 + 0.09
    alt = np.concatenate((
        first_range,
        second_range,
        third_range),
        axis=0
    )
    alt = np.flipud(alt)
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



if __name__ == "__main__":
    pass
