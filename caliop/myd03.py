import numpy as np
from pyhdf.SD import SD, SDC
from pyhdf.error import HDF4Error
from pyhdf.HDF import HDF
import os
import sys


def is_myd03(filename):
    fp, tmpfilename = os.path.split(filename)
    file_str_list = tmpfilename.split('.')
    if len(file_str_list)<=0:
        return False
    productname = file_str_list[0]
    if productname == 'MYD03':
        return True
    else:
        return False

def get_modis_time(filename):
    '''
    Get all Servery time of all MODIS file.
    :param filename: MODIS filename
    :return: Servey time
    '''
    fp, tmpfilename = os.path.split(filename)
    shortname, extension = os.path.splitext(tmpfilename)
    str_list = shortname.split('.')
    if len(str_list) > 3:
        return str_list[1] + str_list[2]


class Myd03:
    def __init__(self, p_myd03):
        self.succeed = False
        self.myd03_filename = p_myd03
        # 读取MYD03并处理
        try:
            # Lontitude, Latitude
            dt = SD(self.myd03_filename)
            latSds = dt.select('Latitude')  # 经纬度
            lonSds = dt.select('Longitude')
            self.modis_latArr = latSds.get()  # latitude
            self.modis_lonArr = lonSds.get()  # longtitude
            dt.end()
            # width, height
            self.width = self.modis_latArr.shape[1]
            self.height = self.modis_latArr.shape[0]
            self.succeed = True
        except HDF4Error:
            print('Modis READ ERROR......(myd03):' + self.myd03_filename)
            print("Unexpected error:", sys.exc_info()[0])
            return




