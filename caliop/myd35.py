import numpy as np
from matplotlib import pyplot as plt
import os.path
from pyhdf.SD import SD, SDC
from pyhdf.HDF import *
from pyhdf.VS import *
import pyflann as pf
import sys
import rs.rs_util as rs_util
from mpl_toolkits.basemap import Basemap, shiftgrid, cm

class Myd35_reader:
    def __init__(self, fileadress):
        self.filename = fileadress
        self.succeed = False

        #open
        try:
            myd35_data = SD(self.filename)
            # 1.SDS Reading
            cloud_mask_sds = myd35_data.select('Cloud_Mask')  # 经纬度
            self.cloud_mask = cloud_mask_sds.get()
            self.succeed = True
        except:
            print('Modis READ ERROR......(myd021km):' + self.filename)
            print("Unexpected error:", sys.exc_info()[0])

    def isValid(self):
        return self.succeed

    def get_cloud_mask(self):
        cloud_mask = np.bitwise_and(self.cloud_mask[0, ...], 7)
        cloud_mask = np.right_shift(cloud_mask, 1)
        return cloud_mask


    def plot_mask(self, filename=''):
        # mask_pic = np.
        plt.imshow(self.get_cloud_mask())
        if filename == '':
            plt.show()
        else:
            plt.savefig(filename)

if __name__ == "__main__":
    filename = r'E:\myd35\2016\MYD35_L2.A2016161.2330.061.2018059094842.hdf'
    m35 =  Myd35_reader(filename)
    m35.plot_mask()