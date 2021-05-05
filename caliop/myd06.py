import numpy as np
from pyhdf.SD import SD, SDC
from pyhdf.error import HDF4Error
from pyhdf.HDF import HDF
import os
import sys


def is_myd06(filename):
    fp, tmpfilename = os.path.split(filename)
    file_str_list = tmpfilename.split('.')
    if len(file_str_list)<=0:
        return False
    productname = file_str_list[0]
    if productname == 'MYD06':
        return True
    else:
        return False



class Myd06:
    def __init__(self, p_myd06):
        self.succeed = False
        self.myd06_filename = p_myd06
        # 读取MYD03并处理
        try:
            # Lontitude, Latitude
            self.succeed = True
        except HDF4Error:
            print('Modis READ ERROR......(myd03):' + self.myd06_filename)
            print("Unexpected error:", sys.exc_info()[0])
            return

    @property
    def cloud_top_height(self):
        if not hasattr(self, '_cloud_top_height'):
            try:
                dt = SD(self.myd06_filename)
                cloud_top_height_sds = dt.select('Cloud_Top_Height')
                cth_attr = cloud_top_height_sds.attributes()
                valid_range = cth_attr.get('valid_range')
                scale = cth_attr.get('scale_factor')
                off_set = cth_attr.get('add_offset')
                self._cloud_top_height = cloud_top_height_sds.get()
                mask = np.logical_or(self._cloud_top_height >= valid_range[1], self._cloud_top_height <= valid_range[0])
                self._cloud_top_height =  self._cloud_top_height.astype(float)
                self._cloud_top_height[mask] = np.nan

                self._cloud_top_height = (self._cloud_top_height - off_set) * scale * 0.001
                dt.end()
            except:
                print('Read Error!!!')
                return None
        return self._cloud_top_height

    @property
    def latitude(self):
        if not hasattr(self, '_latitude'):
            try:
                dt = SD(self.myd06_filename)
                latitude_ds = dt.select('Latitude')
                self._latitude = latitude_ds.get()
                dt.end()
            except:
                print('Read Error!!!')
                return None
        return self._latitude

    @property
    def longitude(self):
        if not hasattr(self, '_longitude'):
            try:
                dt = SD(self.myd06_filename)
                longitude_ds = dt.select('Longitude')
                self._longitude = longitude_ds.get()
                dt.end()
            except:
                print('Read Error!!!')
                return None
        return self._longitude


if __name__ == '__main__':
    filename = r'F:\shared_dir\myd06\MYD06_L2.A2016129.1705.061.2018058075720.hdf'
    file_obj = Myd06(filename)
    print(file_obj.cloud_top_height)

