import numpy as np
from pyhdf.SD import SD, SDC
from pyhdf.error import HDF4Error
from pyhdf.HDF import *
# import pyhdf.HDF as HDF
import pyhdf.VS as VS
import os
import sys
import rs.rs_util as rs_util
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap, shiftgrid, cm
import time

def get_myd1km_sci_time(myd1km_filename_str):
    try:  # open hdf
        f = HDF(myd1km_filename_str, SDC.READ)
        vs = f.vstart()
        data_info_list = vs.vdatainfo()
        # Vset table
        # L1B swam matedata
        L1B_Swath_Matedata_VD = vs.attach('Level 1B Swath Metadata')
        # Read [Swath/scan type]
        begin = L1B_Swath_Matedata_VD[0]
        begin = begin[4]
        end = L1B_Swath_Matedata_VD[-1]
        end = end[4]
        L1B_Swath_Matedata_VD.detach()  # __del__ with handle this.
        vs.end()  # Use with, so __del__ can do this.
        f.close()
        return begin, end, True

    except HDF4Error:
        print("Unexpected error:", sys.exc_info()[0])
        print('READ ERROR......:' + myd1km_filename_str)
        return 0, 0, False

def get_struct_time(myd1km_filename):
    '''
    返回当前文件的测量时间
    :param myd1km_filename: 文件地址
    :return: time.struct_time
    '''
    if not is_myd021km(myd1km_filename):
        return None
    fp, tmpfilename = os.path.split(myd1km_filename)
    file_str_list = tmpfilename.split('.')
    if len(file_str_list) <= 2:
        return None
    try:
        year_month = file_str_list[1]
        hour_min = file_str_list[2]

        year = year_month[1:5]
        day = year_month[5:]
        hour = hour_min[0:2]
        min = hour_min[2:]
        time_struct = time.strptime(f"{year}-{day}-{hour}-{min}", "%Y-%j-%H-%M")
        return time_struct
    except:
        print(f'Get MYD021KM error!!!{myd1km_filename}')
        return None


def get_solor_angle(myd1km_filename_str):
    try:  # open hdf
        myd021km_data = SD(myd1km_filename_str)
        # 1.SDS Reading
        zenith_sds = myd021km_data.select('SensorZenith')  # 经纬度
        zenith = zenith_sds.get()  # 2
        azimuth_sds = myd021km_data.select('SensorAzimuth')  # 经纬度
        azimuth = azimuth_sds.get()

        return zenith * 0.01, azimuth * 0.01


        f = HDF(myd1km_filename_str, SDC.READ)
        vs = f.vstart()
        data_info_list = vs.vdatainfo()
        # Vset table
        # L1B swam matedata
        L1B_Swath_Matedata_VD = vs.attach('Level 1B Swath Metadata')
        # Read [Swath/scan type]
        sd_info = L1B_Swath_Matedata_VD.inquire()
        all_metadata = L1B_Swath_Matedata_VD.read(sd_info[0])
        L1B_Swath_Matedata_VD.detach()  # __del__ with handle this.
        vs.end()  # Use with, so __del__ can do this.
        f.close()
        return all_metadata

    except HDF4Error:
        print("Unexpected error:(get_solor_angle)", sys.exc_info()[0])
        print('READ ERROR......:' + myd1km_filename_str)
        return None

def get_swath_metadata(myd1km_filename_str):
    try:  # open hdf
        f = HDF(myd1km_filename_str, SDC.READ)
        vs = f.vstart()
        data_info_list = vs.vdatainfo()
        # Vset table
        # L1B swam matedata
        L1B_Swath_Matedata_VD = vs.attach('Level 1B Swath Metadata')
        # Read [Swath/scan type]
        sd_info = L1B_Swath_Matedata_VD.inquire()
        all_metadata = L1B_Swath_Matedata_VD.read(sd_info[0])
        L1B_Swath_Matedata_VD.detach()  # __del__ with handle this.
        vs.end()  # Use with, so __del__ can do this.
        f.close()
        return all_metadata

    except HDF4Error:
        print("Unexpected error:", sys.exc_info()[0])
        print('READ ERROR......:' + myd1km_filename_str)
        return None

def get_modis_time(filename):
    '''
    Get all Servey time of all MODIS file.
    :param filename: MODIS filename
    :return: Servey time
    '''
    fp, tmpfilename = os.path.split(filename)
    shortname, extension = os.path.splitext(tmpfilename)
    str_list = shortname.split('.')
    if len(str_list) > 3:
        return str_list[1] + str_list[2]

def is_myd021km(filename):
    '''
    检测当前文件是不是MYD021KM数据产品
    :param filename: 文件地址
    :return:
    '''
    fp, tmpfilename = os.path.split(filename)
    shortname, extension = os.path.splitext(tmpfilename)
    file_str_list = tmpfilename.split('.')
    if len(file_str_list) <= 0:
        return False
    productname = file_str_list[0]
    if productname == 'MYD021KM' and extension == '.hdf':
        return True
    else:
        return False

class Myd1km_Reader:
    '''
    [ADD Verification].(Values bigger than 32767 is invalid.)
    '''
    def __init__(self, p_myd021km_filename):
        self.succeed = False
        self.filename = p_myd021km_filename

        # 初始化图像
        # 读取modis——hdf
        try:
            myd021km_data = SD(self.filename)
            # 1.SDS Reading
            ev_250_SDS = myd021km_data.select('EV_250_Aggr1km_RefSB')  # 经纬度
            ev_500_SDS = myd021km_data.select('EV_500_Aggr1km_RefSB')  # 经纬度
            ev_1kmRef_SDS = myd021km_data.select('EV_1KM_RefSB')  # 经纬度
            ev_1km_SDS = myd021km_data.select('EV_1KM_Emissive')  # 经纬度
            '''250'''
            ev_250 = ev_250_SDS.get()  # 2
            for key, value in ev_250_SDS.attributes().items():
                if key == 'reflectance_offsets':
                    ev_250_offset = value
                if key == 'reflectance_scales':
                    ev_250_scale = value
            '''500'''
            ev_500 = ev_500_SDS.get()  # 2
            for key, value in ev_500_SDS.attributes().items():
                if key == 'reflectance_offsets':
                    ev_500_offset = value
                if key == 'reflectance_scales':
                    ev_500_scale = value
            '''100Ref'''
            ev_1kmRef = ev_1kmRef_SDS.get()  # 2
            for key, value in ev_1kmRef_SDS.attributes().items():
                if key == 'radiance_offsets':
                    ev_1kmRef_offset = value
                if key == 'radiance_scales':
                    ev_1kmRef_scale = value
            '''100EV'''
            ev_1km = ev_1km_SDS.get()  # 2
            for key, value in ev_1km_SDS.attributes().items():
                if key == 'radiance_offsets':
                    ev_1km_offset = value
                if key == 'radiance_scales':
                    ev_1km_scale = value

            # pack
            self.band_num = ev_250.shape[0]+ev_500.shape[0]+ev_1kmRef.shape[0]+ev_1km.shape[0]
            self.width = ev_250.shape[1]
            self.height = ev_250.shape[2]
            self.all_band = np.empty([self.band_num, self.width, self.height], dtype=np.uint16)
            self.scale = np.empty(self.band_num)
            self.offset = np.empty(self.band_num)
            now_band=0
            # 250
            for idx in range(ev_250.shape[0]):
                self.all_band[now_band, ...] = ev_250[idx,...]
                self.scale[now_band] = ev_250_scale[idx]
                self.offset[now_band] = ev_250_offset[idx]
                now_band+=1
            # print(now_band)
            # 500
            for idx in range(ev_500.shape[0]):
                self.all_band[now_band, ...] = ev_500[idx, ...]
                self.scale[now_band] = ev_500_scale[idx]
                self.offset[now_band] = ev_500_offset[idx]
                now_band+=1
            # print(now_band)
            # 1kmref
            for idx in range(ev_1kmRef.shape[0]):
                self.all_band[now_band, ...] = ev_1kmRef[idx, ...]
                self.scale[now_band] = ev_1kmRef_scale[idx]
                self.offset[now_band] = ev_1kmRef_offset[idx]
                now_band += 1
            # print(now_band)
            # 1km
            for idx in range(ev_1km.shape[0]):
                self.all_band[now_band, ...] = ev_1km[idx, ...]
                self.scale[now_band] = ev_1km_scale[idx]
                self.offset[now_band] = ev_1km_offset[idx]
                now_band += 1
            # print(now_band)
            self.succeed = True
        except:
            print('Modis READ ERROR......(myd021km):' + self.filename)
            print("Unexpected error:", sys.exc_info()[0])
            self.isValid = False

    def get_scaled(self):
        all_band = np.empty([self.band_num, self.width, self.height])
        for idx in range(self.band_num):
            all_band[idx, ...]= (self.all_band[idx,...]-self.offset[idx])*self.scale[idx]
        return all_band

    def get_not_scaled(self):
        return self.all_band

    def get_verify_flag(self, index_tuple=None):
        '''
        Make sure the index in smmall than 32767
        :param index_tuple:
        :return:
        '''
        if index_tuple is None:
            flag_valid = self.all_band[0, ...] < 32767
            for idx in range(self.band_num):
                flag_valid = flag_valid & (self.all_band[idx, ...] < 32767)
            return flag_valid
        else:
            flag_valid = self.all_band[0, index_tuple[0], index_tuple[1]] < 32767
            for idx in range(self.band_num):
                flag_valid = flag_valid & (self.all_band[idx, index_tuple[0], index_tuple[1]] < 32767)
            return flag_valid

    def get_by_index(self, index_tuple, verify=False, scaled=False):
        if verify:
            varify_flag = self.get_verify_flag(index_tuple)
            index_tuple = (index_tuple[0][varify_flag], index_tuple[1][varify_flag])
        if not scaled:
            return self.all_band[..., index_tuple[0], index_tuple[1]]
        else:
            raw_data = self.all_band[..., index_tuple[0], index_tuple[1]]
            scaled_data = np.empty(raw_data.shape)
            for idx in range(self.band_num):
                scaled_data[idx, ...] = (raw_data[idx, ...]-self.offset[idx])*self.scale[idx]
            return  scaled_data

    def getValidMap(
            self,
            *,
            band_index: np.ndarray = None,
            ):
        '''
        得到MODIS探测数据是否可用的二维布尔索引，代表了每个测量是否可用
        Info: Values bigger than 32767 is invalid.
        :param band_index:考虑的波段范围
        :return:
        '''
        if band_index is not None:
            band = self.all_band[band_index, ...]
        else:
            band = self.all_band

        flag_valid = band[0, ...] < 32767
        for idx in range(self.band_num):
            flag_valid = flag_valid & (self.all_band[idx, ...] < 32767)
        return flag_valid


    def get_train_data(self, index_tuple, band_select=[0,1,2,3,4,6,7,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]):
        '''
        [Varification & Night eliminite]
        :param index_tuple:
        :return:
        '''
        # [band_num, rowsize]
        raw_data = self.get_by_index(index_tuple, verify=False)
        # Verify
        raw_data = raw_data[band_select, ...]
        valid_flag = np.ones([raw_data.shape[1]], dtype=bool)
        for idx in range(raw_data.shape[0]):
            valid_flag = valid_flag&(raw_data[idx,...]>32767)
        # Daytime mode
        try:  # open hdf
            f = HDF(self.filename, SDC.READ)
            vs = f.vstart()
            data_info_list = vs.vdatainfo()
            # Vset table
            # L1B swam matedata
            L1B_Swath_Matedata_VD = vs.attach('Level 1B Swath Metadata')
            # Read [Swath/scan type]
            svath_matedata = L1B_Swath_Matedata_VD[:]
            for idx in range(valid_flag.shape[0]):
                if svath_matedata[int((index_tuple[0][idx])/10)][2] == 'D   ':
                    valid_flag[idx] = True
                else:
                    valid_flag[idx]=False
            L1B_Swath_Matedata_VD.detach()  # __del__ with handle this.
            vs.end()  # Use with, so __del__ can do this.
            f.close()
        except ValueError:
            print("Unexpected error:", sys.exc_info()[0])
            print('READ ERROR......(%d):' % self.filename)
            return 0, 0, False
        raw_data = self.get_by_index(index_tuple, verify=False, scaled=True)
        raw_data = raw_data[band_select, ...]
        return raw_data, valid_flag

    def get_super_pixel(self, i,j,band_select, kernel_radius = 10):
        indices=np.arange(-kernel_radius, kernel_radius+1)
        indices_i = indices + i
        indices_j = indices + j
        if (np.sum(indices_i<0)+np.sum(indices_i>=self.width)) > 0:
            return None, False
        if (np.sum(indices_j<0)+np.sum(indices_j>=self.height)) > 0:
            return None, False
        return self.all_band[band_select, indices_i, indices_j], True

    def get_train_image(self, band_select=[0, 1, 2, 3, 4, 6, 7, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]):
        raw_band = np.empty([self.width, self.height, len(band_select)], np.uint16)
        for idx in range(len(band_select)):
            raw_band[:,:,idx] = self.all_band[band_select[idx],:,:]

        # handle invalid data
        invalid_flag = np.empty([self.width, self.height], dtype=bool)
        invalid_flag[...]=False
        for idx in range(1,len(band_select)):
            invalid_flag = invalid_flag&(raw_band[...,idx]>32767)
        print('Invalid num is %d.'%np.sum(invalid_flag))

        scaled_data = np.empty([self.width, self.height, len(band_select)])
        for idx in range(len(band_select)):
            scaled_data[...,idx] = (raw_band[..., idx]-self.offset[band_select[idx]])*self.scale[band_select[idx]]
        scaled_data[invalid_flag, ...] = 0
        return scaled_data



    def get_train_data_map(self, index_tuple, kernel_radius = 5, band_select=[0, 1, 2, 3, 4, 6, 7, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]):
        '''
        [Varification & Night eliminite]
        :param index_tuple:
        :return:
        '''
        # [band_num, rowsize]
        selected_band_num = len(band_select)
        kernel_size = 2*kernel_radius+1
        # Verify
        valid_flag = np.ones([index_tuple[0].shape[0]], dtype=bool)
        # Daytime mode
        try:  # open hdf
            f = HDF(self.filename, SDC.READ)
            vs = f.vstart()
            data_info_list = vs.vdatainfo()
            # Vset table
            # L1B swam matedata
            L1B_Swath_Matedata_VD = vs.attach('Level 1B Swath Metadata')
            # Read [Swath/scan type]
            svath_matedata = L1B_Swath_Matedata_VD[:]
            for idx in range(valid_flag.shape[0]):
                if svath_matedata[int((index_tuple[0][idx]) / 10)][2] == 'D   ':
                    valid_flag[idx] = True
                else:
                    valid_flag[idx] = False
            L1B_Swath_Matedata_VD.detach()  # __del__ with handle this.
            vs.end()  # Use with, so __del__ can do this.
            f.close()
        except ValueError:
            print("Unexpected error:", sys.exc_info()[0])
            print('READ ERROR......(%d):' % self.filename)
            return 0, 0, False
        if np.sum(valid_flag) == 0:
            return None, valid_flag

        raw_data = np.empty([index_tuple[0].shape[0], selected_band_num, kernel_size, kernel_size])
        # Create a Mat of kernel
        kernel_i=np.empty([kernel_size, kernel_size], dtype = np.int16)
        for i in range(kernel_size):
            for j in range(kernel_size):
                kernel_i[i,j]=i
        kernel_i -= kernel_radius
        kernel_i = np.reshape(kernel_i, kernel_i.size)
        kernel_j = np.transpose(kernel_i)
        kernel_j = np.reshape(kernel_j, kernel_j.size)


        for idx in range(index_tuple[0].shape[0]):
            if not valid_flag[idx]:
                continue
            temp_kernel_i = kernel_i+index_tuple[0][idx]
            if np.sum((temp_kernel_i<0) | (temp_kernel_i>= self.width)) >0:
                valid_flag[idx] = False
                continue
            temp_kernel_j = kernel_j+index_tuple[1][idx]
            if np.sum((temp_kernel_j<0 ) | (temp_kernel_j>= self.height)) >0:
                valid_flag[idx] = False
                continue
            temp = (self.all_band[..., temp_kernel_i, temp_kernel_j])[band_select, ...]
            if np.sum(temp[..., int(kernel_size*kernel_size*0.5)] > 32767):
                valid_flag[idx] = False
            else:
                indeces = temp>32767
                temp[indeces] = 0
                raw_data[idx, ...] = temp.reshape(selected_band_num, kernel_size, kernel_size)
        for idx in range(selected_band_num):
            raw_data[:,idx, :,:] = (raw_data[:,idx, :,:]-self.offset[band_select[idx]]) * self.scale[band_select[idx]]
        return raw_data, valid_flag

    def get_by_band(self, band_indeces = [0, 3, 2]):
        band_data = np.empty([len(band_indeces), self.width, self.height])

        for idx in range(len(band_indeces)):
            band_idx = band_indeces[idx]
            one_band_data = self.all_band[band_idx, ...]
            # invalid data eliminate
            flag = one_band_data>32767
            one_band_data = (one_band_data-self.offset[band_idx])*self.scale[band_idx]
            # invalid data set to zero.
            one_band_data[flag] = 0
            band_data[idx, ] = one_band_data
        return band_data

    def get_true_color(self):
        return self.get_by_band([0,3,2])

    def get_true_color_uint8(self):
        true_color_float = self.get_by_band([0,3,2])
        # true_color_uint8 = np.empty(true_color_float.shape, dtype=np.uint8)
        min = np.min(true_color_float)
        min = 0
        max = np.max(true_color_float)
        scale = 255/(max-min)
        true_color_uint8 = ((true_color_float-min)*scale).astype(np.uint8)
        return true_color_uint8

    def get_false_color(self):
        return self.get_by_band([1,0,3])

    def get_lat_lon(self):
        # 初始化图像
        # 读取modis——hdf
        try:
            myd021km_data = SD(self.filename)
            # 1.SDS Reading
            latitude_sds = myd021km_data.select('Latitude')  # 经纬度
            longitude_sds = myd021km_data.select('Longitude')  # 经纬度
            latitude = latitude_sds.get()
            longitude = longitude_sds.get()
            # pack
        except:
            print('Modis READ ERROR......(myd021km):' + self.filename)
            print("Unexpected error:", sys.exc_info()[0])
            return None, None
        # x = cv2.imread('C:/Users/Joe-CB/Desktop/images.png')
        # print(x.shape)
        # latitude_resized = cv2.resize(latitude, (2030, 1354))
        # longitude_resized = cv2.resize(longitude, (2030, 1354))

        return longitude, latitude

class Myd1kmPaint(Myd1km_Reader):
    def __init__(self, filename):
        Myd1km_Reader.__init__(filename)

    def plot_true_color(self, image_fn=''):
        # Prepare Data.
        raw_image = self.get_true_color()
        min = np.min(raw_image)
        min = 0
        max = np.max(raw_image)
        scale = 255 / (max - min)
        for idx in range(raw_image.shape[0]):
            raw_image[idx, ...] = (raw_image[idx, ...] - min) * scale
        lons, lats = self.get_lat_lon()
        min_lon = np.min(lons)
        max_lon = np.max(lons)
        min_lat = np.min(lats)
        max_lat = np.max(lats)

        lons = rs_util.interpo2d(lons, [self.width, self.height])    # interpolation[2030, 1354].
        lats = rs_util.interpo2d(lats, [self.width, self.height])
        # Begin to plot.
        fig = plt.figure()
        center_lon = lons[int(lons.shape[0] / 2), int(lons.shape[1] / 2)]
        center_lat = lats[int(lats.shape[0] / 2), int(lats.shape[1] / 2)]
        # center_lon = (min_lon+max_lon)*0.5
        # center_lat = (min_lat+max_lat)*0.5

        # m = Basemap(projection='ortho', llcrnrx = 0, lat_0=center_lat, lon_0=center_lon)
        # llcrnrlon=min_lon,llcrnrlat=min_lat,urcrnrlon=max_lon,urcrnrlat=max_lat,
        # ortho
        # m._fulldisk = False
        # m = Basemap(projection='ortho',
        #            llcrnrlon = min_lon,llcrnrlat = min_lat,urcrnrlon = max_lon,urcrnrlat = max_lat,
        #            lat_0=center_lat, lon_0=center_lon)
        m = Basemap(width=12000000/2, height=9000000/2, projection='lcc',
                    resolution='h', lat_1=45., lat_2=55, lat_0=center_lat, lon_0=center_lon)


        x, y = m(lons, lats)    # Convert [lon, lat] to projection Position.
        # m.plot(x, y, marker='D',color='m')
        min_x = np.min(x)
        max_x = np.max(x)
        min_y = np.min(y)
        max_y = np.max(y)
        # set range.
        # m.llcrnrx = min_x-0.2*(max_x-min_x)
        # m.llcrnry = min_y-0.2*(max_y-min_y)
        # m.urcrnrx = max_x+0.2*(max_x-min_x)
        # m.urcrnry = max_y+0.2*(max_y-min_y)
        x_length = max_x - min_x
        y_length = max_y - min_y
        resolution = [500, 500]
        x_scale = (resolution[0] - 1) / x_length
        y_scale = (resolution[1] - 1) / y_length
        x_pos = (np.floor((x - min_x) * x_scale)).astype(int)
        y_pos = (np.floor((y - min_y) * y_scale)).astype(int)
        image = np.zeros([resolution[0], resolution[1], 3], dtype=np.uint8)
        image[...] = 255
        image[x_pos, y_pos, 0] = raw_image[0, ...]
        image[x_pos, y_pos, 1] = raw_image[1, ...]
        image[x_pos, y_pos, 2] = raw_image[2, ...]
        # m.drawmapboundary(fill_color='#99ffff')
        m.drawcoastlines(color='b')    # Draw Coast Line
        m.drawmeridians(np.arange(-180, 180, 30), labels=[1, 1, 0, 1])      # Draw meridians(经纬脉络)
        m.drawparallels(np.arange(-90, 90, 15), labels=[0, 0, 1, 0])
        x0 = np.min(x)
        x1 = np.max(x)
        y0 = np.min(y)
        y1 = np.max(y)
        extent = (x0, x1, y0, y1)
        # extent = (np.min(lons), np.max(lons), np.min(lats), np.max(lats))
        # m.imshow(image,  extent = extent)
        image=np.rot90(image, k=1, axes=(0, 1))
        # image = hisEqulColor(image)
        plt.imshow(image, extent=extent)
        if image_fn == '':
            plt.show()
        else:
            plt.savefig(image_fn, dpi=512)

    def plot_false_color(self, image_fn=''):
        # Prepare Data.
        raw_image = self.get_false_color()
        min = np.min(raw_image)
        min = 0
        max = np.max(raw_image)
        scale = 255 / (max - min)
        for idx in range(raw_image.shape[0]):
            raw_image[idx, ...] = (raw_image[idx, ...] - min) * scale
        lons, lats = self.get_lat_lon()
        min_lon = np.min(lons)
        max_lon = np.max(lons)
        min_lat = np.min(lats)
        max_lat = np.max(lats)

        lons = rs_util.interpo2d(lons, [self.width, self.height])    # interpolation[2030, 1354].
        lats = rs_util.interpo2d(lats, [self.width, self.height])
        # Begin to plot.
        fig = plt.figure()
        center_lon = lons[int(lons.shape[0] / 2), int(lons.shape[1] / 2)]
        center_lat = lats[int(lats.shape[0] / 2), int(lats.shape[1] / 2)]
        # center_lon = (min_lon+max_lon)*0.5
        # center_lat = (min_lat+max_lat)*0.5

        # m = Basemap(projection='ortho', llcrnrx = 0, lat_0=center_lat, lon_0=center_lon)
        # llcrnrlon=min_lon,llcrnrlat=min_lat,urcrnrlon=max_lon,urcrnrlat=max_lat,
        # ortho
        # m._fulldisk = False
        # m = Basemap(projection='ortho',
        #            llcrnrlon = min_lon,llcrnrlat = min_lat,urcrnrlon = max_lon,urcrnrlat = max_lat,
        #            lat_0=center_lat, lon_0=center_lon)
        m = Basemap(width=12000000/2, height=9000000/2, projection='lcc',
                    resolution='h', lat_1=45., lat_2=55, lat_0=center_lat, lon_0=center_lon)


        x, y = m(lons, lats)    # Convert [lon, lat] to projection Position.
        # m.plot(x, y, marker='D',color='m')
        min_x = np.min(x)
        max_x = np.max(x)
        min_y = np.min(y)
        max_y = np.max(y)
        # set range.
        # m.llcrnrx = min_x-0.2*(max_x-min_x)
        # m.llcrnry = min_y-0.2*(max_y-min_y)
        # m.urcrnrx = max_x+0.2*(max_x-min_x)
        # m.urcrnry = max_y+0.2*(max_y-min_y)
        x_length = max_x - min_x
        y_length = max_y - min_y
        resolution = [500, 500]
        x_scale = (resolution[0] - 1) / x_length
        y_scale = (resolution[1] - 1) / y_length
        x_pos = (np.floor((x - min_x) * x_scale)).astype(int)
        y_pos = (np.floor((y - min_y) * y_scale)).astype(int)
        image = np.zeros([resolution[0], resolution[1], 3], dtype=np.uint8)
        image[...] = 255
        image[x_pos, y_pos, 0] = raw_image[0, ...]
        image[x_pos, y_pos, 1] = raw_image[1, ...]
        image[x_pos, y_pos, 2] = raw_image[2, ...]
        # m.drawmapboundary(fill_color='#99ffff')
        m.drawcoastlines(color='b')    # Draw Coast Line
        m.drawmeridians(np.arange(-180, 180, 30), labels=[1, 1, 0, 1])      # Draw meridians(经纬脉络)
        m.drawparallels(np.arange(-90, 90, 15), labels=[0, 0, 1, 0])
        x0 = np.min(x)
        x1 = np.max(x)
        y0 = np.min(y)
        y1 = np.max(y)
        extent = (x0, x1, y0, y1)
        # extent = (np.min(lons), np.max(lons), np.min(lats), np.max(lats))
        # m.imshow(image,  extent = extent)
        image=np.rot90(image, k=1, axes=(0, 1))
        # image = hisEqulColor(image)
        plt.imshow(image, extent=extent)
        if image_fn == '':
            plt.show()
        else:
            plt.savefig(image_fn, dpi=512)


class MYDReader:
    '''
    定义了一些常用的属性访问方法
    '''
    def __init__(self, filename):
        self.filename = filename
        self._succeed = True
        try:
            self.science_data = SD(self.filename)
        except:
            print('Modis READ ERROR......(myd021km):' + self.filename)
            print("Unexpected error:", sys.exc_info()[0])
            self._succeed = False

    @property
    def succeed(self):
        return self._succeed

    @property
    def coeffs(self):
        '''
        返回辐射系数
        :return (scales: List[], offsets :List[]):
        '''
        if not hasattr(self, '_scale'):
            try:
                # select
                sd = self.science_data
                ev_250_SDS = sd.select('EV_250_Aggr1km_RefSB')  # 经纬度
                ev_500_SDS = sd.select('EV_500_Aggr1km_RefSB')  # 经纬度
                ev_1kmRef_SDS = sd.select('EV_1KM_RefSB')  # 经纬度
                ev_1km_SDS = sd.select('EV_1KM_Emissive')  # 经纬度

                scale_coeffs = []
                offset_coeffs = []

                ''''# Radiance
                scale_coeffs.extend(ev_250_SDS.attributes()['radiance_scales'])
                scale_coeffs.extend(ev_500_SDS.attributes()['radiance_scales'])
                scale_coeffs.extend(ev_1kmRef_SDS.attributes()['radiance_scales'])
                scale_coeffs.extend(ev_1km_SDS.attributes()['radiance_scales'])
                
                offset_coeffs.extend(ev_250_SDS.attributes()['radiance_offsets'])
                offset_coeffs.extend(ev_500_SDS.attributes()['radiance_offsets'])
                offset_coeffs.extend(ev_1kmRef_SDS.attributes()['radiance_offsets'])
                offset_coeffs.extend(ev_1km_SDS.attributes()['radiance_offsets'])
                '''

                # Reflect
                scale_coeffs.extend(ev_250_SDS.attributes()['reflectance_scales'])
                scale_coeffs.extend(ev_500_SDS.attributes()['reflectance_scales'])
                scale_coeffs.extend(ev_1kmRef_SDS.attributes()['reflectance_scales'])
                scale_coeffs.extend(ev_1km_SDS.attributes()['radiance_scales'])

                offset_coeffs.extend(ev_250_SDS.attributes()['reflectance_offsets'])
                offset_coeffs.extend(ev_500_SDS.attributes()['reflectance_offsets'])
                offset_coeffs.extend(ev_1kmRef_SDS.attributes()['reflectance_offsets'])
                offset_coeffs.extend(ev_1km_SDS.attributes()['radiance_offsets'])
                self._scale = (scale_coeffs, offset_coeffs)
            except:
                print(f'Get Offset&Scale Error!!!:{self.filename}')
                return None
        return self._scale

    @property
    def radiance_coeffs(self):
        '''
        返回辐射系数
        :return (scales: List[], offsets :List[]):
        '''
        if not hasattr(self, '_radiance_scale'):
            try:
                # select
                sd = self.science_data
                ev_250_SDS = sd.select('EV_250_Aggr1km_RefSB')  # 经纬度
                ev_500_SDS = sd.select('EV_500_Aggr1km_RefSB')  # 经纬度
                ev_1kmRef_SDS = sd.select('EV_1KM_RefSB')  # 经纬度
                ev_1km_SDS = sd.select('EV_1KM_Emissive')  # 经纬度

                scale_coeffs = []
                offset_coeffs = []

                # Radiance
                scale_coeffs.extend(ev_250_SDS.attributes()['radiance_scales'])
                scale_coeffs.extend(ev_500_SDS.attributes()['radiance_scales'])
                scale_coeffs.extend(ev_1kmRef_SDS.attributes()['radiance_scales'])
                scale_coeffs.extend(ev_1km_SDS.attributes()['radiance_scales'])

                offset_coeffs.extend(ev_250_SDS.attributes()['radiance_offsets'])
                offset_coeffs.extend(ev_500_SDS.attributes()['radiance_offsets'])
                offset_coeffs.extend(ev_1kmRef_SDS.attributes()['radiance_offsets'])
                offset_coeffs.extend(ev_1km_SDS.attributes()['radiance_offsets'])
                self._radiance_scale = (scale_coeffs, offset_coeffs)
            except:
                print(f'Get Offset&Scale Error!!!:{self.filename}')
                return None
        return self._radiance_scale


    @property
    def band(self):
        if not self.succeed:
            return None
        if not hasattr(self, '_band'):
            try:
                sd = self.science_data
                # select
                band = []
                ev_250_SDS = sd.select('EV_250_Aggr1km_RefSB')  # 经纬度
                ev_500_SDS = sd.select('EV_500_Aggr1km_RefSB')  # 经纬度
                ev_1km_ref_SDS = sd.select('EV_1KM_RefSB')  # 经纬度
                ev_1km_emiss_SDS = sd.select('EV_1KM_Emissive')  # 经纬度
                # access data
                band.extend(ev_250_SDS.get())
                band.extend(ev_500_SDS.get())
                band.extend(ev_1km_ref_SDS.get())
                band.extend(ev_1km_emiss_SDS.get())
                self._band = np.array(band)
            except:
                print(f'Read Band Error!!!.{self.filename}')
                self._band = None
        return self._band

    @property
    def scaled_band(self):
        if not hasattr(self, '_scaled_band'):
            band = self.band
            coeffs = self.coeffs
            if band is None or coeffs is None:
                self._scaled_band = None
            invalid_idx = np.where(band>32767)
            scales = coeffs[0]
            offsets = coeffs[1]

            scaled_band = np.zeros(band.shape, dtype=float)
            for i in range(len(scales)):
                scaled_band[i,...] = (band[i,...] - offsets[i]) * scales[i]
            scaled_band[invalid_idx] = np.nan
            self._scaled_band = scaled_band
        return self._scaled_band

    @property
    def radiance_scaled_band(self):
        if not hasattr(self, '_scaled_band'):
            band = self.band
            coeffs = self.radiance_coeffs
            if band is None or coeffs is None:
                self._scaled_band = None
            invalid_idx = np.where(band > 32767)
            scales = coeffs[0]
            offsets = coeffs[1]

            scaled_band = np.zeros(band.shape, dtype=float)
            for i in range(len(scales)):
                scaled_band[i, ...] = (band[i, ...] - offsets[i]) * scales[i]
            scaled_band[invalid_idx] = np.nan
            self._scaled_band = scaled_band
        return self._scaled_band

    @property
    def solar_zenith(self):
        if not self.succeed:
            return None
        if not hasattr(self, '_solar_zenith'):
            try:
                sd = self.science_data
                sds = sd.select('SolarZenith')  # 经纬度
                factor = sds.attributes()['scale_factor']
                self._solar_zenith = sds.get() * factor
            except:
                print(f'Read Band Error!!!.{self.filename}')
                self._solar_zenith = None
        return self._solar_zenith

    @property
    def solar_azimuth(self):
        if not self.succeed:
            return None
        if not hasattr(self, '_solar_azimuth'):
            try:
                sd = self.science_data
                sds = sd.select('SensorAzimuth')  # 经纬度
                factor = sds.attributes()['scale_factor']
                self._solar_azimuth = sds.get() * factor
            except:
                print(f'Read Band Error!!!.{self.filename}')
                self._solar_azimuth = None
        return self._solar_azimuth

    @property
    def lats(self):
        if not self.succeed:
            return None
        if not hasattr(self, '_lats'):
            try:
                sd = self.science_data
                sds = sd.select('Latitude')  # 经纬度
                self._lats = sds.get()
            except:
                print(f'Read Band Error!!!.{self.filename}')
                self._lats = None
        return self._lats

    @property
    def lons(self):
        if not self.succeed:
            return None
        if not hasattr(self, '_lons'):
            try:
                sd = self.science_data
                sds = sd.select('Longitude')  # 经纬度
                self._lons = sds.get()
            except:
                print(f'Read Band Error!!!.{self.filename}')
                self._lons = None
        return self._lons


import time

if __name__ == "__main__":

    filename = 'D:/MYD/2015/0506/MYD021KM.A2015121.0005.061.2018049112654.hdf'
    myd = MYDReader(filename)

    sz = myd.solar_zenith
    sa = myd.solar_azimuth

    coeffs = myd.coeffs
    scaled_band = myd.scaled_band
    band = myd.band

    start = time.time()
    for i in range(1):
        for j in range(10):
            valid_map = myd.getValue(i, j)
    end = time.time()
    print(f'Use time:{end - start}')
    k=10