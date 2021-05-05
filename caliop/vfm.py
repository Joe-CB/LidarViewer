from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题
import numpy as np
from pyhdf.SD import SD, SDC
from pyhdf.error import *
import os
import sys
import numba
from numba import jit
from matplotlib import pyplot as plt
import matplotlib as mpl
import cv2

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


class VfmOpener:
    # 构造函数
    def __init__(self, p_fileAddress):
        # 1. p_fileAddress：vfm数据的文件地址
        # 2. p_range_box 经纬度范围
        self.succeed = False
        self.filename = p_fileAddress
        vfm_filepath, vfm_filename = os.path.split(p_fileAddress)
        vfm_shotname, vfm_extension = os.path.splitext(vfm_filename)
        # 2.1 检验
        if vfm_extension != '.hdf':
            return
        # 2.2数据产品验证(未)

        # 2.3hdf Reading
        tai_timeArr = 0
        latArr = 0
        lonArr = 0
        classifyArr = 0
        try:
            dt = SD(self.filename)
            # SDS Reading(TAI时间，经纬度，分类数据)
            profile_time = dt.select('Profile_Time')
            # UTC_time = dt.select('ssProfile_UTC_Time')
            latSds = dt.select('ssLatitude')
            lonSds = dt.select('ssLongitude')
            Feature_Classification_Flags = dt.select('Feature_Classification_Flags')
            # get() Data
            self.tai_timeArr = profile_time.get()  # TAI time
            self.latArr = latSds.get()  # latitude
            self.lonArr = lonSds.get()  # longtitude
            self.classifyArr = Feature_Classification_Flags.get()
            self.unpacked_classify_arr = None
            self.succeed = True
        except:
            self.succeed = False
            print('VFM READ ERROR:')
            print(vfm_shotname)
            print("Unexpected error:", sys.exc_info()[0])
            return

    def getClassify(self):
        '''
        将vfm压缩后的数据展开
        :return: np.ndarray((profile_size * vertical_size), dtype=np.float)
                info:Vertical Feature Map.
        '''
        if self.unpacked_classify_arr is None:
            classify = self.classifyArr
            # classify = np.left_shift(classify, 13)
            classify = np.bitwise_and(classify, 7)
            classify_unpacked = np.zeros([classify.shape[0] * 15, 545], dtype=np.int16)
            counter = 0
            for block_idx in range(classify.shape[0]):
                block = classify[block_idx]
                temp_5km = np.empty([15, 545], dtype=np.int16)
                # [0, 55)
                for idx in range(0, 15):
                    begin = 55 * int(idx / 5)
                    end = 55 * (int(idx / 5) + 1)
                    temp_5km[idx, range(0, 55)] = block[range(begin, end)]
                # [55, 255)
                augment = 165
                for idx in range(0, 15):
                    begin = 200 * int(idx / 3) + augment
                    end = 200 * (int(idx / 3) + 1) + augment
                    temp_5km[idx, range(55, 255)] = block[range(begin, end)]
                # [255, 545)
                augment = 1165
                for idx in range(0, 15):
                    begin = 290 * int(idx / 1) + augment
                    end = 290 * (int(idx / 1) + 1) + augment
                    temp_5km[idx, range(255, 545)] = block[range(begin, end)]
                begin = counter * 15
                end = begin + 15
                counter += 1
                classify_unpacked[range(begin, end), ...] = temp_5km
            self.unpacked_classify_arr = classify_unpacked
        return self.unpacked_classify_arr



    @staticmethod
    def getLandWaterMask(filename):
        '''

        :param filename:
        :return:
        '''
        try:
            dt = SD(filename)
            # SDS Reading(经纬度，分类数据)
            land_water_mask_sds = dt.select('ssLand_Water_Mask')
            land_water_mask = land_water_mask_sds.get()
            return land_water_mask[:,0]
        except HDF4Error:
            print(f'VFM READ ERROR:{filename}')
            print("HDF4Error:", sys.exc_info()[0])
            return None

    def getClassifyByRange(self, range_box):

        innerFlag = (self.lonArr > range_box[0]) & (self.lonArr < range_box[1]) & (self.latArr > range_box[2]) & (self.latArr < range_box[3])
        innersize = 0
        for flag in innerFlag:
            if flag == True:
                innersize += 1
        if innersize == 0:
            return
        classify = self.classifyArr
        # classify = np.left_shift(classify, 13)
        classify = np.bitwise_and(classify, 7)
        classify_unpacked = np.empty([innersize*15,545], dtype = np.int16)
        counter = 0
        for block_idx in range(classify.shape[0]):
            if innerFlag[block_idx] == False:
                continue
            block = classify[block_idx]
            temp_5km = np.empty([15, 545], dtype = np.int16)
            # [0, 55)
            for idx in range(0,15):
                begin = 55 * int(idx/5)
                end = 55 * (int(idx/5)+1)
                temp_5km[idx, range(0, 55)] = block[range(begin, end)]
            # [55, 255)
            augment = 165
            for idx in range(0,15):
                begin = 200 * int(idx / 3) + augment
                end = 200 * (int(idx / 3) + 1) + augment
                temp_5km[idx, range(55, 255)] = block[range(begin, end)]
            # [255, 545)
            augment = 1165
            for idx in range(0, 15):
                begin = 290 * int(idx / 1) + augment
                end = 290 * (int(idx / 1) + 1) + augment
                temp_5km[idx, range(255, 545)] = block[range(begin, end)]
            begin = counter*15
            end = begin+15
            counter += 1
            classify_unpacked[range(begin, end), ...] = temp_5km
        return classify_unpacked



    @staticmethod
    def draw(classify, lon, lat, size=(1100, 545), output=None):
        '''
        绘制连续的vfm图
        :param classify: 分类数据
        :param lon: 经度
        :param lat: 维度
        :param size: 绘制图的尺寸
        :param output: 图片输出位置，如果为None那么输出到屏幕
        :return: None
        '''
        css = np.rot90(classify, k=3)
        css = cv2.resize(css, size, interpolation=cv2.INTER_NEAREST)
        lat = cv2.resize(lat[:, np.newaxis], (1, size[0]), interpolation=cv2.INTER_NEAREST)
        lat = lat.flatten()
        colors = ['purple', 'blue', 'white', 'gray', 'pink', 'green', 'tab:brown', 'red']
        cmap = mpl.colors.ListedColormap(colors)
        norm = mpl.colors.Normalize(vmin=0, vmax=8)
        fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
        # ax0 = ax.imshow(css, cmap=cmap)
        ax0 = ax.imshow(css, cmap=cmap, norm=norm)
        '''Labels-x'''
        label_index = np.arange(0, size[0], size[0] / 8).astype(int)
        ax.set_xticks(label_index)
        ax.set_xticklabels(lat[label_index])
        '''Labels-y'''
        alt = generate_altitude()
        alt = cv2.resize(alt[:, np.newaxis], (1, size[1]), interpolation=cv2.INTER_NEAREST)
        alt = alt.flatten()
        y_index = np.arange(0, size[1], 100).astype(int)
        ax.set_yticks(y_index)
        ax.set_yticklabels((alt[y_index]*100).astype(int) / 100)

        '''title'''
        ax.set_xlabel('纬度(°)')
        ax.set_ylabel('海拔(km)')
        '''ColorBar'''
        cbar = fig.colorbar(ax0, orientation='horizontal', shrink=0.7)
        # $是斜体
        colorbar_label = ['$invalid$', '$clear Air$', '$cloud$', '$t-Aerosol$', '$s-Aerosol$', '$surface$',
                          '$subsurface$', '$no-signal$']
        # 设置坐标轴
        cbar.ax.set_xlabel(colorbar_label)
        cbar.ax.get_xaxis().set_ticks([])
        cbar.ax.get_yaxis().set_ticks([])
        # 插入文本
        for j, lab in enumerate(colorbar_label):
            cbar.ax.text(j + 0.5, 1.5, lab, ha='center', va='center')
        cbar.ax.get_xaxis().labelpad = 1
        # 设置坐标轴label
        cbar.ax.set_xlabel('Feature Type')

        if output is None:
            plt.show()
        else:
            plt.savefig(output, dpi=2500)

    @staticmethod
    def drawGray(classify, lon, lat, size=(1100, 545), output=None):
        '''
        绘制连续的vfm图
        :param classify: 分类数据
        :param lon: 经度
        :param lat: 维度
        :param size: 绘制图的尺寸
        :param output: 图片输出位置，如果为None那么输出到屏幕
        :return: None
        '''
        css = np.rot90(classify, k=3)
        css = cv2.resize(css, size, interpolation=cv2.INTER_NEAREST)
        lat = cv2.resize(lat[:, np.newaxis], (1, size[0]), interpolation=cv2.INTER_NEAREST)
        lat = lat.flatten()
        colors = ['purple', 'blue', 'white', 'gray', 'pink', 'green', 'tab:brown', 'red']
        colors = [(i * 0.1, i * 0.1, i * 0.1) for i in range(3, 11)]
        colors[2], colors[7] = colors[7], colors[2]

        cmap = mpl.colors.ListedColormap(colors)
        norm = mpl.colors.Normalize(vmin=0, vmax=8)
        fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
        # ax0 = ax.imshow(css, cmap=cmap)
        ax0 = ax.imshow(css, cmap=cmap, norm=norm)
        '''Labels-x'''
        label_index = np.arange(0, size[0], size[0] / 8).astype(int)
        ax.set_xticks(label_index)
        ax.set_xticklabels(lat[label_index])
        '''Labels-y'''
        alt = generate_altitude()
        alt = cv2.resize(alt[:, np.newaxis], (1, size[1]), interpolation=cv2.INTER_NEAREST)
        alt = alt.flatten()
        y_index = np.arange(0, size[1], 100).astype(int)
        ax.set_yticks(y_index)
        ax.set_yticklabels((alt[y_index] * 100).astype(int) / 100)

        '''title'''
        ax.set_xlabel('纬度(°)')
        ax.set_ylabel('海拔(km)')
        '''ColorBar'''
        cbar = fig.colorbar(ax0, orientation='horizontal', shrink=0.7)
        # $是斜体
        colorbar_label = ['$invalid$', '$clear Air$', '$cloud$', '$t-Aerosol$', '$s-Aerosol$', '$surface$',
                          '$subsurface$', '$no-signal$']
        # 设置坐标轴
        cbar.ax.set_xlabel(colorbar_label)
        cbar.ax.get_xaxis().set_ticks([])
        cbar.ax.get_yaxis().set_ticks([])
        # 插入文本
        for j, lab in enumerate(colorbar_label):
            cbar.ax.text(j + 0.5, 1.5, lab, ha='center', va='center')
        cbar.ax.get_xaxis().labelpad = 1
        # 设置坐标轴label
        cbar.ax.set_xlabel('Feature Type')

        if output is None:
            plt.show()
        else:
            plt.savefig(output, dpi=2500)

    @staticmethod
    def imshow(vfm_filename, range_box=None, output=None):
        vfm_dataset = VfmOpener(vfm_filename)
        if vfm_dataset.succeed == False:
            print(f'Read Vfm error:{vfm_filename}')
            return
        if range_box is None:
            css = vfm_dataset.getClassify()
        else:
            css = vfm_dataset.getClassifyByRange(range_box)

        css = np.rot90(css, k=3)
        css = cv2.resize(css, (2400, 1800), interpolation=cv2.INTER_NEAREST)
        colors = ['purple', 'blue', 'white', 'gray', 'pink', 'green', 'tab:brown', 'red']
        cmap = mpl.colors.ListedColormap(colors)
        norm = mpl.colors.Normalize(vmin=0, vmax=8)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax0 = ax.imshow(css, cmap=cmap, norm=norm)
        cbar = fig.colorbar(ax0, orientation='horizontal', shrink=0.5)
        # $是斜体
        colorbar_label = ['$invalid$', '$clear Air$', '$cloud$', '$t-Aerosol$', '$s-Aerosol$', '$surface$',
                          '$subsurface$', '$no-signal$']
        # 设置坐标轴
        cbar.ax.set_xlabel(colorbar_label)
        cbar.ax.get_xaxis().set_ticks([])
        cbar.ax.get_yaxis().set_ticks([])
        # 插入文本
        for j, lab in enumerate(colorbar_label):
            cbar.ax.text(j+0.5, 1.5, lab, ha='center', va='center')
        cbar.ax.get_xaxis().labelpad = 1
        # 设置坐标轴label
        cbar.ax.set_xlabel('Feature Type')

        if output is None:
            plt.show()
        else:
            plt.savefig(output, dpi=2500)

@jit(nopython=True)
def getUnpackedClassify(classify, classify_unpacked):
    for i in range(classify.shape[0]):
        for j in range(15):
            idx = i * 15 + j
            idx_0 = int(j / 5)
            idx_1 = int(j / 3)
            idx_2 = j

            # range 0
            for k in range(55):
                classify_unpacked[idx, k] = classify[i, k + idx_0*55]
                # print(k + idx_0*55)

            for k in range(0, 200):
                classify_unpacked[idx, k+55] = classify[i, k + idx_1 * 200 + 165]
                # print(k + idx_1 * 200 + 165)

            for k in range(0, 290):
                classify_unpacked[idx, k+255] = classify[i, k + idx_2 * 290 + 1165]
                # print(k + idx_2 * 290 + 1165)
    return classify_unpacked


@jit(nopython=True)
def getCloudHeightByIndex(features, indices, heights):
    '''
    给定一个序列，计算其云高
    :param features: vfm的2D map
    :param indices: 请求的索引
    :param heights: 云高，默认已赋值为-1
    :return: None
    '''
    for i in range(indices.shape[0]):
        heights[i] = -1
        for j in range(features[i, :].shape[0]):
            if features[i, j] == 2:
                heights[i] = j
                break


class VfmCloudOpener(VfmOpener):
    def __init__(self, file):
        VfmOpener.__init__(self, file)

    @staticmethod
    def get_latitude():
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

    @property
    def cloud_height(self):
        if not hasattr(self, '_cloud_height'):
            feature_type = self.getClassify()
            indices_length = int(feature_type.shape[0])
            if indices_length == 0:
                return None
            cloud_height_map = np.zeros((indices_length), dtype=np.int)
            for i in range(indices_length):
                feature = feature_type[i, :]
                bin_idx = 0
                while bin_idx < feature.shape[0]:
                    if feature[bin_idx] == 2:  # 2 是云的标识符
                        break
                    else:
                        bin_idx += 1
                if bin_idx == feature.shape[0]:
                    bin_idx = -1  # 无云
                cloud_height_map[i] = bin_idx
            self._cloud_height = cloud_height_map

        return self._cloud_height

    def getCloudHeight(self, *, idx):
        feature = self.getClassify()[idx,:]

        indices = np.where(feature == 2)[0]
        if indices.size == 0:
            return -1
        else:
            return int(indices[0])

    def getCloudHeights(self, *, indices: np.ndarray=None):
        if indices is None:
            indices = np.arange(self.classifyArr.shape[0] * 15)
        heights = np.zeros(indices.shape[0])

        classify = self.classifyArr
        classify = np.bitwise_and(classify, 7)

        # QA_flag = np.bitwise_and(classify, 24)
        # QA_flag = np.right_shift(QA_flag, 3)
        # classify[np.logical_and(QA_flag<=2, QA_flag>0)] = 1

        classify_unpacked = np.empty([classify.shape[0] * 15, 545], dtype=np.int16)
        features = getUnpackedClassify(classify, classify_unpacked)

        # features =self.getClassify()
        getCloudHeightByIndex(features, indices, heights)
        return heights

        # np.sum(features2!=features)

    def getClassifyJit(self):
        classify = self.classifyArr
        classify = np.bitwise_and(classify, 7)
        classify_unpacked = np.empty([classify.shape[0] * 15, 545], dtype=np.int16)
        classify_unpacked = getUnpackedClassify(classify, classify_unpacked)
        return classify_unpacked

    def getFeatureFlag(self):
        classify = self.classifyArr
        classify = np.bitwise_and(classify, 7)
        classify_unpacked = np.empty([classify.shape[0] * 15, 545], dtype=np.int16)
        getUnpackedClassify(classify, classify_unpacked)
        return classify_unpacked

    @staticmethod
    def getLaserEnergy(filename):
        '''

        :param filename: 文件名
        :return: 小于80 mJ的为能量太弱的光束
        '''
        try:
            dt = SD(filename)
            # SDS Reading(经纬度，分类数据)
            laser_energy_sds = dt.select('ssLaser_Energy_532')
            laser_energy = laser_energy_sds.get()
            return laser_energy[:,0]
        except HDF4Error:
            print(f'VFM READ ERROR:{filename}')
            print("HDF4Error:", sys.exc_info()[0])
            return None



if __name__ == "__main__":
    vfm_file = r'D:/cal/cal_2015\CAL_LID_L2_VFM-Standard-V4-20.2015-05-06T22-54-20ZD.hdf'
    range_box = [-180 - 15, -150, 60, 80]
    VfmOpener.imshow(vfm_file,range_box=range_box)
    vfm_data = VfmCloudOpener(vfm_file)
    vfm_data.getCloudHeights(indices=np.arange(10000))