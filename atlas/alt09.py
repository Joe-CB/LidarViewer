from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False

import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import math
import atlas.utils as icesat_utils


class Profile:
    '''
    Handle profile data of ALT09.
    如果已经从文件中读取过，那么会进行备份来加快处理速度。
    '''
    def __init__(self, profile_hdf_obj):
        self.file = profile_hdf_obj
        self.high_rate = {}
        self._surf_type = None
        self._solar_elevation = None

    @property
    def solar_elevation(self):
        if self._solar_elevation is None:
            self._solar_elevation = self.file['high_rate/solar_elevation']
        return self._solar_elevation

    @property
    def surf_type(self):
        if self._surf_type is None:
            self._surf_type = self.file['high_rate/surf_type']
        return self._surf_type

    def getHighRate(
            self,
            name:str,
    )->np.ndarray:
        if self.high_rate.get(name, None) is None:
            self.high_rate[name] = self.file[f'high_rate/{name}']
        return self.high_rate[name][:]

    def getHighRateObj(
            self,
            name:str,
    )->np.ndarray:
        if self.high_rate.get(name, None) is None:
            self.high_rate[name] = self.file[f'high_rate/{name}']
        return self.high_rate[name][:]

    def getHighRateObj(
            self,
            name: str,
    ):
        if self.high_rate.get(name, None) is None:
            self.high_rate[name] = self.file[f'high_rate/{name}']
        return self.high_rate[name]

    def getCaliAttenBackscatter(self, index=None):
        if index is None:
            return self.file['high_rate/cab_prof'][:]
        else:
            return self.file['high_rate/cab_prof'][index]

    def getLongtitude(self):
        return self.file['high_rate/longitude']

    def getLatitude(self):
        return self.file['high_rate/latitude']

    def showLayers(self, *, range=None):
        pass

    def drawCAB(
            self,
            *,
            index=None,
            axis_x:str=None,
            axis_y:str='alt',
    ):
        '''

        :param index:
        :param axis_x: label of x. enum['index', 'lats', 'lons']
        :param axis_y: label of y. enum['index', 'alt']
        :return:
        '''
        lons = self.getLongtitude()[:]
        lats = self.getLatitude()[:]
        surf_type = self.surf_type[:]
        cloud_fold_flag = self.getCloudFoldFlag()[:]
        cbs = self.getCaliAttenBackscatter()[:]
        if index is not None:
            lons = lons[index]
            lats = lats[index]
            surf_type = surf_type[index]
            cloud_fold_flag = cloud_fold_flag[index]
            cbs = self.getCaliAttenBackscatter()[index]

        if cbs.size <= 0:  # cbs log.
            return
        cbs = np.flip(cbs, axis=1)
        invalid_index = cbs == cbs[0][0]
        minus_index = cbs < 0
        cbs = np.log10(cbs)
        cbs[minus_index] = -13
        cbs[invalid_index] = -14
        cbs[cbs > 0] = -14
        cbs[np.isnan(cbs)] = -1
        cbs = np.rot90(cbs, k=1)

        fig, ax = plt.subplots(figsize=(10, 2), dpi=430)
        colors = ["white", "gold", "yellow", "red"]
        cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)
        psm = plt.imshow(cbs, cmap=cmap1)
        if axis_x == 'latitude':
            '''Labels-x'''
            label_index = np.arange(0, cbs[0], 200).astype(int)
            ax.set_xticks(label_index)
            ax.set_xticklabels(label_index)
            ax.set_xticklabels(lats[label_index])
            ax.set_xlabel('Latitude(°)')
        elif axis_x == 'profile_id':
            label_index = np.arange(0, cbs[0], 200).astype(int)
            ax.set_xticks(label_index)
            ax.set_xticklabels(label_index)
            ax.set_xlabel('Profile Sequence')
        else:
            ax.set_xlabel('Profile Sequence')
        fig.colorbar(psm, ax=ax)
        plt.plot(np.arange(0, cbs.shape[1], 1), np.zeros(cbs.shape[1])+667, 'g')
        plt.show()

    def getCloudFoldFlag(self):
        return self.file['high_rate/cloud_fold_flag']

    def drawLayers(
            self,
            *,
            index: np.ndarray = None,
            paint_cloud_fold: bool = False,
            paint_sea_height: bool = False,
            verbose=False,
            output=None,
            lat_ax_reso:float=500,
            profile_axis_type:str='profile_id',
    ):
        '''

        :param index:
        :param paint_cloud_fold:
        :param paint_sea_height:
        :param verbose:
        :param output:
        :param lat_ax_reso The Axis-Latitude resolution. default is 500 m.
        :param profile_axis_type: This enum is used to declear the labels of axis-x.
        :return:
        '''
        layer_attr = self.getHighRate('layer_attr')
        layer_top = self.getHighRate('layer_top')
        layer_bot = self.getHighRate('layer_bot')
        lats = self.getHighRate('latitude')
        lons = self.getHighRate('longitude')
        surface_height = self.getHighRate('surface_height')
        surface_bin = self.getHighRate('surface_bin')

        if index is not None:
            layer_attr = layer_attr[index]
            layer_top = layer_top[index]
            layer_bot = layer_bot[index]
            lats = lats[index]
            lons = lons[index]
            surface_height = surface_height[index]
            surface_bin = surface_bin[index]

        # 1st. Generate Feature Mask
        width = layer_attr.shape[0]
        feature = np.zeros((width, 700), dtype=np.int)
        for i in range(width):
            for k in range(10):
                if layer_attr[i, k] != 0:
                    top = int((layer_top[i, k] - 250) / 30) + 100
                    bot = int((layer_bot[i, k] - 250) / 30) + 100
                    if (layer_attr[i, k] == 3):
                        feature[i, bot:top] = 1         # 此处不做风吹雪的判定
                    else:
                        feature[i, bot:top] = layer_attr[i, k]

        # 2st. Paint Feature Mask
        feature_size = feature.shape
        css = np.rot90(feature, k=1)

        # css = css * 3 - 1
        # css = cv2.resize(css, size, interpolation=cv2.INTER_NEAREST)
        # lat = cv2.resize(lat[:, np.newaxis], (1, size[0]), interpolation=cv2.INTER_NEAREST)
        # lat = lat.flatten()

        colors = ['purple', 'gray', 'white']
        cmap = mpl.colors.ListedColormap(colors)
        norm = mpl.colors.Normalize(vmin=0, vmax=2)
        fig, ax = plt.subplots(figsize=(10, 2), dpi=100)
        # ax0 = ax.imshow(css, cmap=cmap)
        ax0 = ax.imshow(css, cmap=cmap, norm=norm)

        if profile_axis_type == 'latitude':
            '''Labels-x'''
            label_index = np.arange(0, feature_size[0], 200).astype(int)
            ax.set_xticks(label_index)
            ax.set_xticklabels(label_index)
            ax.set_xticklabels(lats[label_index])
            ax.set_xlabel('Latitude(°)')

        elif profile_axis_type == 'profile_id':
            label_index = np.arange(0, feature_size[0], 200).astype(int)
            ax.set_xticks(label_index)
            ax.set_xticklabels(label_index)
            ax.set_xlabel('Profile Sequence')

        '''Labels-y'''
        alt = np.arange(21, -3, -0.03)
        y_index = np.arange(700, 0, -lat_ax_reso/30).astype(int)
        alt = np.round(np.arange(-3, 18, lat_ax_reso/1000) * 10) / 10
        ax.set_yticks(y_index)
        ax.set_yticklabels(alt)
        '''title'''
        ax.set_ylabel('Altitude(km)')
        '''ColorBar'''
        cbar = fig.colorbar(ax0, orientation='horizontal', shrink=0.7)
        # $是斜体
        colorbar_label = ['$Clear air$', '$Cloud$', '$Aerosol$']
        # 设置坐标轴
        cbar.ax.set_xlabel(colorbar_label)
        cbar.ax.get_xaxis().set_ticks([])
        cbar.ax.get_yaxis().set_ticks([])
        # 插入文本
        for j, lab in enumerate(colorbar_label):
            cbar.ax.text(j/2+0.5, 1., lab, ha='center', va='center')
        cbar.ax.get_xaxis().labelpad = 1
        # 设置坐标轴label
        cbar.ax.set_xlabel('Feature Type')

        # 3th. Paint Sea Surface
        if paint_sea_height:
            lines = []
            idx = 0
            while idx < len(surface_height):
                line = []
                while idx < len(surface_height):
                    if surface_height[idx] > 20000:
                        if len(line) > 1:
                            lines.append(line)
                        line = []
                    else:
                        line.append([idx, surface_height[idx]])
                    idx += 1
            if len(line) > 1:
                lines.append(line)

            for line in lines:
                line = np.array(line)
                x = line[:, 0]
                y = 700 - (line[:, 1] - 250) / 30 - 100
                # y[:] = 100
                if len(x) >= 1:
                    plt.plot(x, y, 'r-')
                else:
                    plt.plot(x, y, '*r')

        plt.title('Mask of Cloud and Aerosol(Using ICESAT-2)')
        output = None
        if output is None:
            plt.show()
        else:
            plt.savefig(output, dpi=2500)

class ATL09:
    """
    This class is the tools for ATL09 data.

    Func:
    get_layers()
    get_bsc()       # back scatter

    Attributs:
    info: information of the alt09 dataset.
    """


    def __init__(self, filename):
        self.filename = filename
        try:
            self.filename = filename
            self.file = h5py.File(filename, 'r')
        except :
            print(f'File read error...{self.filename}')
            self.file=None

    @property
    def info(self):
        return self.filename

    def check(self):
        def decorator(func):
            def wrapper(*args, **kw):
                if not self.exist:
                    print(f'Check Error: {self.info}')
                    return None
                else:
                    return func(*args, **kw)

            return wrapper

        return decorator

    def getProfile(self,idx):
        root_dir = f'profile_{int(idx)}'
        return Profile(self.file[root_dir])

counter = 0


def TEST_BackScatter(filename):
    (filepath, tempfilename) = os.path.split(filename)
    (shotname, extension) = os.path.splitext(tempfilename)

    alt09_data = ATL09(filename)
    profile1 = alt09_data.getProfile(idx=3)

    lons = profile1.getLongtitude()[:]
    lats = profile1.getLatitude()[:]
    surf_type = profile1.surf_type[:]

    index = np.logical_and(np.logical_and(lons >= 120, lons <= 125), surf_type[:, 1] == 1)


    # index = np.logical_and(index, np.logical_and(latitude >= 35, latitude <= 40))
    surf_type = surf_type[index]
    icesat_utils.Painter.drawCoordination(lons[index], lats[index])

    cbs = profile1.getCaliAttenBackscatter()[index]
    cloud_fold_flag = profile1.getCloudFoldFlag()[index]
    index = np.where(cloud_fold_flag > 0)

    if cbs.size <= 0:
        return

    cbs = np.flip(cbs, axis=1)
    invalid_index = cbs == cbs[0][0]
    minus_index = cbs < 0
    cbs = np.log10(cbs)
    cbs[minus_index] = -13
    cbs[invalid_index] = -14
    cbs[cbs > 0] = -14

    # plt.figure(0)
    # plt.hist(cbs.flatten(), bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
    # plt.show()

    cbs[np.isnan(cbs)] = -1
    cbs = np.rot90(cbs, k=1)
    plt.figure()
    plt.imshow(cbs)
    plt.plot(index[0], np.zeros(index[0].shape), '*r')
    global counter
    plt.savefig(f'{counter}_{shotname}.jpg', dpi=2500)
    # plt.show()


def printAOI(filename):
    '''
    功能：输出纬度上30-60的中纬度海区测量数据，排除脉冲混叠情况，并输出为图片
    目的：来寻找合适的海雾案例
    :param filename:
    :return: None
    '''
    (filepath, tempfilename) = os.path.split(filename)              # 加载数据
    (shotname, extension) = os.path.splitext(tempfilename)
    alt09_data = ATL09(filename)
    profile1 = alt09_data.getProfile(idx=3)
    lons = profile1.getLongtitude()[:]
    lats = profile1.getLatitude()[:]
    surf_type = profile1.surf_type[:]
    cloud_fold_flag = profile1.getCloudFoldFlag()[:]

    abs_lats = np.abs(lats)
    index = np.logical_and(              # 条件筛选
        np.logical_and(
            abs_lats >= 30,
            abs_lats <= 60,
        ),
        np.logical_and(
            surf_type[:, 1] == 1,
            cloud_fold_flag == 0,
        ),
    )
    # icesat_utils.Painter.drawCoordination(lons[index], lats[index])
    cbs = profile1.getCaliAttenBackscatter()[index]

    if cbs.size <= 0:                       # cbs log.
        return
    cbs = np.flip(cbs, axis=1)
    invalid_index = cbs == cbs[0][0]
    minus_index = cbs < 0
    cbs = np.log10(cbs)
    cbs[minus_index] = -13
    cbs[invalid_index] = -14
    cbs[cbs > 0] = -14
    cbs[np.isnan(cbs)] = -1
    cbs = np.rot90(cbs, k=1)

    start = 0
    plt.figure()  # save to image
    while start < cbs.shape[1]:
        end = start + 1500
        if end > cbs.shape[1]:
            end = cbs.shape[1]
        plt.cla()
        plt.imshow(cbs[:, start:end])
        global counter
        plt.savefig(f'K:/icesat/alt09_image/{counter}_{start}_{shotname}.jpg', dpi=500)
        # plt.show()
        start += 1500
    plt.close('all')


def print_night_aoi(filename):
    '''
    功能：输出纬度上30-60的中纬度海区测量数据，排除脉冲混叠情况，并输出为图片
    目的：来寻找合适的海雾案例
    :param filename:
    :return: None
    '''
    (filepath, tempfilename) = os.path.split(filename)              # 加载数据
    (shotname, extension) = os.path.splitext(tempfilename)
    alt09_data = ATL09(filename)
    profile1 = alt09_data.getProfile(idx=3)
    lons = profile1.getLongtitude()[:]
    lats = profile1.getLatitude()[:]
    surf_type = profile1.surf_type[:]
    cloud_fold_flag = profile1.getCloudFoldFlag()[:]
    solar_elevation = profile1.solar_elevation[:]

    abs_lats = np.abs(lats)
    index = np.logical_and(              # 条件筛选
        np.logical_and(
            abs_lats >= 30,
            abs_lats <= 60,
        ),
        np.logical_and(
            surf_type[:, 1] == 1,
            cloud_fold_flag == 0,
        ),
    )
    index = np.logical_and(index, solar_elevation < -7.0)
    # icesat_utils.Painter.drawCoordination(lons[index], lats[index])
    cbs = profile1.getCaliAttenBackscatter()[index]

    if cbs.size <= 0:                       # cbs log.
        return
    cbs = np.flip(cbs, axis=1)
    invalid_index = cbs == cbs[0][0]
    minus_index = cbs < 0
    cbs = np.log10(cbs)
    cbs[minus_index] = -13
    cbs[invalid_index] = -14
    cbs[cbs > 0] = -14
    cbs[np.isnan(cbs)] = -1
    cbs = np.rot90(cbs, k=1)

    start = 0
    plt.figure()  # save to image
    while start < cbs.shape[1]:
        end = start + 1500
        if end > cbs.shape[1]:
            end = cbs.shape[1]
        plt.cla()
        plt.imshow(cbs[:, start:end])
        global counter
        plt.savefig(f'D:/icesat/alt09_night_image/{counter}_{start}_{shotname}.jpg', dpi=500)
        # plt.show()
        start += 1500
    plt.close('all')


def print_day_aoi(filename):
    '''
    功能：输出纬度上30-60的中纬度海区测量数据，排除脉冲混叠情况，并输出为图片
    目的：来寻找合适的海雾案例
    :param filename:
    :return: None
    '''
    (filepath, tempfilename) = os.path.split(filename)              # 加载数据
    (shotname, extension) = os.path.splitext(tempfilename)
    alt09_data = ATL09(filename)
    profile1 = alt09_data.getProfile(idx=3)
    lons = profile1.getLongtitude()[:]
    lats = profile1.getLatitude()[:]
    surf_type = profile1.surf_type[:]
    cloud_fold_flag = profile1.getCloudFoldFlag()[:]
    solar_elevation = profile1.solar_elevation[:]

    abs_lats = np.abs(lats)
    index = np.logical_and(              # 条件筛选
        np.logical_and(
            abs_lats >= 30,
            abs_lats <= 60,
        ),
        np.logical_and(
            surf_type[:, 1] == 1,
            cloud_fold_flag == 0,
        ),
    )
    index = np.logical_and(index, solar_elevation > 0)
    # icesat_utils.Painter.drawCoordination(lons[index], lats[index])
    cbs = profile1.getCaliAttenBackscatter()[index]

    if cbs.size <= 0:                       # cbs log.
        return
    cbs = np.flip(cbs, axis=1)
    invalid_index = cbs == cbs[0][0]
    minus_index = cbs < 0
    cbs = np.log10(cbs)
    cbs[minus_index] = -13
    cbs[invalid_index] = -14
    cbs[cbs > 0] = -14
    cbs[np.isnan(cbs)] = -1
    cbs = np.rot90(cbs, k=1)

    start = 0
    plt.figure()  # save to image
    while start < cbs.shape[1]:
        end = start + 1500
        if end > cbs.shape[1]:
            end = cbs.shape[1]
        plt.cla()
        plt.imshow(cbs[:, start:end])
        global counter
        plt.savefig(f'I:/icesat/alt09_day_image/{counter}_{start}_{shotname}.jpg', dpi=500)
        # plt.show()
        start += 1500
    plt.close('all')


if __name__ == "__main__":
    # filename = 'G:/icesat/ATL09/ATL09_20200724114313_04450801_003_01.h5'
    # TEST_BackScatter(filename)


    DIR1 = 'I:/icesat/ATL09_201910'
    files = os.listdir(DIR1)
    for file in files:
        print(file)
        counter += 1
        # if counter < 254:
        #     continue
        # file = 'ATL09_20191008130832_01780501_003_01.h5'
        file = 'ATL09_20191021214339_03820501_003_01.h5'
        # TEST_BackScatter(DIR1+'/'+file)
        # printAOI(DIR1 + '/' + file)
        print_night_aoi(DIR1 + '/' + file)

'''
fog instance:
ATL09_20191001033347_00650501_003_01.h5
'''


