import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib
import matplotlib.pyplot as plt
from numba import jit


def generate_latitude(
        *,
        tick_size=6,
):
    '''
    生成ICESAT 的海拔高度图
    :param resoluion: 海拔高度分辨率
    :return: alt_ticks, alt_ticklabels
    '''

    bin_max = 467
    ticks = np.arange(0, bin_max, 1)
    ticklabels = ticks * 0.03 - 0.25
    ticklabels = ticklabels[::-1]
    ticklabels = np.around(ticklabels, decimals=2)

    indices = np.arange(0, bin_max + bin_max/tick_size/2, bin_max/tick_size).astype(int)
    indices[-1] = bin_max - 1
    return ticks[indices], ticklabels[indices]


@jit(nopython=True)
def extract_valid_profile_jit(profile, begin_idx, end_idx, valid_profile):
    N = profile.shape[0]
    for i in range(N):
        b = begin_idx[i]
        e = end_idx[i]
        for j in range(b, e):
            valid_profile[i, j - b] = profile[i, j]
    return valid_profile


def extract_valid_profile(profile, begin_idx, end_idx, jit=True):
    '''

    :param profiles: 大气廓线数据
    :param begin_idx: 大气廓线有效值的开始位置
    :param end_idx: 大气廓线有效值的结束位置
    :return: 大气廓线
    '''
    if jit:
        valid_profile = np.zeros((profile.shape[0], 467), dtype=float)
        extract_valid_profile_jit(profile, begin_idx, end_idx, valid_profile)
        return valid_profile
    else:
        valid_profile = np.zeros((profile.shape[0], 467), dtype=float)
        for i in range(profile.shape[0]):
            b = begin_idx[i]
            e = end_idx[i]
            for j in range(b, e):
                valid_profile[i, j-b] = profile[i, j]
        return valid_profile

class AxLabels:
    def __init__(self, label, ticks, ticklabels):
        '''
        绘制坐标轴的参数（参数都可以为 None）
        :param label: 坐标轴名称
        :param xticks: 标注
        :param xticks_index: 标注位置
        '''
        if ticks is None or ticklabels is None:
            ticks = ticklabels = None
        else:
            assert ticklabels.shape == ticks.shape       # ticks和ticklabels是一一对应的，必须长度相等。
        self.ticks = ticks
        self.ticklabels = ticklabels
        self.label = label

class Painter:
    def __init__(self):
        '''
        通用的绘图类
        '''
        pass

    @staticmethod
    def drawCoordination(lons, lats):
        # map1 = Basemap(projection='mill', lon_0=180)

        plt.subplot(1,2,1)

        map1 = Basemap(projection='ortho',lon_0=117, lat_0=0,resolution='l')
        # draw coastlines, country boundaries, fill continents.
        map1.drawcoastlines(linewidth=0.25)
        map1.drawcountries(linewidth=0.25)
        map1.fillcontinents(color='coral', lake_color='aqua')
        map1.drawmapboundary(fill_color='aqua')
        map1.drawmeridians(np.arange(0, 360, 30))
        map1.drawparallels(np.arange(-90, 90, 30))
        cs = map1.plot(lons, lats, 'r.', latlon=True)
        plt.title('Draw lons and Lats in map.')

        plt.subplot(1, 2, 2)

        map1 = Basemap(projection='ortho', lon_0=-63, lat_0=0, resolution='l')
        # draw coastlines, country boundaries, fill continents.
        map1.drawcoastlines(linewidth=0.25)
        map1.drawcountries(linewidth=0.25)
        map1.fillcontinents(color='coral', lake_color='aqua')
        map1.drawmapboundary(fill_color='aqua')
        map1.drawmeridians(np.arange(0, 360, 30))
        map1.drawparallels(np.arange(-90, 90, 30))
        cs = map1.plot(lons, lats, 'r.', latlon=True)
        plt.title('Draw lons and Lats in map.')
        plt.show()

        # map1 = Basemap(projection='mill', lon_0=180)
        # # draw coastlines, country boundaries, fill continents.
        # map1.drawcoastlines(linewidth=0.25)
        # map1.drawcountries(linewidth=0.25)
        # map1.fillcontinents(color='coral', lake_color='aqua')
        # map1.drawmapboundary(fill_color='aqua')
        # map1.drawmeridians(np.arange(0, 360, 30), labels=[True, False, False, True])
        # map1.drawparallels(np.arange(-90, 90, 30), labels=[False, True, True, False])
        #
        #
        # cs = map1.plot(lons, lats, 'r.', latlon=True)
        # plt.title('Draw lons and Lats in map.')
        # plt.show()

    @staticmethod
    def paint_profile(
            profile: np.ndarray,
            *,
            title: str="Unknown-title",
            ax_y: AxLabels=AxLabels(None, None, None),
            ax_x: AxLabels=AxLabels(None, None, None),
            colorbar_label='Unknown',
            dpi=420
    ):
        '''
        该方法用于实现对profile数据的统一可视化
        :param profile: 廓线数据
        :param title: 绘图标题
        :param y_ax_labels: y坐标轴参数
        :param x_ax_labels: x坐标轴参数
        :return: None
        '''
        ticks_size = 5
        title_size = 6
        fontdict = {
            'fontsize': ticks_size,
        }
        title_fontdict = {
            'fontsize': title_size,
        }

        fig, ax = plt.subplots(figsize=(1, 1), dpi=dpi)
        # show    vmin=0, vmax=1,
        plt.imshow(profile, cmap='jet', interpolation='nearest')#vmin=13, vmax=19)
        # colorbar
        cb = plt.colorbar(ax=ax, )
        cb.ax.tick_params(labelsize=ticks_size)  #设置色标刻度字体大小。
        cb.set_label(colorbar_label)
        colorbar_ax = cb.ax
        text = colorbar_ax.yaxis.label
        font = matplotlib.font_manager.FontProperties(
            family='times new roman',
            style='italic',
            size=title_size,
        )
        text.set_font_properties(font)


        # y-label-latitude(y使用的海拔高度)
        if ax_y.label is None:
            ax.set_ylabel('Unknown', fontdict=title_fontdict)
        else:
            ax.set_ylabel(ax_y.label, fontdict=title_fontdict)
        if ax_y.ticks is not None and ax_y.ticklabels is not None:
            ax.set_yticks(ax_y.ticks)
            ax.set_yticklabels(ax_y.ticklabels, fontdict=fontdict)
        # x-label
        if ax_x.label is None:
            ax.set_xlabel('Unknown', fontdict=title_fontdict)
        else:
            ax.set_xlabel(ax_x.label, fontdict=title_fontdict)
        if ax_x.ticks is not None and ax_x.ticklabels is not None:
            ax.set_xticks(ax_x.ticks)
            ax.set_xticklabels(ax_x.ticklabels, fontdict =fontdict)
        plt.show()


def TEST_Painter():
    lons = np.arange(0, 360, 10)
    lats = np.arange(-90, 90, 5)

    Painter.drawCoordination(lons, lats)






if __name__ == "__main__":
    TEST_Painter()