import matplotlib.pyplot as plt
import matplotlib.colors as col
from mpl_toolkits.basemap import Basemap, shiftgrid, cm
import numpy as np
from typing import List, Tuple

def draw_cloud_top_height_map(
        cloud_top_height,
        lons,
        lats,
        *,
        gray: bool=True,
        output_title :str =None,
        center: Tuple[float, float]=None
    ):

    raw_image = cloud_top_height
    # 将云顶高度缩放到[0-1]便于显示？
    CTO_min = -1                    # the minimum of Cloud Top Height(CTO)
    CTO_max = 12                    # the maximun of CTO
    min_lon = np.min(lons)
    max_lon = np.max(lons)
    min_lat = np.min(lats)
    max_lat = np.max(lats)
    # 开始绘图
    fig = plt.figure()
    if center is None:
        if len(lons.shape) == 2:
            center_lon = lons[int(lons.shape[0] / 2), int(lons.shape[1] / 2)]
            center_lat = lats[int(lats.shape[0] / 2), int(lats.shape[1] / 2)]
        else:
            center_lon = np.mean(lons)
            center_lat = np.mean(lats)
    else:
        center_lon = center[0]
        center_lat = center[1]

    print(center_lon)
    print(center_lat)

    m = Basemap(width=12000000 / 2, height=9000000 / 2, projection='lcc',
                resolution='h', lat_1=45., lat_2=55, lat_0=center_lat, lon_0=center_lon)

    x, y = m(lons, lats)  # Convert [lon, lat] to projection Position.
    # m.plot(x, y, marker='D',color='m')
    min_x = np.min(x)
    max_x = np.max(x)
    min_y = np.min(y)
    max_y = np.max(y)
    # set range.
    x_length = max_x - min_x
    y_length = max_y - min_y
    resolution = [250, 250]
    x_scale = (resolution[0] - 1) / x_length
    y_scale = (resolution[1] - 1) / y_length
    x_pos = (np.floor((x - min_x) * x_scale)).astype(int)
    y_pos = (np.floor((y - min_y) * y_scale)).astype(int)
    image = np.zeros([resolution[0], resolution[1]], dtype=float)
    image[...] = np.nan
    image[x_pos, y_pos] = raw_image
    m.drawcoastlines(color='b')  # Draw Coast Line
    m.drawmeridians(np.arange(-180, 180, 30), labels=[1, 0, 0, 1])  # Draw meridians(经纬脉络)
    m.drawparallels(np.arange(-90, 90, 15), labels=[0, 0, 1, 0])
    x0 = np.min(x)
    x1 = np.max(x)
    y0 = np.min(y)
    y1 = np.max(y)
    extent = (x0, x1, y0, y1)
    image = np.rot90(image, k=1, axes=(0, 1))
    # image = hisEqulColor(image)

    if gray:
        gray_cmap = col.LinearSegmentedColormap.from_list('own2', ['gray', 'black'])
        colormesh = plt.imshow(image, extent=extent, vmin=CTO_min, vmax=CTO_max, cmap=gray_cmap)
    else:
        colormesh = plt.imshow(image, extent=extent, vmin=CTO_min, vmax=CTO_max, cmap='jet')
    m.colorbar(colormesh)

    if output_title is None:
        plt.show()
    else:
        plt.savefig(f'output_title.png', dpi=512)
