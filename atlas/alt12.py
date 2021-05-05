import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import icesat2_package.utils as is2_utils


class GroundTrack:
    '''
    Handle profile data of ALT12
    '''
    def __init__(self, profile_hdf_obj):
        self.file = profile_hdf_obj
        self._lats = None
        self._lons = None
        self._sea_surface_height = None

    @property
    def latitude(self):
        if self._lats is None:
             self._lats = self.file['ssh_segments/latitude']
        return self._lats

    @property
    def longitude(self):
        if self._lons is None:
             self._lons = self.file['ssh_segments/longitude']
        return self._lons

    @property
    def mean_surface_height(self):
        if self._sea_surface_height is None:
            self._sea_surface_height = self.file['ssh_segments/heights/h']
        return self._sea_surface_height


class ATL12:
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

    def ground_track_left(self,idx):
        return GroundTrack(self.file[f'gt{idx}l'])

    def ground_track_right(self,idx):
        return GroundTrack(self.file[f'gt{idx}r'])


def TEST(atl12_fn):
    atl12_file = ATL12(atl12_fn)
    ground_track_l = atl12_file.ground_track_left(1)
    ground_track_r = atl12_file.ground_track_right(1)
    height_l = ground_track_l.mean_surface_height[:]
    height_r = ground_track_r.mean_surface_height[:]

    is2_utils.Painter.drawCoordination(ground_track_l.longitude[:], ground_track_l.latitude[:])
    is2_utils.Painter.drawCoordination(ground_track_r.longitude[:], ground_track_r.latitude[:])

    plt.figure()
    plt.plot(height_l, '-r')
    plt.show()

    plt.figure()
    plt.plot(height_r, '--g')
    plt.show()


if __name__ == "__main__":
    atl12_fn = 'L:/icesat/ATL12_201910/ATL12_20191029040948_04930501_003_01.h5'
    TEST(atl12_fn)