import h5py
import numpy as np
import os
import matplotlib.pyplot as plt


def generate_altitude():
    alt = []
    alt.extend(-0.25 + 0.03 * np.arange(0, 467, 1) + 0.015)
    alt = np.array(alt)
    alt = np.around(alt, decimals=2)
    alt = alt[::-1]
    return alt


class Profile:
    '''
    Handle profile data of ALT04
    '''
    def __init__(self, profile_hdf_obj):
        self.file = profile_hdf_obj
        self._lats = None
        self._lons = None
        self._nrb_profile = None
        self._nrb_bot_bin = None
        self._nrb_top_bin = None
        self._surf_type = None
        self._cloud_fold_flag = None
        self._solar_elevation = None

    @property
    def lats(self):
        if self._lats is None:
             self._lats = self.file['latitude']
        return self._lats

    @property
    def lons(self):
        if self._lons is None:
             self._lons = self.file['longitude']
        return self._lons

    @property
    def nrb(self):
        if self._nrb_profile is None:
            self._nrb_profile = self.file['nrb_profile']
        if self._nrb_bot_bin is None:
            self._nrb_bot_bin = self.file['nrb_bot_bin']
        if self._nrb_top_bin is None:
            self._nrb_top_bin = self.file['nrb_top_bin']
        return self._nrb_profile, self._nrb_top_bin, self._nrb_bot_bin

    @property
    def surf_type(self):
        if self._surf_type is None:
            self._surf_type = self.file['surf_type']
        return self._surf_type

    @property
    def solar_elevation(self):
        if self._solar_elevation is None:
            self._solar_elevation = self.file['solar_elevation']
        return self._solar_elevation

    def getCloudFoldFlag(self):
        if self._cloud_fold_flag is None:
            self._cloud_fold_flag = self.file['cloud_fold_flag']
        return self._cloud_fold_flag



class ATL04:
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
        profile_dir = f'profile_{int(idx)}'
        return Profile(self.file[profile_dir])



if __name__ == "__main__":
    atl04 = ATL04('K:/icesat/ATL04_201910/ATL04_20191008130832_01780501_003_01.h5')
    profile0 = atl04.getProfile(idx=1)
    lons = profile0.lons[:]
    lats = profile0.lats[:]
    nrb, nrb_top, nrb_bot = profile0.nrb
    nrb, nrb_top, nrb_bot = nrb[:], nrb_top[:], nrb_bot[:]

    x, y = np.meshgrid(np.arange(0,nrb.shape[0]), np.arange(0, nrb.shape[1]), indexing='ij')
    bool_index = np.logical_and(y.T >= nrb_top, y.T < nrb_bot).T
    bool_index = np.logical_not(bool_index)
    nrb[bool_index] = np.nan
    index = np.logical_and(lons > 120, lons<125)
    nrb = nrb[index]
    nrb = np.log10(nrb)
    nrb = np.rot90(nrb, k=-1)
    f = plt.figure()
    plt.imshow(nrb)
    plt.colorbar()
    plt.show()