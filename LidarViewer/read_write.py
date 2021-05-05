import os
import sys
import numpy as np
import atlas.alt04 as atl04
import atlas.alt09 as atl09
import atlas.utils as atlas_utils
import caliop.caliop_l1 as caliop_l1


class Reader:
    def __init__(self,):
        pass

class AtlasReader(Reader):
    def __init__(self):
        pass

    @staticmethod
    def readNRB(filename):
        if filename.find('ATL04') != -1:
            file = atl04.ATL04(filename)
            all_profile = []
            for i in range(1, 4, 1):
                profile = file.getProfile(idx=i)
                _nrb_profile, _nrb_top_bin, _nrb_bot_bin = profile.nrb
                _nrb_profile = atlas_utils.extract_valid_profile(_nrb_profile[:], _nrb_top_bin[:], _nrb_bot_bin[:])
                _nrb_profile = np.log10(_nrb_profile)
                _nrb_profile[_nrb_profile > 33] = np.nan
                all_profile.extend(_nrb_profile)
            return np.array(all_profile)
        return None

    @staticmethod
    def readCAB(filename):
        # ATL09?
        all_profile = []
        file = atl09.ATL09(filename)
        for i in range(1, 4, 1):
            profile = file.getProfile(idx=i)
            cab = profile.getHighRate(name='cab_prof')
            cab = np.log10(cab)
            cab[cab > 33] = np.nan
            all_profile.extend(cab)
        return np.array(all_profile)

        return None

class CaliopReader(Reader):
    def __init__(self):
        pass

    @staticmethod
    def read_TAB_532(filename: str):
        total_attenual_backscatter = caliop_l1.CaliopL1.getTAB(filename)
        total_attenual_backscatter = np.log10(total_attenual_backscatter)
        # cab[cab > 33] = np.nan
        return total_attenual_backscatter
