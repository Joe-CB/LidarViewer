'''
Have some gemeralized function about modis data.
'''
import os


def get_modis_time(filename):
    '''
    Get all Servery time of all MODIS file.
    :param filename: MODIS filename
    :return: Servey time
    '''
    fp, tmpfilename = os.path.split(filename)
    shortname, extension = os.path.splitext(tmpfilename)
    str_list = shortname.split('.')
    if len(str_list) > 3:
        return str_list[1] + str_list[2]


