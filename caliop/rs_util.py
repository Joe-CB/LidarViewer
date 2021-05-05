import os.path
import os
import numpy as np
from math import radians, cos, sin, asin, sqrt
# from geopy.distance import geodesic
import sqlite3
import io
import numba
from numba import jit
import math


def make_p(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


@jit(nopython=True)
def deg2rad(deg):
  return deg * (math.pi/180)


@jit(nopython=True)
def get_distance_from_lat_lon_in_km(lat1,lon1,lat2,lon2):
    R = 6371          # Radius of the earth in km
    dLat = deg2rad(lat2-lat1)
    dLon = deg2rad(lon2-lon1)
    a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(deg2rad(lat1)) * math.cos(deg2rad(lat2)) * math.sin(dLon/2) * math.sin(dLon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c
    return d

# dis1 = get_distance_from_lat_lon_in_km(60, 120, 60.0002, 120)
# dis2 = get_distance_from_lat_lon_in_km(60, 120, 60, 120.0002)
# dis3 = get_distance_from_lat_lon_in_km(60, 120, 60.0002, 120.0002)

# @jit(nopython=True)
def get_distance(coor_arr1, coor_arr2, distance_arr):
    size = coor_arr1.shape[0]
    for i in range(size):
        distance_arr[i] = get_distance_from_lat_lon_in_km(
            coor_arr1[i,0],
            coor_arr1[i,1],
            coor_arr2[i,0],
            coor_arr2[i,1])

# @jit(nopython=True)
# def get_distance(lats1, lons1, lats2, lons2, distance_arr):
#     size = lats1.shape
#     for i in range(size):
#         distance_arr[i] = get_distance_from_lat_lon_in_km(
#             lats1[i],
#             lons1[i],
#             lats2[i],
#             lons2[i],
#         )

'''Numpy with Sqlite3'''
def adapt_ndarray(ndarray):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, ndarray)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_ndarray(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)
sqlite3.register_adapter(np.ndarray, adapt_ndarray)
sqlite3.register_converter("ndarray", convert_ndarray)

def distance_by_lon_lat(begin, end):
    '''

    :param begin: [longitude, latitude]
    :param end: [longitude, latitude]
    :return: distance
    '''
    return geodistance(begin[0], begin[1], end[0], end[1])

def geodistance(lng1,lat1,lng2,lat2):
    # lng1,lat1,lng2,lat2 = (120.12802999999997,30.28708,115.86572000000001,28.7427)
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)]) # 经纬度转换成弧度
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance=2*asin(sqrt(a))*6371*1000 # 地球平均半径，6371km
    distance=round(distance/1000,3)
    return distance


def get_all_files(dir):
    files_ = []
    list = os.listdir(dir)
    for i in range(0, len(list)):
        path = os.path.join(dir, list[i])
        if os.path.isdir(path):
            files_.extend(get_all_files(path))
        if os.path.isfile(path):
            files_.append(path)
    return files_


# a/b.c-->b.c
def get_simple_filename(filename):

    fp, short_name = os.path.split(filename)
    return short_name


#a/b.c-->a
def get_path(file_path):
    fp, short_name = os.path.split(file_path)
    return fp

# a/b.c-->.c
def get_file_extension(filename):
    fp, short_name = os.path.split(filename)
    stripped, extension = os.path.splitext(short_name)
    return  extension


# a/b.c-->b
def get_stripped_name(filename):
    fp, short_name = os.path.split(filename)
    stripped, extension = os.path.splitext(short_name)
    return stripped


# a/b.c-->a/b
def get_no_extension(filename):
    fp, short_name = os.path.split(filename)
    stripped, extension = os.path.splitext(short_name)
    return fp + str('/') + stripped


def make_dir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


def  interpo2d(src, out_size):
    ''' interpolation 2D'''
    w_scale = (src.shape[0]-1)/out_size[0]
    h_scale = (src.shape[1]-1)/out_size[1]
    [grid_y, grid_x]=np.meshgrid(np.arange(out_size[1]), np.arange(out_size[0]))
    scale_grid_x = grid_x*w_scale
    scale_grid_y = grid_y*h_scale
    min_x = (np.floor(scale_grid_x)).astype(int)
    max_x = (np.ceil(scale_grid_x)).astype(int)
    min_y = (np.floor(scale_grid_y)).astype(int)
    max_y = (np.ceil(scale_grid_y)).astype(int)
    diff_x = scale_grid_x-min_x
    diff_y = scale_grid_y-min_y

    temp = src[min_x, min_y]
    temp2 = src[max_x, min_y]
    min_mid_x = src[min_x, min_y]*(1-diff_x)+src[max_x, min_y]*diff_x
    max_mid_x = src[min_x, max_y]*(1-diff_x)+src[max_x, max_y]*diff_x

    interpo = min_mid_x*(1-diff_y)+max_mid_x*diff_y
    return interpo


# eg:MYD021KM.A2016*.hdf
#   return MYD021KM
def get_file_type(filename):
    fp, short_name = os.path.split(filename)
    return short_name[0:short_name.find('.')]