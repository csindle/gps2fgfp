#!/usr/bin/env python3

import datetime
import sys
import pathlib
import xml.etree.ElementTree as et

import geopy.distance
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pykalman

from flightgear import convert2flightplan


# https://www.hdm-stuttgart.de/~maucher/Python/ComputerVision/html/Tracking.html
# https://pykalman.github.io/#pykalman.KalmanFilter.smooth
# https://balzer82.github.io/Kalman/
# http://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/
# http://www.thealgoengineer.com/2014/online_linear_regression_kalman_filter/

def __init():
    """
    Initialise settings.
    """
    pd.set_option('display.mpl_style', 'default')
    pd.set_option('display.max_columns', 200)
    pd.set_option('display.width', 200)
    pd.set_option('precision', 5)
    pd.set_option('max_rows', 20)


def __read_gpx(gpx_filename: pathlib.Path) -> pd.DataFrame:
    """
    Load data from GPX XML.
    """
    root = et.parse(gpx_filename).getroot()
    trkpts = []
    times = []
    for n, trkpt in enumerate(root[1][4]):
        t = datetime.datetime.strptime(trkpt[1].text, '%Y-%m-%dT%H:%M:%SZ')
        times.append(t)
        trkpts.append((n, float(trkpt.get('lat')), float(trkpt.get('lon')), float(trkpt[0].text)))

    df = pd.DataFrame(trkpts, columns=['n', 'lat', 'lon', 'masl'], index=pd.DatetimeIndex(times), dtype=float)
    #df['dt'] = pd.Series(times, index=times).diff()
    return df


def __diff(data: pd.DataFrame) -> pd.DataFrame:
    """
    Differentiate columns. The first row gets lost.
    """
    data['dmasl'] = data['masl'].diff()
    data['dlat'] = data['lat'].diff()
    data['dlon'] = data['lon'].diff()
    return data.tail(-1)


def __kalman(df: pd.DataFrame) -> pd.DataFrame:
    """
    :return: Kalman smooth these columns: ['masl', 'lat', 'lon', 'dmasl', 'dlat', 'dlon']
    """
    columns = ['masl', 'lat', 'lon', 'dmasl', 'dlat', 'dlon']
    trans_mat = np.array([
        [1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ])
    obs_mat = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ])
    kf = pykalman.KalmanFilter(
        transition_matrices=trans_mat,
        transition_covariance=1.0e-4 * np.eye(6),
        observation_matrices=obs_mat,
        observation_covariance=1.0e-1 * np.eye(6),
        initial_state_mean=[df.masl[0], df.lat[0], df.lon[0], df.dmasl[0], df.dlat[0], df.dlon[0]],
        initial_state_covariance=1.0e-3 * np.eye(6),
    )
    x = df.as_matrix(columns=columns)

    (state_means, state_covs) = kf.em(x, n_iter=6).smooth(x)

    df['k_masl'] = state_means[:, 0]
    df['k_lat'] = state_means[:, 1]
    df['k_lon'] = state_means[:, 2]
    df['k_dmasl'] = state_means[:, 3]
    df['k_dlat'] = state_means[:, 4]
    df['k_dlong'] = state_means[:, 5]
    return df


def main(gpx_fn: pathlib.Path) -> pd.DataFrame:
    """
    """
    __init()
    df = __read_gpx(gpx_fn)
    print('Raw: ', df)

    df = df.resample('15S').mean()   # todo
    print('Resampled: ', df)

    df = df.interpolate(limit=1000)
    print('Interpolated: ', df)

    # Add differential columns:
    df = __diff(data=df)
    print('Differential: ', df)

    #pd.rolling_apply(df, 2, func)  #geopy.distance.vincenty((), ()), min_periods=None, freq=None, center=False, args=(), kwarg
    #todo df.rolling(window=4, center=False).apply(func=func, axis=1)  #, args= < tuple >, kwargs = < dict >, func = < function >)
    #print('Rolling: ', df)

    # Smooth:
    df = __kalman(df=df)
    print('Kalman: ', df)


    # df['lat2'], df['lon2'], df['masl2'], df['dt2'] = \
    #     df['lat'].shift(-1), df['lon'].shift(-1), df['masl'].shift(-1), df['dt'].shift(-1)
    # #df = df.head(-1)
    # df['dt_sec'] = df.apply(lambda x: float(x['dt2'].seconds), axis=1)
    # print(df)
    # # Nautical miles between points.
    # df['dm'] = df.apply(lambda x: geopy.distance.vincenty(
    #     (x['lat'], x['lon'], x['masl']),
    #     (x['lat2'], x['lon2'], x['masl2'])).m, axis=1)
    # df.drop(['lat2', 'lon2', 'masl2', 'dt', 'dt2'], axis=1, inplace=True)

    return df

def func(df, **kwargs):
    print(type(df))
    print(kwargs)
    print(df)
    return 2.4


def visualise(df: pd.DataFrame):
    """
    Plot df.
    """

    plt.figure(1)
    plt.subplot(121)
    plt.plot(
        df['lon'], df['lat'], '-b.',
        df['k_lon'], df['k_lat'], '-r.',
        )

    plt.subplot(122)
    plt.plot(
        df['masl'], '-b.',
        df['k_masl'], '-r.',
    )
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())

    plt.show(block=True)
    return


if __name__ == '__main__':

    df = main(gpx_fn=sys.argv[1])
    #visualise(df)
    if len(sys.argv) > 2:
        with open(sys.argv[2] or 'output.xml', 'wb') as f:
            f.write(convert2flightplan(df))
    pass


####################################################################

# matplotlib.rcParams['figure.figsize'] = (16, 9)
# # df.plot(subplots=True, grid=True, use_index=True, layout=(4, 2), )  # figsize=(64, 40), )
# plt.plot(df['lon'], df['lat'])
# plt.plot(df['lon'], df['lat'])
# # plt.subplots()
# pd.DataFrame(
#     dict(
#         masl=df.masl,
#         # masl_smooth=state_means[:, 0],
#         lat=df.lat,
#         # lat_smooth=state_means[:, 1],
#         lon=df.lon,
#         # lon_smooth=state_means[:, 2],
#     ),
#     index=df.index,
# ).plot(subplots=True)
# # plt.plot(state_means[:, 1], state_means[:, 2])
# #     df[['fasl', 'knots', ]].plot(subplots=True, grid=True, use_index=True, layout=(1, 2), ) # figsize=(64, 40), )
# #     plt.plot(df['lon'], df['lat'])
# # plt.tight_layout()
# plt.figure()
# plt.savefig('df.png')

#xml = convert2flightplan(df)
#print(xml)
#df.to_pickle('df.pickle')
