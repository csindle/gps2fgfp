#!/usr/bin/env python3

import datetime
import sys
import typing

import xml.etree.ElementTree as et
import geopy.distance
import pandas as pd

import numpy as np

from pykalman import KalmanFilter

# https://www.hdm-stuttgart.de/~maucher/Python/ComputerVision/html/Tracking.html
# https://pykalman.github.io/#pykalman.KalmanFilter.smooth
# https://balzer82.github.io/Kalman/
# http://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/
# http://www.thealgoengineer.com/2014/online_linear_regression_kalman_filter/

pd.set_option('display.mpl_style', 'default')
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 200)
pd.set_option('precision', 5)
pd.set_option('max_rows', 99)

# To split into flights, use:
#  csplit --quiet  --prefix=flight  --suffix-format %02d.xml   many.xml   '/version/'  "{*}"

HEADER = '''<?xml version="1.0"?>
<!-- J3 for aerotow
Format:
  <name>       Waypoint name.  When a waypoint named END is reached the AI airplane will delete itself.
  <lat>        Latitude. degrees (-90 to 90)
  <lon>        Longitude. degrees (-180 to 180)
  <alt>        Altitude above sea level. feet
  <crossat>    Crossing altitude. feet
  <ktas>       Knots true airspeed.
  <on-ground>  Set true if on the ground.
  <gear-down>  Set true for gear down. false for gear up.
  <flaps-down> Set true for flaps down. false for flaps up.
-->
<PropertyList>
  <flightplan>'''
WPT = '''
    <wpt>
      <name>{state} {n}</name>
      <lat>{lat}</lat>
      <lon>{lon}</lon>
      <alt>{fasl}</alt>
      <ktas>{knots}</ktas>
      <on-ground>{ground}</on-ground>
      <flaps-down>{ground}</flaps-down>
    </wpt>'''
FOOTER = '''
    <wpt>
      <name>END</name>
    </wpt>
  </flightplan>
</PropertyList>
'''


def fgfp(df):
    """
    """
    rv = HEADER
    prev_state = STOP

    for index, row in df.iterrows():
        state = row['state']
        if state in (TAXI, RUNWAY, AERO,):
            # Only output when really moving.

            rv += WPT.format(**row, ground='true' if state in (STOP, TAXI, RUNWAY,) else 'false')

            if state == TAXI and prev_state == STOP:
                # Landed new flight.
                rv += FOOTER + HEADER + "<!-- -lat={lat} -lon={lon} -->".format(**row)
        prev_state = state

    rv += FOOTER
    return rv


STOP = 'STOP'
TAXI = 'TAXI'
# TAKEOFF = 'TAKEOFF'
RUNWAY = 'RUNWAY'
AERO = 'AEROTOW'


# FLIGHT = 'FREE_FLIGHT'
# LANDING = 'LANDING'

def flight_state(kts):
    """Return the state of flight for the point."""
    V_TAXI = 6  # Minumum taxing speed. Probably GPS noise below this speed [knots].
    V_RUNWAY = 12  # Probable takeoff or landing above this.
    V_AERO = 43  # Tug take off speed [knots] V_rotate.

    state = STOP
    if kts > V_AERO:
        state = AERO
    elif kts > V_RUNWAY:
        state = RUNWAY
    elif kts > V_TAXI:
        state = TAXI
    return state


def compute(gpx_filename):
    """
    """
    tree = et.parse(gpx_filename)
    root = tree.getroot()

    trkpts = []
    times = []

    for n, trkpt in enumerate(root[1][4]):
        t = datetime.datetime.strptime(trkpt[1].text, '%Y-%m-%dT%H:%M:%SZ')
        times.append(t)
        trkpts.append((n, float(trkpt.get('lat')), float(trkpt.get('lon')), float(trkpt[0].text) / 1000))

    df = pd.DataFrame(trkpts, columns=['n', 'lat', 'lon', 'kmasl'], index=pd.DatetimeIndex(times), dtype=float)

    df['dt'] = pd.Series(times, index=times).diff()
    df['dkmasl'] = df['kmasl'].diff()
    df['dlat'] = df['lat'].diff()
    df['dlon'] = df['lon'].diff()
    df = df.fillna(value=0.0)

    df = kalman(df=df)

    df['lat2'], df['lon2'], df['kmasl2'], df['dt2'] = df['lat'].shift(-1), df['lon'].shift(-1), df['kmasl'].shift(-1), \
                                                      df['dt'].shift(-1)
    df = df.head(-1)
    df['dt_sec'] = df.apply(lambda x: float(x['dt2'].seconds), axis=1)

    # Feet above sea level
    df['fasl'] = df['kmasl'] * 3280.84

    # Nautical miles between points.
    df['d_km'] = df.apply(lambda x: geopy.distance.vincenty(
        (x['lat'], x['lon'], x['kmasl']),
        (x['lat2'], x['lon2'], x['kmasl2'])).km, axis=1)

    df.drop(['lat2', 'lon2', 'kmasl2', 'dt', 'dt2'], axis=1, inplace=True)

    # Knots are nm (1.852 km) per hour.
    df['knots'] = (df['d_km'] / 1.852) / (df['dt_sec'] / 60 / 60)

    # df = df.rolling(window=11, min_periods=0, center=True,).mean()
    # resample(str(RESAMPLE) + 'S').rolling(window=1, min_periods=0, center=True, win_type='hamming', ).mean()

    # Take of at Vr (stop ignoring altitude).
    df['state'] = df.apply(lambda x: flight_state(x['knots']), axis=1)
    df = df.asfreq("5S", method='ffill')
    return df


def kalman(df: pd.DataFrame) -> pd.DataFrame:
    """
    :return: Kalman smooth these columns: ['kmasl', 'lat', 'lon', 'dkmasl', 'dlat', 'dlon']
    """

    columns = ['kmasl', 'lat', 'lon', 'dkmasl', 'dlat', 'dlon']
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
    kf = KalmanFilter(
        transition_matrices=trans_mat,
        observation_matrices=obs_mat,
        transition_covariance=1.0e-4 * np.eye(6),
        observation_covariance=1.0e-1 * np.eye(6),
        initial_state_mean=[df.kmasl[0], df.lat[0], df.lon[0], df.dkmasl[0], df.dlat[0], df.dlon[0]],
        initial_state_covariance=1.0e-3 * np.eye(6),
    )
    x = df.as_matrix(columns=columns)

    (state_means, state_covs) = kf.em(x, n_iter=6).smooth(x)

    df.kmasl = state_means[:, 0]
    df.lat = state_means[:, 1]
    df.lon = state_means[:, 2]
    df.dkmasl = state_means[:, 3]
    df.dlat = state_means[:, 4]
    df.dlong = state_means[:, 5]

    return df


if __name__ == '__main__':
    gpx_filename = sys.argv[1]
    df = compute(gpx_filename)
    #print(df)
    xml = fgfp(df)
    #print(xml)

    df.to_pickle('df.pickle')

    import matplotlib
    import matplotlib.pyplot as plt

    #matplotlib.rcParams['figure.figsize'] = (12, 12)

    # df.plot(subplots=True, grid=True, use_index=True, layout=(8, 2), )  # figsize=(64, 40), )
    plt.plot(df['lon'], df['lat'])

    # pd.DataFrame(
    #     dict(
    #         kmasl=df.kmasl,
    #         # kmasl_smooth=state_means[:, 0],
    #         lat=df.lat,
    #         # lat_smooth=state_means[:, 1],
    #         lon=df.lon,
    #         # lon_smooth=state_means[:, 2],
    #     ),
    #     index=df.index,
    # ).plot(subplots=True)
    # plt.plot(state_means[:, 1], state_means[:, 2])
    #     df[['fasl', 'knots', ]].plot(subplots=True, grid=True, use_index=True, layout=(1, 2), ) # figsize=(64, 40), )
    #     plt.plot(df['lon'], df['lat'])
    #plt.tight_layout();
    plt.show()
    #plt.figure()
    plt.savefig('df.png')
