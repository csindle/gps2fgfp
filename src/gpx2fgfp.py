#!/usr/bin/env python3

import datetime
import sys

import xml.etree.ElementTree as et
import geopy.distance
import pandas as pd
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 200)
pd.set_option('precision', 3)
pd.set_option('max_rows', 20)

"""
todo:
* Hysteresis for take-off and landing
* Segment into separate flights
* Output suggested aircraft's (not AI bot) start lat and lon.

Nice to haves:
* Better de-noising

"""

RESAMPLE = 10  # Seconds per FGFP waypoint.
V_BORING = 13   # Ignore velocities below this [knots].
Vr = 35        # Tug take off speed [knots].


def fgfp(df, ignore_slow=10):
    """
    :param df:
    :param ignore_slow:
    :return:
    """
    rv = '''<?xml version="1.0"?>
<!-- J3 for aerotow
Format:
    <name>       Waypoint name.  When a waypoint named END is reached the AI airplane will delete itself.
    <lat>        latitude. degrees (-90 to 90)
    <lon>        Longitude. degrees (-180 to 180)
    <alt>        altitude above sea level. feet
    <crossat>    Crossing altitude. feet
    <ktas>       Knots true airspeed
    <on-ground>  set true if on the ground
    <gear-down>  set true for gear down. false for gear up
    <flaps-down> set true for flaps down. false for flaps up
-->
<PropertyList>
    <flightplan>'''

    #ACCEL_THRESH = 15  # Knots
    #DECEL_THRESH = 4   # Knots
    #PADDING = 60       #  Seconds.

    """
    Include samples from PADDING seconds before ACCEL_THRESH is reached
    and up until PADDING seconds after DECEL_THRESH is reached.
    Alternatively, a Kalman filter.

    states: stop, taxi, take-off-roll, take-off, flight-towing, flight-free, landing-roll, taxi, stop.

    """

    for index, row in df.iterrows():
        if row['valid'] == 'true':
            rv += """
        <wpt>
            <name>{n}</name>
            <lat>{lat}</lat>
            <lon>{lon}</lon>
            <alt>{feet}</alt>
            <ktas>{knots}</ktas>
            <on-ground>{ground}</on-ground>
        </wpt>""".format(**row)

    rv += '''
        <wpt>
            <name>END</name>
        </wpt>
    </flightplan>
</PropertyList>
'''
    return rv


def compute(gpx_filename):
    """
    :param gpx_filename:
    :return:
    """
    tree = et.parse(gpx_filename)
    root = tree.getroot()

    trkpts = []
    times = []

    for n, trkpt in enumerate(root[1][4]):
        stamp = datetime.datetime.strptime(trkpt[1].text, '%Y-%m-%dT%H:%M:%SZ')
        trkpts.append((n, trkpt.get('lat'), trkpt.get('lon'), trkpt[0].text))
        times.append(stamp)
        
    dt = pd.Series(times, index=times).diff()
    #print('dt\n', dt)

    df = pd.DataFrame(trkpts, columns=['n', 'lat', 'lon', 'alt'], index=pd.DatetimeIndex(times), dtype=float)

    

    #df = df.resample(str(RESAMPLE) + 'S').max()  #  rolling(window=6, min_periods=0, center=True, win_type='hamming', ).mean()
    #print('downsampled\n', df)

    df['lat2'], df['lon2'], df['alt2'] = df['lat'].shift(-1), df['lon'].shift(-1), df['alt'].shift(-1)
    df['dt'] = dt
    df = df.tail(-1)
    
    df['dt_sec'] = df.apply(lambda x: float(x['dt'].seconds), axis=1)
    df = df.head(-1)
    
    # Nautical miles between points.
    df['nm'] = df.apply(lambda x: geopy.distance.vincenty(
        (x['lat'], x['lon'], x['alt']),
        (x['lat2'], x['lon2'], x['alt2'])).nm, axis=1)

    # Knots are NM per hour.
    df['knots'] = df['nm']/df['dt_sec'] * 3600 #!!! / RESAMPLE  # FYI: 1 nm = 1852 m
    
    df = df.asfreq("S", method='nearest')
    #print('per second:\n', df)
    
    df = df.resample(str(RESAMPLE) + 'S').rolling(window=10, min_periods=0, center=True, win_type='hamming', ).mean()
    #print('rolling', df)
    
    
    # Take of at Vr (stop ignoring altitude).
    df['ground'] = df.apply(lambda x: 'false' if x['knots'] > Vr else 'true', axis=1)

    
    # Ignore noisy GPS data.
    df['valid'] = df.apply(lambda x: 'true' if x['knots'] > V_BORING else 'false', axis=1)

    df['feet'] = df['alt'] * 3.28084 + 0   # todo Offset

    return df


if __name__ == '__main__':
    gpx_filename = sys.argv[1]
    df = compute(gpx_filename)
    xml = fgfp(df)
    print(xml)

    #import matplotlib.pyplot as plt
    #df[['alt', 'knots', 'lat', 'lon', ]].plot(subplots=True, grid=True, use_index=True, layout=(2, 2), ) # figsize=(64, 40), )
    #plt.figure()
    #plt.scatter(df['lon'], df['lat'])
    #plt.show()
    #plt.savefig('data.png')

