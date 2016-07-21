#!/usr/bin/env python3

import datetime
import sys

import xml.etree.ElementTree as et
import geopy.distance
import pandas as pd

pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 200)
pd.set_option('precision', 5)
pd.set_option('max_rows', 20)

"""
todo:
* Hysteresis for take-off and landing
* Segment into separate flights
* Output suggested aircraft's (not AI bot) start lat and lon.
"""

#RESAMPLE = 10  # Seconds per FGFP waypoint.
#V_BORING = 5   # Ignore velocities below this [knots].
Vr = 35        # Tug take off speed [knots].

FGTRUE = 'true'
FGFALSE = 'false'

# To split into flights, use:
#  csplit --quiet  --prefix=flight  --suffix-format %02d.xml   many.xml   '/___NEW_FLIGHT___/'  "{*}"
FD  = "\n<!-- ___NEW_FLIGHT___ -->\n"

HEADER = '''<?xml version="1.0"?>
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
    
WPT = '''
        <wpt>
            <name>{n}</name>
            <lat>{lat}</lat>
            <lon>{lon}</lon>
            <alt>{fasl}</alt>
            <ktas>{knots}</ktas>
            <on-ground>{ground}</on-ground>
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
    old_ground = FGTRUE  # Start on the ground
    rv = HEADER
    
    for index, row in df.iterrows():
        if row['ground'] != old_ground:
            # Insert flight delimiter:
            rv += FOOTER + FD + HEADER
            old_ground = row['ground']
            
        rv += WPT.format(**row)

    rv += FOOTER 
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
        t = datetime.datetime.strptime(trkpt[1].text, '%Y-%m-%dT%H:%M:%SZ')
        times.append(t)
        trkpts.append((n, float(trkpt.get('lat')), float(trkpt.get('lon')), float(trkpt[0].text)/1000))

    df = pd.DataFrame(trkpts, columns=['n', 'lat', 'lon', 'kmasl'], index=pd.DatetimeIndex(times), dtype=float)
    # Feet above sea level
    df['fasl'] = df['kmasl'] * 3.28084

    df['dt'] = pd.Series(times, index=times).diff()
    df['lat2'], df['lon2'], df['kmasl2'], df['dt2'] = df['lat'].shift(-1), df['lon'].shift(-1), df['kmasl'].shift(-1), df['dt'].shift(-1)    
    df = df.head(-1)
   
    df['dt_sec'] = df.apply(lambda x: float(x['dt2'].seconds), axis=1)
    
    # Nautical miles between points.
    df['d_km'] = df.apply(lambda x: geopy.distance.vincenty(
        (x['lat'], x['lon'], x['kmasl']),
        (x['lat2'], x['lon2'], x['kmasl2'])).km, axis=1)

    df.drop(['lat2', 'lon2', 'kmasl2', 'dt', 'dt2'], axis=1, inplace=True)

    # Knots are 1.852 km per hour.
    df['knots'] = df['d_km']/df['dt_sec'] / 1.852 * 3600.0
    
    #df = df.asfreq("S", method='nearest')
    df = df.rolling(window=11, min_periods=0, center=True,).mean()
    # resample(str(RESAMPLE) + 'S').rolling(window=1, min_periods=0, center=True, win_type='hamming', ).mean()    
        
    # Take of at Vr (stop ignoring altitude).
    df['ground'] = df.apply(lambda x: FGFALSE if x['knots'] > Vr else FGTRUE, axis=1)
    #print(df)

    return df


if __name__ == '__main__':
    gpx_filename = sys.argv[1]
    df = compute(gpx_filename)
    xml = fgfp(df)
    print(xml)

    if 0:
        import matplotlib.pyplot as plt
        df[['fasl', 'knots', ]].plot(subplots=True, grid=True, use_index=True, layout=(1, 2), ) # figsize=(64, 40), )
        plt.figure()
        plt.plot(df['lon'], df['lat'])
        plt.show()
        plt.savefig('data.png')

