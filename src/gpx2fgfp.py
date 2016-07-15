#!/usr/bin/env python3

import datetime
import xml.etree.ElementTree as et
import pandas as pd
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 200)
pd.set_option('precision', 2)

import geopy.distance


RESAMPLE = 10  # Seconds per FGAI waypoint.
V_BORING = 10  # Ignore velocities below this [knots].
Vr = 40        # Tugs Take off speed [knots].


def fgai(df, ignore_slow=10):
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

    for index, row in df.iterrows():
        if row['valid']:
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


def main(gpx_filename):
    """
    :param gpx_filename:
    :return:
    """
    tree = et.parse(gpx_filename)
    root = tree.getroot()

    trkpts = []
    times = []

    for n, trkpt in enumerate(root[1][4]):
        trkpts.append((n, trkpt.get('lat'), trkpt.get('lon'), trkpt[0].text))
        times.append(datetime.datetime.strptime(trkpt[1].text, '%Y-%m-%dT%H:%M:%SZ'))

    df = pd.DataFrame(trkpts, columns=['n', 'lat', 'lon', 'alt'], index=pd.DatetimeIndex(times), dtype=float)
    df = df.asfreq("S", method='nearest')

    df = df.resample(str(RESAMPLE) + 'S').mean()

    df['lat2'], df['lon2'], df['alt2'] = df['lat'].shift(-1), df['lon'].shift(-1), df['alt'].shift(-1)
    df = df.head(-1)

    # Nautical miles between points.q

    df['nm'] = df.apply(lambda x: geopy.distance.vincenty(
        (x['lat'], x['lon'], x['alt']),
        (x['lat2'], x['lon2'], x['alt'])).nm, axis=1)

    # Knots are NM per hour.
    df['knots'] = df['nm'] * 3600 / RESAMPLE  # FYI: 1 nm = 1852 m

    # Take of at Vr (stop ignoring altitude).
    df['ground'] = df.apply(lambda x: 'false' if x['knots'] > Vr else 'true', axis=1)

    # Ignore noisy GPS data.
    df['valid'] = df.apply(lambda x: 'true' if x['knots'] > V_BORING else 'false', axis=1)

    df['feet'] = df['alt'] * 3.28084 + 0   # todo Offset

    #print(df.head(20))
    #print(df.tail(20))
    return fgai(df)


if __name__ == '__main__':
    xml = main('flight17.gpx')
    print(xml)
