#!/usr/bin/env python3

import datetime
import sys

import xml.etree.ElementTree as et
import geopy.distance
import pandas as pd

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
        if state in (TAXI, RUNWAY, AERO, ):
            # Only output when really moving.
            
            rv += WPT.format(**row, ground='true' if state in (STOP, TAXI, RUNWAY, ) else 'false')
            
            if state == TAXI and prev_state == STOP:
                # Landed new flight.
                rv += FOOTER + HEADER + "<!-- -lat={lat} -lon={lon} -->".format(**row)
        prev_state = state

    rv += FOOTER 
    return rv

STOP = 'STOP'
TAXI = 'TAXI'
#TAKEOFF = 'TAKEOFF'
RUNWAY = 'RUNWAY'
AERO = 'AEROTOW'
#FLIGHT = 'FREE_FLIGHT'
#LANDING = 'LANDING'

def flight_state(kts):
    """Return the state of flight for the point."""
    V_TAXI = 6     # Minumum taxing speed. Probably GPS noise below this speed [knots].
    V_RUNWAY = 12  # Probable takeoff or landing above this.
    V_AERO = 43    # Tug take off speed [knots] V_rotate.
    
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
        trkpts.append((n, float(trkpt.get('lat')), float(trkpt.get('lon')), float(trkpt[0].text)/1000))

    df = pd.DataFrame(trkpts, columns=['n', 'lat', 'lon', 'kmasl'], index=pd.DatetimeIndex(times), dtype=float)
    # Feet above sea level
    df['fasl'] = df['kmasl'] * 3280.84
    df['dt'] = pd.Series(times, index=times).diff()
    df['lat2'], df['lon2'], df['kmasl2'], df['dt2'] = df['lat'].shift(-1), df['lon'].shift(-1), df['kmasl'].shift(-1), df['dt'].shift(-1)    
    df = df.head(-1)
   
    df['dt_sec'] = df.apply(lambda x: float(x['dt2'].seconds), axis=1)
    
    # Nautical miles between points.
    df['d_km'] = df.apply(lambda x: geopy.distance.vincenty(
        (x['lat'], x['lon'], x['kmasl']),
        (x['lat2'], x['lon2'], x['kmasl2'])).km, axis=1)

    df.drop(['lat2', 'lon2', 'kmasl2', 'dt', 'dt2'], axis=1, inplace=True)

    # Knots are nm (1.852 km) per hour.
    df['knots'] = (df['d_km']/ 1.852) / (df['dt_sec'] / 60 / 60)
    
    #df = df.rolling(window=11, min_periods=0, center=True,).mean()
    # resample(str(RESAMPLE) + 'S').rolling(window=1, min_periods=0, center=True, win_type='hamming', ).mean()    
    
    # Take of at Vr (stop ignoring altitude).
    df['state'] = df.apply(lambda x: flight_state(x['knots']), axis=1)
    df = df.asfreq("5S", method='ffill')
    return df

    
if __name__ == '__main__':
    gpx_filename = sys.argv[1]
    df = compute(gpx_filename)
    xml = fgfp(df)

    if 1:
        print(xml)
    else:
        print(df)
        
        import matplotlib.pyplot as plt
        df[['fasl', 'knots', ]].plot(subplots=True, grid=True, use_index=True, layout=(1, 2), ) # figsize=(64, 40), )
        plt.figure()
        plt.plot(df['lon'], df['lat'])
        plt.show()
        plt.savefig('data.png')

