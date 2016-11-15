import pandas as pd

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


STOP = 'STOP'
TAXI = 'TAXI'
# TAKEOFF = 'TAKEOFF'
RUNWAY = 'RUNWAY'
AERO = 'AEROTOW'
# FLIGHT = 'FREE_FLIGHT'
# LANDING = 'LANDING'

def convert2flightplan(df: pd.DataFrame):
    """
    Convert dataframe to Flight Gear Flight Plan.
    """

    # Feet above sea level
    df['fasl'] = df['masl'] * 3.28084

    # Knots are nm (1852 m) per hour.
    df['knots'] = (df['dm'] / 1852) / (df['dt_sec'] / 60 / 60)

    rv = HEADER
    prev_state = STOP
    # Take off at Vr (stop ignoring altitude).
    df['state'] = df.apply(lambda x: flight_state(x['knots']), axis=1)

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


def flight_state(kts: float):
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
