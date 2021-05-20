#!/usr/bin/env python
import sys, glob, os
import numpy as np

"""
generate a csv file for the sds plugin
call it with (for the demo)

    python sds_to_csv.py ./data "*" "*"

or for a specific year
    python sds_to_csv.py ./data 2000 "*"

or for a specific day
    python sds_to_csv.py ./data 2000 223

results printed to stdout
"""

HEADER = 'network,station,location,channel,dataquality,year,julday,starttime_in_day_sec,endtime_in_day_sec'
LINEFMT = '{network},{station},{location},{channel},{dataquality},{year},{julday},{starttime_in_day_sec},{endtime_in_day_sec}'


if __name__ == '__main__':

    sds  = sys.argv[1]  # sds root directory
    year = sys.argv[2]
    jday = sys.argv[3] # jba APR 2021

    assert os.path.isdir(sds)

    lines = []
    #searchpath = os.path.join(sds, "[0-9][0-9][0-9][0-9]", "*", "*", "??Z.?")
    searchpath = os.path.join(sds, year, "*", "*", "??Z.?")

    for dirname in glob.iglob(searchpath):
        if not os.path.isdir(dirname) and not os.path.islink(dirname):
            continue

        channel, dq = os.path.basename(dirname).split('.')
        #channel = '?H?'
        dirname = os.path.dirname(dirname)

        station = os.path.basename(dirname)
        dirname = os.path.dirname(dirname)

        network = os.path.basename(dirname)
        dirname = os.path.dirname(dirname)

        year = os.path.basename(dirname)

        filesearch = os.path.join(
            sds, year, network, station,
#            f"*.{dq}", f"{network}.{station}.*.{channel}.{dq}.{year}.[0-9][0-9][0-9]")
            f"*.{dq}", f"{network}.{station}.*.{channel}.{dq}.{year}." + jday)
        for filename in glob.iglob(filesearch):
            if not os.path.isfile(filename):
                continue

            location = filename.split('.')[-5]
            julday = filename.split('.')[-1]

            lines.append((network, station, location, channel.replace('Z', "?"),
                dq, year, julday))

    # convert to arrays
    network, station, location, channel, \
        dataquality, year, julday = \
        [np.array(item, str) for item in zip(*lines)]

    i_sort = np.lexsort((channel, station, network, julday, year))

    print(HEADER)
    for i in i_sort:
        print(f'{network[i]},{station[i]},{location[i]},{channel[i]},{dataquality[i]},{year[i]},{julday[i]},0.0,{24*3600.}')



