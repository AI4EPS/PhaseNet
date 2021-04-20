import sys, glob, os
import numpy as np
import matplotlib.pyplot as plt
from obspy.core import Stream, read as ocread, UTCDateTime
import pandas as pd

for m, pick_data in enumerate(
        [pd.read_csv('/home/lehujeur/Desktop/output/picks.csv', header=0, dtype=str),
         pd.read_csv('./demo/sds/output/picks.csv', header=0, dtype=str)]):
    picks = {}

    seedid = [pick_data.iloc[npick]['seedid'].rstrip('.D') for npick in range(len(pick_data))]
    phasename = [pick_data.iloc[npick]['phasename'] for npick in range(len(pick_data))]
    picktime = [UTCDateTime(pick_data.iloc[npick]['time']) for npick in range(len(pick_data))]
    probability = [float(pick_data.iloc[npick]['probability']) for npick in range(len(pick_data))]

    I = np.lexsort((picktime, phasename, seedid))
    with open({0: "toto", 1: "tata"}[m], 'w') as fid:
        for i in I:
            fid.write(f'{seedid[i]},{phasename[i]},{picktime[i]},{probability[i]}\n')

data = {}
for f in glob.iglob('demo/sds/data/*/*/*/*/*'):
    st = ocread(f, format="MSEED")
    for tr in st:
        seedid = f"{tr.stats.network}.{tr.stats.station}.{tr.stats.location}.{tr.stats.channel}"
        try:
            data[seedid].append(tr)
        except KeyError:
            data[seedid] = Stream([tr])

fig = plt.figure()
ax = fig.add_subplot(111)
gain = 0.1
for n, (seedid, st) in enumerate(data.items()):

    for tr in st:
        tr.detrend("linear")
        tr.data /= np.std(tr.data)
        t = tr.stats.starttime.timestamp + np.arange(tr.stats.npts) * tr.stats.delta
        ax.plot(t, gain * tr.data + n, color="k")

for m, pick_data in enumerate(
        [pd.read_csv('/home/lehujeur/Desktop/output/picks.csv', header=0, dtype=str),
         pd.read_csv('./demo/sds/output/picks.csv', header=0, dtype=str)]):
    picks = {}

    for npick in range(len(pick_data)):
        # print(npick, pick_data.iloc[npick])
        seedid = pick_data.iloc[npick]['seedid'].rstrip('.D')
        phasename = pick_data.iloc[npick]['phasename']
        picktime = UTCDateTime(pick_data.iloc[npick]['time']).timestamp
        probability = float(pick_data.iloc[npick]['probability'])

        for completter in "ZNE":

            try:
                picks[seedid + completter].append((phasename, picktime, probability))
            except KeyError:
                picks[seedid + completter] = [(phasename, picktime, probability)]

    for n, seedid in enumerate(data.keys()):
        try:
            if m == 0:
                color = {"P": "r", "S": "b"}[phasename]
                marker = "x-"
            else:
                color = {"P": "m", "S": "g"}[phasename]
                marker = "+-"

            for (phasename, picktime, probability) in picks[seedid]:
                ax.plot(picktime * np.ones(2), 10. * probability * np.array([-1, 1]) * gain + n, marker, color=color)

        except KeyError:
            pass

plt.show()
