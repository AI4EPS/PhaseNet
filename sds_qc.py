import sys, glob, os
import numpy as np
import matplotlib.pyplot as plt
from obspy.core import Stream, read as ocread, UTCDateTime
import pandas as pd
import h5py
from obspy.core import Trace
from sds_plugin import show_sds_prediction_results, DataReaderSDS


class Args(object):
    # generate a fake argument object for testing
    # reproduce the defaults options after run.py
    mode = "pred"
    epochs = 100
    batch_size = 20
    learning_rate = 0.01
    decay_step = -1
    decay_rate = 0.9
    momentum = 0.9
    filters_root = 8
    depth = 5
    kernel_size = [7, 1]
    pool_size = [4, 1]
    drop_rate = 0
    dilation_rate = [1, 1]
    loss_type = "cross_entropy"
    weight_decay = 0
    optimizer = 'adam'
    summary = True
    class_weights = [1, 1, 1]
    log_dir = None
    model_dir = os.path.join("model", "190703-214543")
    num_plots = 10
    tp_prob = 0.3
    ts_prob = 0.3
    input_length = None
    input_mseed = False
    input_sds = True
    data_dir = os.path.join("demo", "sds", "data")
    data_list = os.path.join("demo", "sds", "fname_sds.csv")
    train_dir = None
    valid_dir = None
    valid_list = None
    output_dir = os.path.join("demo", "sds", "output")
    plot_figure = False
    save_result = True
    fpred = "picks"


args = Args()
# assert os.path.isdir(args.model_dir)
# assert os.path.isdir(args.data_dir)
# assert os.path.isfile(args.data_list)
# assert not os.path.isdir(args.output_dir), f"output dir exists already {args.output_dir}"

# logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
# coord = tf.train.Coordinator()

data_reader = DataReaderSDS(
    data_dir=args.data_dir,
    data_list=args.data_list,
    queue_size=args.batch_size * 10,
    coord=None,
    input_length=args.input_length)


fig = plt.figure(figsize=(12, 4))
show_sds_prediction_results(fig=fig, data_reader=data_reader, log_dir=args.output_dir)
plt.show()


exit()
show_sds_results(data_list='./demo/sds/fname_sds.csv')


# exit()
stz = ocread('./demo/sds/data/2000/XX/AAAA/EHZ.D/XX.AAAA.00.EHZ.D.2000.223')
stn = ocread('./demo/sds/data/2000/XX/AAAA/EHZ.D/XX.AAAA.00.EHZ.D.2000.223')
ste = ocread('./demo/sds/data/2000/XX/AAAA/EHZ.D/XX.AAAA.00.EHZ.D.2000.223')

for n, st in enumerate([stz, stn, ste]):
    for tr in st:
        tr.detrend()
        tr.data /= tr.data.std()  ## kill amp

        t = tr.stats.starttime.timestamp + np.arange(tr.stats.npts) * tr.stats.delta
        plt.plot(t, 0.05 * tr.data + n, 'k', alpha=0.4)

with h5py.File('./demo/sds/output/results/sample_results.hdf5', 'r') as fid:
    for phasename in "PS":
        group = fid[f'2000/XX/AAAA/EH{phasename}.D/223']

        for sample_name in group.keys():
            print(sample_name)
            print(group[sample_name][:])
            print(group[sample_name].attrs)
            tr = Trace(
                header=dict(**group[sample_name].attrs),
                data=group[sample_name][:])
            t = tr.stats.starttime.timestamp + np.arange(tr.stats.npts) * tr.stats.delta
            plt.plot(t, tr.data / 256. + 3, color={'P': 'r', 'S': 'b'}[phasename])

with h5py.File('/home/lehujeur/Desktop/diverged/PhaseNet/demo/output/results/sample_results.hdf5', 'r') as fid:
    for phasename in "PS":
        group = fid[f'2000/XX/AAAA/EH{phasename}.D/223']

        for sample_name in group.keys():
            print(sample_name)
            print(group[sample_name][:])
            print(group[sample_name].attrs)
            tr = Trace(
                header=dict(**group[sample_name].attrs),
                data=group[sample_name][:])
            t = tr.stats.starttime.timestamp + np.arange(tr.stats.npts) * tr.stats.delta
            plt.plot(t, tr.data / 256. + 3, color={'P': 'm', 'S': 'g'}[phasename], linestyle='--')

plt.show()


for m, pick_data in enumerate(
        [pd.read_csv('/home/lehujeur/Desktop/diverged/PhaseNet/demo/output/picks.csv', header=0, dtype=str),
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
        [pd.read_csv('/home/lehujeur/Desktop/diverged/PhaseNet/demo/output/picks.csv', header=0, dtype=str),
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
                marker = "^-"
                y = 10. * probability * np.array([-1, 0])
            else:
                color = {"P": "m", "S": "g"}[phasename]
                marker = "v-"
                y = 10. * probability * np.array([0., 1.])

            for (phasename, picktime, probability) in picks[seedid]:
                ax.plot(picktime * np.ones(2), y * gain + n, marker, color=color)

        except KeyError:
            pass

plt.show()
