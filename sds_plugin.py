#!/usr/bin/env python
from __future__ import division
import glob, os, time, logging
import numpy as np
import pandas as pd
import h5py
import tensorflow as tf
import multiprocessing
from tqdm import tqdm
from functools import partial
from obspy.core import UTCDateTime, read as ocread

from data_reader import DataReader, Config
from run import set_config
from model import Model
from detect_peaks import detect_peaks

logger = logging.getLogger(__name__)
tf.compat.v1.disable_eager_execution()
pd.options.mode.chained_assignment = None

"""
The aim of this plugin is to call Phasenet in prediction mode on a large SDS data structure
For giant datasets, the packing of 3 components waveforms into one single file can be difficult to implement
In this plugin, we propose to access the 3-component waveforms directly as they are stored in a SDS (seiscomp like)
data structure.


"""

# ============ global variables
# conventional (formatable) path name for SDS data archive
SDSPATH = os.path.join(
    "{data_dir}", "{year}",
    "{network}", "{station}",
    "{channel}.{dataquality}",
    "{network}.{station}.{location}.{channel}.{dataquality}"
    ".{year:04d}.{julday:03d}")

# first part of the sample name
SEEDID = "{network:s}.{station:s}.{location:s}.{channel2:2s}.{dataquality:1s}"

# location of the sample results inside the hdf5 archive
HDF5PATH = "{year:04d}/{network:s}/{station:s}/{channel2:2s}{phasename:1s}.{dataquality:1s}/{julday}"

# use a more detailed name for each sample to preserve the time information
SAMPLENAME = \
    "{seedid:s}_{year:04d}-{julday:03d}-{hour:02d}-{minute:02d}-{second:09.6f}_" \
    "{sampling_rate:f}Hz_NPTS{input_length}"


# ============ utils
def _decode_sample_name(sample_name: str):
    """get meta data from batch name formatted as defined by global variable SAMPLENAME"""
    seedid: str
    sample_start: UTCDateTime
    sampling_rate: float
    sample_npts: int
    seedid_details: tuple

    seedid, sample_start_s, sampling_rate_s, sample_npts_s = sample_name.split("_")
    try:
        year, julday, hour, minute, second = sample_start_s.split('-')
        second, microsecond = second.split('.')

        sample_start = UTCDateTime(
               year=int(year),
             julday=int(julday),
               hour=int(hour),
             minute=int(minute),
             second=int(second),
        microsecond=int(microsecond))
    except Exception as e:
        # for _ in range(1000):
        #     print(year, julday, hour, minute, second, microsecond, str(e))
        raise ValueError(f'could not decode sample name {sample_name}')

    sampling_rate = float(sampling_rate_s.split('Hz')[0])

    sample_npts = int(sample_npts_s.split('NPTS')[-1])
    network, station, location, channel2, dataquality = seedid.split('.')
    seedid_details = (network, station, location, channel2, dataquality)
    return seedid, sample_start, sampling_rate, sample_npts, seedid_details


def _save_predictions_to_hdf5_archive(hdf5_pointer, fname_batch, pred_batch):
    """
    :param hdf5_pointer:
    :param fname_batch:
    :param pred_batch:
    """

    for i in range(len(fname_batch)):
        seedid, sample_start, sampling_rate, sample_npts, \
        (network, station, location, channel2, dataquality) = \
            _decode_sample_name(sample_name=fname_batch[i].decode())

        sample_mid = sample_start + 0.5 * (sample_npts - 1) / sampling_rate
        sample_end = sample_start + 1.0 * (sample_npts - 1) / sampling_rate
        year = sample_mid.year
        julday = sample_mid.julday
        if f"{year}.{julday}" != f"{sample_start.year}.{sample_start.julday}" or \
            f"{year}.{julday}" != f"{sample_end.year}.{sample_end.julday}":
            # the sample probably overlaps two files because of padding, skip it
            continue

        for nphase, phasename in enumerate("PS"):
            groupname = HDF5PATH.format(
                year=year, julday=julday,
                network=network, station=station,
                channel2=channel2, phasename=phasename,
                dataquality=dataquality)

            # AVOID THE EDGES OF THE SAMPLE (because of the 50% overlap)
            n = sample_npts // 4

            try:
                grp = hdf5_pointer[groupname]
            except KeyError:
                grp = hdf5_pointer.create_group(groupname)

            sample_dataset = grp.create_dataset(
                fname_batch[i].decode(),
                data=255 * pred_batch[i, n:-n, 0, nphase + 1],  # 1 for P and 2 for S
                dtype=np.dtype('uint8'))  # to save disk space, proba scaled by 255

            sample_dataset.attrs["network"] = network
            sample_dataset.attrs["station"] = station
            sample_dataset.attrs["location"] = location
            sample_dataset.attrs["channel"] = channel2 + phasename
            sample_dataset.attrs["dataquality"] = dataquality
            sample_dataset.attrs["starttime"] = str(sample_start + n / sampling_rate)
            sample_dataset.attrs["sampling_rate"] = sampling_rate


def _detect_peaks_thread_sds(i, pred, fname=None, result_dir=None, args=None):
    """
    customized for sds
    """
    input_length = pred.shape[1]
    nedge = input_length // 4  # do not pick maxima in the 25% edge zone each side to avoid repeated picks

    if args is None:
        itp = detect_peaks(pred[i, nedge:-nedge, 0, 1], mph=0.5, mpd=0.5 / Config().dt, show=False)
        its = detect_peaks(pred[i, nedge:-nedge, 0, 2], mph=0.5, mpd=0.5 / Config().dt, show=False)
    else:
        itp = detect_peaks(pred[i, nedge:-nedge, 0, 1], mph=args.tp_prob, mpd=0.5 / Config().dt, show=False)
        its = detect_peaks(pred[i, nedge:-nedge, 0, 2], mph=args.ts_prob, mpd=0.5 / Config().dt, show=False)

    itp = [_ + nedge for _ in itp]
    its = [_ + nedge for _ in its]

    prob_p = pred[i, itp, 0, 1]
    prob_s = pred[i, its, 0, 2]
    if (fname is not None) and (result_dir is not None):
        npzout = os.path.join(result_dir, fname[i].decode())
        pathout = os.path.dirname(npzout)

        os.makedirs(pathout, exist_ok=True)

        np.savez(npzout,
                 pred=pred[i],
                 itp=itp,
                 its=its,
                 prob_p=prob_p,
                 prob_s=prob_s)

    return [(itp, prob_p), (its, prob_s)]


def _postprocessing_thread_sds(i, pred, X, Y=None, itp=None, its=None, fname=None, result_dir=None, figure_dir=None,
                          args=None):
    """
    :param i: batch number
    :param pred: probability functions for P and S phases and?
    :param X: seismic data
    :param Y:
    :param itp: indexs of picked P phases
    :param its: indexs of picked S phases
    :param fname:
    :param result_dir:
    :param figure_dir:
    """
    (itp_pred, prob_p), (its_pred, prob_s) = \
        _detect_peaks_thread_sds(i, pred, fname, result_dir, args)

    # if (fname is not None) and (figure_dir is not None):
    #     plot_result_thread(i, pred, X, Y, itp, its, itp_pred, its_pred, fname, figure_dir)
    return [(itp_pred, prob_p), (its_pred, prob_s)]


# ============ data_reader
class DataReaderSDS(DataReader):
    """
    a DataReader object dedicated to SDS data structure
    """

    def __init__(self, data_dir, data_list, queue_size, coord, input_length=3000, config=Config()):

        # the default object do not read in str mode => force it to preserve the location code
        tmp_list = pd.read_csv(data_list, header=0, dtype=str)

        # use inheritance to initiate self
        DataReader.__init__(
            self,
            data_dir=data_dir, data_list=data_list, mask_window=0,
            queue_size=queue_size, coord=coord, config=config)

        self.data_list = tmp_list  # force pre-readed data on top of what DataReader read
        self.mask_window = None  # not used by this class
        self.input_length = config.X_shape[0]

        if input_length is not None:
            logger.warning("Using input length: {}".format(input_length))
            self.X_shape[0] = input_length
            self.Y_shape[0] = input_length
            self.input_length = input_length

        # check the input directory
        assert os.path.isdir(self.data_dir)

        # check the input file
        try:
            self.data_list.iloc[0]['network']
            self.data_list.iloc[0]['station']
            self.data_list.iloc[0]['location']
            self.data_list.iloc[0]['dataquality']
            self.data_list.iloc[0]['channel']
            self.data_list.iloc[0]['year']
            self.data_list.iloc[0]['julday']
            self.data_list.iloc[0]['starttime_in_day_sec']
            self.data_list.iloc[0]['endtime_in_day_sec']
        except KeyError as e:
            e.args = ('unexpected csv header : I need the following keys on first line (no space)'
                      'network,station,location,dataquality,channel,year,julday,starttime_in_day_sec,endtime_in_day_sec', )
            raise e

    def add_placeholder(self):
        self.sample_placeholder = tf.compat.v1.placeholder(dtype=tf.float32, shape=None)
        self.fname_placeholder = tf.compat.v1.placeholder(dtype=tf.string, shape=None)
        self.queue = tf.queue.PaddingFIFOQueue(self.queue_size,
                                         ['float32', 'string'],
                                         shapes=[self.config.X_shape, []])

        self.enqueue = self.queue.enqueue([self.sample_placeholder,
                                           self.fname_placeholder])

    def dequeue(self, num_elements):
        output = self.queue.dequeue_up_to(num_elements)
        return output

    def find_filenames(self, network: str, station: str, location: str, channel: str, dataquality: str,
                       year: int, julday: int) -> list:
        """
        find a file name in a SDS data structure
        return 3 file names in a list [east_component_filename, north_component_filename, vertical_component_filename]
        """

        if location in ["*", "??", ""]:
            # accepted locations :
            pass

        elif len(location) != 2:
            raise ValueError(
                f"location code must be *, ??, empty "
                f"or a two digit number. got: {location} type: {type(location)}")

        if not (len(channel) == 3 and channel.endswith('?') or channel.endswith('*')):
            raise ValueError(f"unexpected channel {channel}, "
                             "use something like HH? or EH?")

        filenames = []
        for comp in "ENZ":

            # generate a filename using component letter comp
            filepath = SDSPATH.format(
                data_dir=self.data_dir, year=year, julday=julday,
                dataquality=dataquality,
                network=network, station=station,
                location=location, channel=channel[:2] + comp)

            if os.path.isfile(filepath):
                # file exists
                filenames.append(filepath)

            else:
                # file doesn't exist, maybe this is a path, including wildcards, ...
                ls = glob.iglob(filepath)
                filename = next(ls, None)  # None if filepath correspond to no file
                more = next(ls, None)  # None if filepath correspond to exactly one file
                ls.close()

                if filename is None:
                    raise ValueError('no file responding to the file path "{}"'.format(filepath))

                if more is not None:
                    raise ValueError('there are more than one file responding to the file path "{}"'.format(filepath))

                filenames.append(filename)
        return filenames

    def read_mseed(self,
                   east_component_filename: str,
                   north_component_filename: str,
                   vertical_component_filename: str,
                   window_starttime: UTCDateTime,
                   window_endtime: UTCDateTime) -> (str, np.ndarray, np.ndarray):
        """
        default mseed preprocessing here
        modif 06 apr 2020, M.L.
            The method now works with 3 indep files for components E, N, Z, as in real world
            the time array is preserved so that accurate time picks are provided
            (indexs (itp, its) were not accurate because the data is transformed by obspy.merge)

        """
        estream = ocread(east_component_filename, format="MSEED", starttime=window_starttime, endtime=window_endtime)
        nstream = ocread(north_component_filename, format="MSEED", starttime=window_starttime, endtime=window_endtime)
        zstream = ocread(vertical_component_filename, format="MSEED", starttime=window_starttime, endtime=window_endtime)

        for st, expected_comp in zip([estream, nstream, zstream], 'ENZ'):
            if not len(st):
                raise ValueError('no traces found in stream (trim window outside?)')
            for tr in st:
                if tr.stats.sampling_rate != self.config.sampling_rate:
                    logger.warning(f'Sampling rate was {tr.stats.sampling_rate}Hz')

                    # try obspy resampler ...
                    tr.resample(
                        sampling_rate=self.config.sampling_rate,
                        no_filter=False,
                        strict_length=False)

                    if tr.stats.sampling_rate != self.config.sampling_rate:
                        raise Exception('obspy resample failed')

                if tr.stats.channel[2] != expected_comp:
                    raise ValueError(
                        f'Channel was {tr.stats.channel} '
                        f'and I was expecting ??{expected_comp}')

        for st in estream, nstream, zstream:

            st.detrend('constant')
            st.merge(fill_value=0)

            if not len(st) == 1:
                raise ValueError(f'obspy merge failed {len(st)}')  # QC

            st.trim(window_starttime, window_endtime, pad=True, fill_value=0)

            if not st[0].stats.sampling_rate == estream[0].stats.sampling_rate:
                raise ValueError('inconsistent sampling rates')  # QC

            if not np.abs(st[0].stats.starttime.timestamp - estream[0].stats.starttime.timestamp) < 1.e-6:
                raise ValueError('inconsistent starttimes')  # QC

        seedid = SEEDID.format(
            network=st[0].stats.network,
            station=st[0].stats.station,
            location=st[0].stats.location,
            channel2=st[0].stats.channel[:2],
            dataquality=st[0].stats.mseed.dataquality)

        data = np.vstack([st[0].data for st in [estream, nstream, zstream]])

        start = zstream[0].stats.starttime
        nt = data.shape[1]
        dt = zstream[0].stats.delta
        timearray = start.timestamp + np.arange(nt) * dt

        # # can test small sampling rate for longer distance
        # meta = meta.interpolate(sampling_rate=100)

        pad_width = int((np.ceil((nt - 1) / self.input_length)) * self.input_length - nt)
        if pad_width == -1:
            data = data[:, :-1]
            nt -= 1
            timearray = timearray[:-1]
        else:
            # pad the data
            data = np.pad(data, ((0, 0), (0, pad_width)), 'constant', constant_values=(0, 0))
            # recompute the time array
            nt = data.shape[1]
            timearray = start.timestamp + np.arange(nt) * dt

        # repeat the data twice for 50% overlapping
        data = np.hstack([
            data,
            np.zeros_like(data[:, :self.input_length // 2]),
            data[:, :-self.input_length // 2]])

        # naive version, do exactly the same with the time array as with the data
        # to ensure that the time synchronization is preserved
        timearray = np.hstack([timearray, timearray - self.input_length // 2 * dt])

        # one depth (axis 0) per component E, N, Z
        # one raw (axis 1) per window
        # one column (axis 2) per sample in the window
        data = data.reshape((3, -1, self.input_length))
        timearray = timearray.reshape((-1, self.input_length))  # naive
        timearray = timearray[:, 0]  # keep only the starttime of each window in s since epoch

        # depths become the window numbers
        # lines become the samples inside the windows
        # columns become the component number (e,n,z)
        # then a 1d axis is added in 2nd dimension
        data = data.transpose(1, 2, 0)[:, :, np.newaxis, :]

        return seedid, data, timearray

    def get_csv_row(self, row_index: int) -> (str, str, str, str, str, int, int, UTCDateTime, UTCDateTime):
        """get station indications from csv, may include wildcards"""

        network = str(self.data_list.iloc[row_index]['network'])  # e.g. "FR"
        station = str(self.data_list.iloc[row_index]['station'])  # e.g. "ABC" or "ABCD" or "ABCDE"
        location = str(self.data_list.iloc[row_index]['location'])  # e.g. "00" or "*" or ""
        dataquality = str(self.data_list.iloc[row_index]['dataquality'])  # e.g. "D" or "?"
        channel = str(self.data_list.iloc[row_index]['channel'])  # e.g. "EH*" or "EH?" or "HH?" ...
        year = int(self.data_list.iloc[row_index]['year'])  # e.g. 2014
        julday = int(self.data_list.iloc[row_index]['julday'])  # e.g. 14
        starttime_in_day_sec = float(self.data_list.iloc[row_index]['starttime_in_day_sec'])  # e.g. 0.
        endtime_in_day_sec = float(self.data_list.iloc[row_index]['endtime_in_day_sec'])  # e.g. 86400. (24 * 60 * 60)

        # warning blank fields in csv will correspond to "nan" string here
        if location == "nan":
            location = ""

        if not 0. <= starttime_in_day_sec < endtime_in_day_sec <= 24 * 60 * 60:
            raise ValueError(
                f' I need 0. '
                f'<= starttime_in_day_sec ({starttime_in_day_sec}) '
                f'< endtime_in_day_sec ({endtime_in_day_sec}) <= 24. * 60. * 60.')

        # define the time window to read here
        window_starttime = UTCDateTime(year=year, julday=julday, hour=0) + starttime_in_day_sec
        window_endtime = UTCDateTime(year=year, julday=julday, hour=0) + endtime_in_day_sec

        return network, station, location, dataquality, channel, year, julday, window_starttime, window_endtime

    def thread_main(self, sess, n_threads=1, start=0):

        for i in range(start, self.num_data, n_threads):

            # ======== this section must not fail due to reading errors
            network, station, location, dataquality, channel, \
                year, julday, \
                window_starttime, window_endtime = \
                self.get_csv_row(row_index=i)

            # ======== this section will ignore reading errors with a warning message
            try:
                # look for 3 component data files according to csv data
                filenames = self.find_filenames(
                    network, station, location, channel, dataquality, year, julday)

                # read the files
                seedid, data, timearray = self.read_mseed(
                    east_component_filename=filenames[0],  # east comp
                    north_component_filename=filenames[1],  # north comp
                    vertical_component_filename=filenames[2],  # vert comp
                    window_starttime=window_starttime,
                    window_endtime=window_endtime)

            except (IOError, ValueError, TypeError) as e:
                # an error occured, notify user but do not interrupt the process => replace by exception to debug
                warining_message = \
                    f"WARNING : reading data " \
                    f"network:{network} station:{station} " \
                    f"location:{location} channel:{channel} " \
                    f"year:{year} julday:{julday}" \
                    f"window_starttime:{str(window_starttime)} " \
                    f"window_endtime:{str(window_endtime)}" \
                    f" failed (reason:{str(e)})"

                logger.warning(warining_message)
                continue

            except BaseException as e:
                logger.error(
                    'Exception or BaseException should not be skept, '      
                    'add the following type to the exception list above if appropriate '
                    '{}'.format(e.__class__.__name__))
                raise e

            # ========
            thread_description = \
                f"{seedid:s}.{year:04d}.{julday:03d} " \
                f"[{window_starttime.hour:02d}:" \
                f"{window_starttime.minute:02d}:" \
                f"{window_starttime.second:02d}:" \
                f"{window_starttime.microsecond*1e-6:06.0f}-" \
                f"{window_endtime.hour:02d}:" \
                f"{window_endtime.minute:02d}:" \
                f"{window_endtime.second:02d}:" \
                f"{window_endtime.microsecond*1e-6:06.0f}]"

            for i in tqdm(range(data.shape[0]), desc=thread_description):

                sample_starttime_timestamp = timearray[i]
                sample_starttime = UTCDateTime(sample_starttime_timestamp)

                sample_name = \
                    SAMPLENAME.format(
                    seedid=seedid,
                    year=sample_starttime.year,
                    julday=sample_starttime.julday,
                    hour=sample_starttime.hour,
                    minute=sample_starttime.minute,
                    second=sample_starttime.second + 1.e-6 * sample_starttime.microsecond,
                    sampling_rate=self.config.sampling_rate,
                    input_length=self.input_length)

                # loop over windows
                sample = data[i]
                sample = self.normalize(sample)
                sample = self.adjust_missingchannels(sample)
                sess.run(self.enqueue,
                         feed_dict={
                             self.sample_placeholder: sample,
                             self.fname_placeholder: sample_name})


# ============ run
def show_sds_prediction_results(fig, data_reader: DataReaderSDS, log_dir=None):
    assert os.path.isdir(log_dir)
    hdf5_archive = os.path.join(log_dir, 'results', "sample_results.hdf5")
    picks_file = os.path.join(log_dir, "picks.csv")
    figure_dir = os.path.join(log_dir, 'figures')
    if not os.path.isdir(figure_dir):
        os.mkdir(figure_dir)

    assert os.path.isdir(figure_dir)
    assert os.path.isfile(picks_file)
    assert os.path.isfile(hdf5_archive)

    # read picks.csv
    A = np.genfromtxt(picks_file, skip_header=1, delimiter=',', dtype=str)
    pick_data = {}
    pick_data['seedid'] = A[:, 0]
    pick_data['phasename'] = A[:, 1]
    pick_data['picktime'] = np.asarray([UTCDateTime(_).timestamp for _ in A[:, 2]], float)
    pick_data['probability'] = np.asarray(A[:, 3], float)


    fig.clf()
    ax = fig.add_subplot(111)
    bx = ax.twinx()
    gain = 0.1

    with h5py.File(hdf5_archive, 'r') as h5fid:
        for row_index in range(len(data_reader.data_list)):

            # loop over rows in the input csv file
            network, station, location, dataquality, channel, \
                year, julday, \
                window_starttime, window_endtime = \
                data_reader.get_csv_row(row_index=row_index)

            # find the picks that correspond to the current station
            I = (pick_data['seedid'] == f"{network}.{station}.{location}.{channel[:2]}.{dataquality}")

            # find the picks that fall in the current time window
            I &= (window_starttime.timestamp <= pick_data['picktime'])
            I &= (window_endtime.timestamp >= pick_data['picktime'])

            if not I.any():
                continue

            # convert mask to indexs
            ipicks = np.arange(len(I))[I]

            # find the probability data
            from obspy.core import Trace, Stream
            stp, sts = Stream([]), Stream([])
            for phasename in "PS":
                sample_path = HDF5PATH.format(
                    network=network, station=station,
                    location=location, channel2=channel[:2],
                    dataquality=dataquality,
                    year=year, julday=julday,
                    phasename=phasename)

                for sample_name in h5fid[sample_path]:
                    sample = h5fid[sample_path][sample_name]
                    tr = Trace(data=np.array(sample[:], float) / 255., header=dict(**sample.attrs))
                    if phasename == "P":
                        stp.append(tr)
                    elif phasename == "S":
                        sts.append(tr)
                    else:
                        raise Exception

            # find the seismic datafiles
            east_component_filename, north_component_filename, vertical_component_filename = \
                data_reader.find_filenames(
                network, station, location, channel, dataquality, year, julday)

            # seedid, data, timearray = data_reader.read_mseed(
            #     east_component_filename, north_component_filename, vertical_component_filename,
            #     window_starttime, window_endtime)

            # read the row data
            ste = ocread(east_component_filename, format="MSEED", starttime=window_starttime, endtime=window_endtime)
            stn = ocread(north_component_filename, format="MSEED", starttime=window_starttime, endtime=window_endtime)
            stz = ocread(vertical_component_filename, format="MSEED", starttime=window_starttime, endtime=window_endtime)
            if not len(stz) or not len(ste) or not len(stn):
                continue

            # preprocessing for display
            def stpro(st):
                tr = st.merge(fill_value=0.)[0]
                tr.detrend('linear')
                tr.data /= tr.data.std()
                return tr
            trz, trn, tre = [stpro(st) for st in [stz, stn, ste]]

            # display the background traces
            ax.cla()
            bx.cla()
            for n, tr in enumerate([trz, trn, tre]):
                t = tr.stats.starttime.timestamp + np.arange(tr.stats.npts) * tr.stats.delta
                ax.plot(t, gain * tr.data + n, 'k', alpha=0.4)

            for n, st in enumerate([stp, sts]):
                for tr in st:
                    t = tr.stats.starttime.timestamp + np.arange(tr.stats.npts) * tr.stats.delta
                    bx.plot(t, tr.data, {0: "r", 1: "b"}[n], alpha=1.0)

            # add the picks
            for ipick in ipicks:

                ax.plot(pick_data['picktime'][ipick] * np.ones(2),
                        [-1, 3.],
                        color={"P": "r", "S": "b"}[pick_data['phasename'][ipick]])

            ax.set_title(f"{network}.{station}.{location}.{channel[:2]}.{dataquality}")
            ax.set_xlabel(f"time")
            ax.set_yticks([0, 1, 2])
            ax.set_yticklabels(["Z", "N", "E"])
            ax.set_ylim(-1., 3.)
            bx.set_ylim(-0.1, 1.1)
            bx.set_ylabel("probability")

            # make one plot per pick, centered on the pick
            for ipick in ipicks:
                t = pick_data['picktime'][ipick]
                figstart = t - 1500. * trz.stats.delta
                figend = t + 1500. * trz.stats.delta
                ax.set_xlim(figstart, figend)

                ax.figure.canvas.draw()
                # ax.figure.show()
                # input('pause')

                figname = f"{network}.{station}.{location}.{channel[:2]}.{dataquality}.{ipick:04d}.png"
                figname = os.path.join(figure_dir, figname)
                print(figname)
                ax.figure.savefig(figname)


def pred_fn_sds(args, data_reader: DataReaderSDS, figure_dir=None, result_dir=None, log_dir=None):
    """
    prediction function, modified after pred_fn for SDS data
    :param args: from the argument parser
    :param data_reader: the data reader object (DataReaderSDS)
    :param figure_dir:
    :param result_dir:
    :param log_dir:
    :return:
    """
    assert args.input_sds
    if args.plot_figure:
        raise NotImplementedError('was disabled in this version')

    current_time = time.strftime("%y%m%d-%H%M%S")
    if log_dir is None:
        log_dir = os.path.join(args.log_dir, "pred", current_time)
    logging.info("Pred log: %s" % log_dir)
    logging.info("Dataset size: {}".format(data_reader.num_data))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    #
    # if args.plot_figure and (figure_dir is None):
    #     figure_dir = os.path.join(log_dir, 'figures')
    #     if not os.path.exists(figure_dir):
    #         os.makedirs(figure_dir)

    if args.save_result and (result_dir is None):
        result_dir = os.path.join(log_dir, 'results')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

    config = set_config(args, data_reader)
    with open(os.path.join(log_dir, 'config.log'), 'w') as fp:
        fp.write('\n'.join("%s: %s" % item for item in vars(config).items()))

    with tf.name_scope('Input_Batch'):
        batch = data_reader.dequeue(args.batch_size)

    model = Model(config, batch, "pred")
    sess_config = tf.compat.v1.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.log_device_placement = False

    with tf.compat.v1.Session(config=sess_config) as sess:

        threads = data_reader.start_threads(sess, n_threads=8)

        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=5)
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        logging.info("restoring models...")
        latest_check_point = tf.compat.v1.train.latest_checkpoint(args.model_dir)
        saver.restore(sess, latest_check_point)

        if args.plot_figure:
            num_pool = multiprocessing.cpu_count() * 2
        elif args.save_result:
            num_pool = multiprocessing.cpu_count()
        else:
            num_pool = 2
        pool = multiprocessing.Pool(num_pool)
        fclog = open(os.path.join(log_dir, args.fpred + '.csv'), 'w')

        if args.save_result:
            assert result_dir is not None
            hdf5_archive = os.path.join(result_dir, "sample_results.hdf5")

        # write pick file header
        # fclog.write("batchname,itp,tp_prob,its,ts_prob\n")
        fclog.write("seedid,phasename,time,probability\n")

        while True:
            if sess.run(data_reader.queue.size()) >= args.batch_size:
                break
            time.sleep(2)
            # print("waiting data_reader...")

        while True:
            last_batch = True
            for i in range(10):
                if sess.run(data_reader.queue.size()) >= args.batch_size:
                    last_batch = False
                    break
                time.sleep(1)
            if last_batch:
                for t in threads:
                    t.join()
                last_size = sess.run(data_reader.queue.size())
                logger.info(f"Last batch: {last_size} samples")
                sess.run(data_reader.queue.close())
                if last_size == 0:
                    break

            pred_batch, X_batch, fname_batch = \
                sess.run([model.preds, batch[0], batch[1]],
                         feed_dict={model.drop_rate: 0,
                                    model.is_training: False})

            # place predictions results into a large hdf5 archive
            if args.save_result:
                with h5py.File(hdf5_archive, 'a') as hdf5_pointer:
                    _save_predictions_to_hdf5_archive(
                        hdf5_pointer, fname_batch, pred_batch)

            # picks
            picks_batch = pool.map(
                partial(_postprocessing_thread_sds,
                        pred=pred_batch,
                        X=X_batch,
                        fname=fname_batch,
                        result_dir=None,  # force ignore this
                        figure_dir=None,  # force ignore this
                        args=args),
                range(len(pred_batch)))

            # get the picks and write it to csv (picks.csv)
            for i in range(len(fname_batch)):
                seedid, sample_start, sampling_rate, sample_npts, _ = \
                    _decode_sample_name(fname_batch[i].decode())

                itp, tpprob = picks_batch[i][0]
                its, tsprob = picks_batch[i][1]

                for idx, pb in zip(itp, tpprob):
                    # find pick time from batchname metadata
                    tpick = sample_start + idx / sampling_rate
                    fclog.write(f"{seedid},P,{tpick},{pb}\n")

                for idx, pb in zip(its, tsprob):
                    tpick = sample_start + idx / sampling_rate
                    fclog.write(f"{seedid},S,{tpick},{pb}\n")

            if last_batch:
                break

        if args.save_result:
            # load sample prediction and concatenate them into
            # mseed files with the same structure as the input sds tree
            logging.info('forming mseed files with the P and S prediction series...')
            logging.warning('disabled for now. TODO : move this to a separate script')
            # with h5py.File(hdf5_archive, 'r') as hdf5_pointer:
            #     reform_mseed_files_from_predictions(
            #         hdf5_pointer, result_dir)

        fclog.close()
        logger.info("Done")

    return 0


if __name__ == '__main__':

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
    assert os.path.isdir(args.model_dir)
    assert os.path.isdir(args.data_dir)
    assert os.path.isfile(args.data_list)
    assert not os.path.isdir(args.output_dir), f"output dir exists already {args.output_dir}"

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    coord = tf.train.Coordinator()

    data_reader = DataReaderSDS(
        data_dir=args.data_dir,
        data_list=args.data_list,
        queue_size=args.batch_size * 10,
        coord=coord,
        input_length=args.input_length)

    pred_fn_sds(args, data_reader, log_dir=args.output_dir)
