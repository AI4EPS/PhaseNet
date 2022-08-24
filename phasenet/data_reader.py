import tensorflow as tf

tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import logging
import os

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None
import json

# import s3fs
import h5py
import obspy
from scipy.interpolate import interp1d
from tqdm import tqdm


def py_func_decorator(output_types=None, output_shapes=None, name=None):
    def decorator(func):
        def call(*args, **kwargs):
            nonlocal output_shapes
            # flat_output_types = nest.flatten(output_types)
            flat_output_types = tf.nest.flatten(output_types)
            # flat_values = tf.py_func(
            flat_values = tf.numpy_function(func, inp=args, Tout=flat_output_types, name=name)
            if output_shapes is not None:
                for v, s in zip(flat_values, output_shapes):
                    v.set_shape(s)
            # return nest.pack_sequence_as(output_types, flat_values)
            return tf.nest.pack_sequence_as(output_types, flat_values)

        return call

    return decorator


def dataset_map(iterator, output_types, output_shapes=None, num_parallel_calls=None, name=None, shuffle=False):
    dataset = tf.data.Dataset.range(len(iterator))
    if shuffle:
        dataset = dataset.shuffle(len(iterator), reshuffle_each_iteration=True)

    @py_func_decorator(output_types, output_shapes, name=name)
    def index_to_entry(idx):
        return iterator[idx]

    return dataset.map(index_to_entry, num_parallel_calls=num_parallel_calls)


def normalize(data, axis=(0,)):
    """data shape: (nt, nsta, nch)"""
    data -= np.mean(data, axis=axis, keepdims=True)
    std_data = np.std(data, axis=axis, keepdims=True)
    std_data[std_data == 0] = 1
    data /= std_data
    # data /= (std_data + 1e-12)
    return data


def normalize_long(data, axis=(0,), window=3000):
    """
    data: nt, nch
    """
    nt, nar, nch = data.shape
    if window is None:
        window = nt
    shift = window // 2

    ## std in slide windows
    data_pad = np.pad(data, ((window // 2, window // 2), (0, 0), (0, 0)), mode="reflect")
    t = np.arange(0, nt, shift, dtype="int")
    std = np.zeros([len(t) + 1, nar, nch])
    mean = np.zeros([len(t) + 1, nar, nch])
    for i in range(1, len(std)):
        std[i, :] = np.std(data_pad[i * shift : i * shift + window, :, :], axis=axis)
        mean[i, :] = np.mean(data_pad[i * shift : i * shift + window, :, :], axis=axis)

    t = np.append(t, nt)
    # std[-1, :] = np.std(data_pad[-window:, :], axis=0)
    # mean[-1, :] = np.mean(data_pad[-window:, :], axis=0)
    std[-1, ...], mean[-1, ...] = std[-2, ...], mean[-2, ...]
    std[0, ...], mean[0, ...] = std[1, ...], mean[1, ...]
    # std[std == 0] = 1.0

    ## normalize data with interplated std
    t_interp = np.arange(nt, dtype="int")
    std_interp = interp1d(t, std, axis=0, kind="slinear")(t_interp)
    # std_interp = np.exp(interp1d(t, np.log(std), axis=0, kind="slinear")(t_interp))
    mean_interp = interp1d(t, mean, axis=0, kind="slinear")(t_interp)
    tmp = np.sum(std_interp, axis=(0, 1))
    std_interp[std_interp == 0] = 1.0
    data = (data - mean_interp) / std_interp
    # data = (data - mean_interp)/(std_interp + 1e-12)

    ### dropout effect of < 3 channel
    nonzero = np.count_nonzero(tmp)
    if (nonzero < 3) and (nonzero > 0):
        data *= 3.0 / nonzero

    return data


def normalize_batch(data, window=3000):
    """
    data: nsta, nt, nch
    """
    nsta, nt, nar, nch = data.shape
    if window is None:
        window = nt
    shift = window // 2

    ## std in slide windows
    data_pad = np.pad(data, ((0, 0), (window // 2, window // 2), (0, 0), (0, 0)), mode="reflect")
    t = np.arange(0, nt, shift, dtype="int")
    std = np.zeros([nsta, len(t) + 1, nar, nch])
    mean = np.zeros([nsta, len(t) + 1, nar, nch])
    for i in range(1, len(t)):
        std[:, i, :, :] = np.std(data_pad[:, i * shift : i * shift + window, :, :], axis=1)
        mean[:, i, :, :] = np.mean(data_pad[:, i * shift : i * shift + window, :, :], axis=1)

    t = np.append(t, nt)
    # std[:, -1, :] = np.std(data_pad[:, -window:, :], axis=1)
    # mean[:, -1, :] = np.mean(data_pad[:, -window:, :], axis=1)
    std[:, -1, :, :], mean[:, -1, :, :] = std[:, -2, :, :], mean[:, -2, :, :]
    std[:, 0, :, :], mean[:, 0, :, :] = std[:, 1, :, :], mean[:, 1, :, :]
    # std[std == 0] = 1

    # ## normalize data with interplated std
    t_interp = np.arange(nt, dtype="int")
    std_interp = interp1d(t, std, axis=1, kind="slinear")(t_interp)
    # std_interp = np.exp(interp1d(t, np.log(std), axis=1, kind="slinear")(t_interp))
    mean_interp = interp1d(t, mean, axis=1, kind="slinear")(t_interp)
    tmp = np.sum(std_interp, axis=(1, 2))
    std_interp[std_interp == 0] = 1.0
    data = (data - mean_interp) / std_interp
    # data = (data - mean_interp)/(std_interp + 1e-12)

    ### dropout effect of < 3 channel
    nonzero = np.count_nonzero(tmp, axis=-1)
    data[nonzero > 0, ...] *= 3.0 / nonzero[nonzero > 0][:, np.newaxis, np.newaxis, np.newaxis]

    return data


class DataConfig:

    seed = 123
    use_seed = True
    n_channel = 3
    n_class = 3
    sampling_rate = 100
    dt = 1.0 / sampling_rate
    X_shape = [3000, 1, n_channel]
    Y_shape = [3000, 1, n_class]
    min_event_gap = 3 * sampling_rate
    label_shape = "gaussian"
    label_width = 30
    dtype = "float32"

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class DataReader:
    def __init__(self, format="numpy", config=DataConfig(), **kwargs):
        self.buffer = {}
        self.n_channel = config.n_channel
        self.n_class = config.n_class
        self.X_shape = config.X_shape
        self.Y_shape = config.Y_shape
        self.dt = config.dt
        self.dtype = config.dtype
        self.label_shape = config.label_shape
        self.label_width = config.label_width
        self.config = config
        self.format = format
        if "highpass_filter" in kwargs:
            self.highpass_filter = kwargs["highpass_filter"]
        if format in ["numpy", "mseed", "sac"]:
            self.data_dir = kwargs["data_dir"]
            try:
                csv = pd.read_csv(kwargs["data_list"], header=0, sep="[,|\s+]", engine="python")
            except:
                csv = pd.read_csv(kwargs["data_list"], header=0, sep="\t")
            self.data_list = csv["fname"]
            self.num_data = len(self.data_list)
        elif format == "hdf5":
            self.h5 = h5py.File(kwargs["hdf5_file"], "r", libver="latest", swmr=True)
            self.h5_data = self.h5[kwargs["hdf5_group"]]
            self.data_list = list(self.h5_data.keys())
            self.num_data = len(self.data_list)
        elif format == "s3":
            self.s3fs = s3fs.S3FileSystem(
                anon=kwargs["anon"],
                key=kwargs["key"],
                secret=kwargs["secret"],
                client_kwargs={"endpoint_url": kwargs["s3_url"]},
                use_ssl=kwargs["use_ssl"],
            )
            self.num_data = 0
        else:
            raise (f"{format} not support!")

    def __len__(self):
        return self.num_data

    def read_numpy(self, fname):
        # try:
        if fname not in self.buffer:
            npz = np.load(fname)
            meta = {}
            if len(npz["data"].shape) == 2:
                meta["data"] = npz["data"][:, np.newaxis, :]
            else:
                meta["data"] = npz["data"]
            if "p_idx" in npz.files:
                if len(npz["p_idx"].shape) == 0:
                    meta["itp"] = [[npz["p_idx"]]]
                else:
                    meta["itp"] = npz["p_idx"]
            if "s_idx" in npz.files:
                if len(npz["s_idx"].shape) == 0:
                    meta["its"] = [[npz["s_idx"]]]
                else:
                    meta["its"] = npz["s_idx"]
            if "itp" in npz.files:
                if len(npz["itp"].shape) == 0:
                    meta["itp"] = [[npz["itp"]]]
                else:
                    meta["itp"] = npz["itp"]
            if "its" in npz.files:
                if len(npz["its"].shape) == 0:
                    meta["its"] = [[npz["its"]]]
                else:
                    meta["its"] = npz["its"]
            if "station_id" in npz.files:
                meta["station_id"] = npz["station_id"]
            if "sta_id" in npz.files:
                meta["station_id"] = npz["sta_id"]
            if "t0" in npz.files:
                meta["t0"] = npz["t0"]
            self.buffer[fname] = meta
        else:
            meta = self.buffer[fname]
        return meta
        # except:
        #     logging.error("Failed reading {}".format(fname))
        #     return None

    def read_hdf5(self, fname):
        data = self.h5_data[fname][()]
        attrs = self.h5_data[fname].attrs
        meta = {}
        if len(data.shape) == 2:
            meta["data"] = data[:, np.newaxis, :]
        else:
            meta["data"] = data
        if "p_idx" in attrs:
            if len(attrs["p_idx"].shape) == 0:
                meta["itp"] = [[attrs["p_idx"]]]
            else:
                meta["itp"] = attrs["p_idx"]
        if "s_idx" in attrs:
            if len(attrs["s_idx"].shape) == 0:
                meta["its"] = [[attrs["s_idx"]]]
            else:
                meta["its"] = attrs["s_idx"]
        if "itp" in attrs:
            if len(attrs["itp"].shape) == 0:
                meta["itp"] = [[attrs["itp"]]]
            else:
                meta["itp"] = attrs["itp"]
        if "its" in attrs:
            if len(attrs["its"].shape) == 0:
                meta["its"] = [[attrs["its"]]]
            else:
                meta["its"] = attrs["its"]
        if "t0" in attrs:
            meta["t0"] = attrs["t0"]
        return meta

    def read_s3(self, format, fname, bucket, key, secret, s3_url, use_ssl):
        with self.s3fs.open(bucket + "/" + fname, "rb") as fp:
            if format == "numpy":
                meta = self.read_numpy(fp)
            elif format == "mseed":
                meta = self.read_mseed(fp)
            else:
                raise (f"Format {format} not supported")
        return meta

    def read_mseed(self, fname):

        mseed = obspy.read(fname)
        mseed = mseed.detrend("spline", order=2, dspline=5 * mseed[0].stats.sampling_rate)
        mseed = mseed.merge(fill_value=0)
        if self.highpass_filter > 0:
            mseed = mseed.filter("highpass", freq=self.highpass_filter)
        starttime = min([st.stats.starttime for st in mseed])
        endtime = max([st.stats.endtime for st in mseed])
        mseed = mseed.trim(starttime, endtime, pad=True, fill_value=0)
        if abs(mseed[0].stats.sampling_rate - self.config.sampling_rate) > 1:
            logging.warning(
                f"Sampling rate mismatch in {fname.split('/')[-1]}: {mseed[0].stats.sampling_rate}Hz != {self.config.sampling_rate}Hz "
            )

        order = ["3", "2", "1", "E", "N", "Z"]
        order = {key: i for i, key in enumerate(order)}
        comp2idx = {"3": 0, "2": 1, "1": 2, "E": 0, "N": 1, "Z": 2}

        t0 = starttime.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        nt = len(mseed[0].data)
        data = np.zeros([nt, self.config.n_channel], dtype=self.dtype)
        ids = [x.get_id() for x in mseed]

        for j, id in enumerate(sorted(ids, key=lambda x: order[x[-1]])):
            if len(ids) != 3:
                if len(ids) > 3:
                    logging.warning(f"More than 3 channels {ids}!")
                j = comp2idx[id[-1]]
            data[:, j] = mseed.select(id=id)[0].data.astype(self.dtype)

        data = data[:, np.newaxis, :]
        meta = {"data": data, "t0": t0}
        return meta

    def read_sac(self, fname):

        mseed = obspy.read(fname)
        mseed = mseed.detrend("spline", order=2, dspline=5 * mseed[0].stats.sampling_rate)
        mseed = mseed.merge(fill_value=0)
        if self.highpass_filter > 0:
            mseed = mseed.filter("highpass", freq=self.highpass_filter)
        starttime = min([st.stats.starttime for st in mseed])
        endtime = max([st.stats.endtime for st in mseed])
        mseed = mseed.trim(starttime, endtime, pad=True, fill_value=0)
        if abs(mseed[0].stats.sampling_rate - self.config.sampling_rate) > 1:
            logging.warning(
                f"Sampling rate mismatch in {fname.split('/')[-1]}: {mseed[0].stats.sampling_rate}Hz != {self.config.sampling_rate}Hz "
            )

        order = ["3", "2", "1", "E", "N", "Z"]
        order = {key: i for i, key in enumerate(order)}
        comp2idx = {"3": 0, "2": 1, "1": 2, "E": 0, "N": 1, "Z": 2}

        t0 = starttime.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        nt = len(mseed[0].data)
        data = np.zeros([nt, self.config.n_channel], dtype=self.dtype)
        ids = [x.get_id() for x in mseed]
        for j, id in enumerate(sorted(ids, key=lambda x: order[x[-1]])):
            if len(ids) != 3:
                if len(ids) > 3:
                    logging.warning(f"More than 3 channels {ids}!")
                j = comp2idx[id[-1]]
            data[:, j] = mseed.select(id=id)[0].data.astype(self.dtype)

        data = data[:, np.newaxis, :]
        meta = {"data": data, "t0": t0}
        return meta

    def read_mseed_array(self, fname, stations, amplitude=False, remove_resp=True):

        data = []
        station_id = []
        t0 = []
        raw_amp = []

        try:
            mseed = obspy.read(fname)
            read_success = True
        except Exception as e:
            read_success = False
            print(e)

        if read_success:
            try:
                mseed = mseed.merge(fill_value=0)
            except Exception as e:
                print(e)

            for i in range(len(mseed)):
                if mseed[i].stats.sampling_rate != self.config.sampling_rate:
                    logging.warning(
                        f"Resampling {mseed[i].id} from {mseed[i].stats.sampling_rate} to {self.config.sampling_rate} Hz"
                    )
                    try:
                        mseed[i] = mseed[i].interpolate(self.config.sampling_rate, method="linear")
                    except Exception as e:
                        print(e)
                        mseed[i].data = mseed[i].data.astype(float) * 0.0  ## set to zero if resampling fails

            if self.highpass_filter == 0:
                try:
                    mseed = mseed.detrend("spline", order=2, dspline=5 * mseed[0].stats.sampling_rate)
                except:
                    logging.error(f"Error: spline detrend failed at file {fname}")
                    mseed = mseed.detrend("demean")
            else:
                mseed = mseed.filter("highpass", freq=self.highpass_filter)

            starttime = min([st.stats.starttime for st in mseed])
            endtime = max([st.stats.endtime for st in mseed])
            mseed = mseed.trim(starttime, endtime, pad=True, fill_value=0)

            order = ["3", "2", "1", "E", "N", "Z"]
            order = {key: i for i, key in enumerate(order)}
            comp2idx = {"3": 0, "2": 1, "1": 2, "E": 0, "N": 1, "Z": 2}

            nsta = len(stations)
            nt = len(mseed[0].data)
            # for i in range(nsta):
            for sta in stations:
                trace_data = np.zeros([nt, self.config.n_channel], dtype=self.dtype)
                if amplitude:
                    trace_amp = np.zeros([nt, self.config.n_channel], dtype=self.dtype)
                empty_station = True
                # sta = stations.iloc[i]["station"]
                # comp = stations.iloc[i]["component"].split(",")
                comp = stations[sta]["component"]
                if amplitude:
                    # resp = stations.iloc[i]["response"].split(",")
                    resp = stations[sta]["response"]

                for j, c in enumerate(sorted(comp, key=lambda x: order[x[-1]])):

                    resp_j = resp[j]
                    if len(comp) != 3:  ## less than 3 component
                        j = comp2idx[c]

                    if len(mseed.select(id=sta + c)) == 0:
                        print(f"Empty trace: {sta+c} {starttime}")
                        continue
                    else:
                        empty_station = False

                    tmp = mseed.select(id=sta + c)[0].data.astype(self.dtype)
                    trace_data[: len(tmp), j] = tmp[:nt]
                    if amplitude:
                        # if stations.iloc[i]["unit"] == "m/s**2":
                        if stations[sta]["unit"] == "m/s**2":
                            tmp = mseed.select(id=sta + c)[0]
                            tmp = tmp.integrate()
                            tmp = tmp.filter("highpass", freq=1.0)
                            tmp = tmp.data.astype(self.dtype)
                            trace_amp[: len(tmp), j] = tmp[:nt]
                        # elif stations.iloc[i]["unit"] == "m/s":
                        elif stations[sta]["unit"] == "m/s":
                            tmp = mseed.select(id=sta + c)[0].data.astype(self.dtype)
                            trace_amp[: len(tmp), j] = tmp[:nt]
                        else:
                            print(
                                f"Error in {stations.iloc[i]['station']}\n{stations.iloc[i]['unit']} should be m/s**2 or m/s!"
                            )
                    if amplitude and remove_resp:
                        # trace_amp[:, j] /= float(resp[j])
                        trace_amp[:, j] /= float(resp_j)

                if not empty_station:
                    data.append(trace_data)
                    if amplitude:
                        raw_amp.append(trace_amp)
                    station_id.append(sta)
                    t0.append(starttime.datetime.isoformat(timespec="milliseconds"))

        if len(data) > 0:
            data = np.stack(data)
            if len(data.shape) == 3:
                data = data[:, :, np.newaxis, :]
            if amplitude:
                raw_amp = np.stack(raw_amp)
                if len(raw_amp.shape) == 3:
                    raw_amp = raw_amp[:, :, np.newaxis, :]
        else:
            nt = 60 * 60 * self.config.sampling_rate  # assume 1 hour data
            data = np.zeros([1, nt, 1, self.config.n_channel], dtype=self.dtype)
            if amplitude:
                raw_amp = np.zeros([1, nt, 1, self.config.n_channel], dtype=self.dtype)
            t0 = ["1970-01-01T00:00:00.000"]
            station_id = ["None"]

        if amplitude:
            meta = {"data": data, "t0": t0, "station_id": station_id, "fname": fname.split("/")[-1], "raw_amp": raw_amp}
        else:
            meta = {"data": data, "t0": t0, "station_id": station_id, "fname": fname.split("/")[-1]}
        return meta

    def generate_label(self, data, phase_list, mask=None):
        # target = np.zeros(self.Y_shape, dtype=self.dtype)
        target = np.zeros_like(data)

        if self.label_shape == "gaussian":
            label_window = np.exp(
                -((np.arange(-self.label_width // 2, self.label_width // 2 + 1)) ** 2)
                / (2 * (self.label_width / 5) ** 2)
            )
        elif self.label_shape == "triangle":
            label_window = 1 - np.abs(
                2 / self.label_width * (np.arange(-self.label_width // 2, self.label_width // 2 + 1))
            )
        else:
            print(f"Label shape {self.label_shape} should be guassian or triangle")
            raise

        for i, phases in enumerate(phase_list):
            for j, idx_list in enumerate(phases):
                for idx in idx_list:
                    if np.isnan(idx):
                        continue
                    idx = int(idx)
                    if (idx - self.label_width // 2 >= 0) and (idx + self.label_width // 2 + 1 <= target.shape[0]):
                        target[idx - self.label_width // 2 : idx + self.label_width // 2 + 1, j, i + 1] = label_window

            target[..., 0] = 1 - np.sum(target[..., 1:], axis=-1)
            if mask is not None:
                target[:, mask == 0, :] = 0

        return target

    def random_shift(self, sample, itp, its, itp_old=None, its_old=None, shift_range=None):
        # anchor = np.round(1/2 * (min(itp[~np.isnan(itp.astype(float))]) + min(its[~np.isnan(its.astype(float))]))).astype(int)
        flattern = lambda x: np.array([i for trace in x for i in trace], dtype=float)
        shift_pick = lambda x, shift: [[i - shift for i in trace] for trace in x]
        itp_flat = flattern(itp)
        its_flat = flattern(its)
        if (itp_old is None) and (its_old is None):
            hi = np.round(np.median(itp_flat[~np.isnan(itp_flat)])).astype(int)
            lo = -(sample.shape[0] - np.round(np.median(its_flat[~np.isnan(its_flat)])).astype(int))
            if shift_range is None:
                shift = np.random.randint(low=lo, high=hi + 1)
            else:
                shift = np.random.randint(low=max(lo, shift_range[0]), high=min(hi + 1, shift_range[1]))
        else:
            itp_old_flat = flattern(itp_old)
            its_old_flat = flattern(its_old)
            itp_ref = np.round(np.min(itp_flat[~np.isnan(itp_flat)])).astype(int)
            its_ref = np.round(np.max(its_flat[~np.isnan(its_flat)])).astype(int)
            itp_old_ref = np.round(np.min(itp_old_flat[~np.isnan(itp_old_flat)])).astype(int)
            its_old_ref = np.round(np.max(its_old_flat[~np.isnan(its_old_flat)])).astype(int)
            # min_event_gap = np.round(self.min_event_gap*(its_ref-itp_ref)).astype(int)
            # min_event_gap_old = np.round(self.min_event_gap*(its_old_ref-itp_old_ref)).astype(int)
            if shift_range is None:
                hi = list(range(max(its_ref - itp_old_ref + self.min_event_gap, 0), itp_ref))
                lo = list(range(-(sample.shape[0] - its_ref), -(max(its_old_ref - itp_ref + self.min_event_gap, 0))))
            else:
                lo_ = max(-(sample.shape[0] - its_ref), shift_range[0])
                hi_ = min(itp_ref, shift_range[1])
                hi = list(range(max(its_ref - itp_old_ref + self.min_event_gap, 0), hi_))
                lo = list(range(lo_, -(max(its_old_ref - itp_ref + self.min_event_gap, 0))))
            if len(hi + lo) > 0:
                shift = np.random.choice(hi + lo)
            else:
                shift = 0

        shifted_sample = np.zeros_like(sample)
        if shift > 0:
            shifted_sample[:-shift, ...] = sample[shift:, ...]
        elif shift < 0:
            shifted_sample[-shift:, ...] = sample[:shift, ...]
        else:
            shifted_sample[...] = sample[...]

        return shifted_sample, shift_pick(itp, shift), shift_pick(its, shift), shift

    def stack_events(self, sample_old, itp_old, its_old, shift_range=None, mask_old=None):

        i = np.random.randint(self.num_data)
        base_name = self.data_list[i]
        if self.format == "numpy":
            meta = self.read_numpy(os.path.join(self.data_dir, base_name))
        elif self.format == "hdf5":
            meta = self.read_hdf5(base_name)
        if meta == -1:
            return sample_old, itp_old, its_old

        sample = np.copy(meta["data"])
        itp = meta["itp"]
        its = meta["its"]
        if mask_old is not None:
            mask = np.copy(meta["mask"])
        sample = normalize(sample)
        sample, itp, its, shift = self.random_shift(sample, itp, its, itp_old, its_old, shift_range)

        if shift != 0:
            sample_old += sample
            # itp_old = [np.hstack([i, j]) for i,j in zip(itp_old, itp)]
            # its_old = [np.hstack([i, j]) for i,j in zip(its_old, its)]
            itp_old = [i + j for i, j in zip(itp_old, itp)]
            its_old = [i + j for i, j in zip(its_old, its)]
            if mask_old is not None:
                mask_old = mask_old * mask

        return sample_old, itp_old, its_old, mask_old

    def cut_window(self, sample, target, itp, its, select_range):
        shift_pick = lambda x, shift: [[i - shift for i in trace] for trace in x]
        sample = sample[select_range[0] : select_range[1]]
        target = target[select_range[0] : select_range[1]]
        return (sample, target, shift_pick(itp, select_range[0]), shift_pick(its, select_range[0]))


class DataReader_train(DataReader):
    def __init__(self, format="numpy", config=DataConfig(), **kwargs):

        super().__init__(format=format, config=config, **kwargs)

        self.min_event_gap = config.min_event_gap
        self.buffer_channels = {}
        self.shift_range = [-2000 + self.label_width * 2, 1000 - self.label_width * 2]
        self.select_range = [5000, 8000]

    def __getitem__(self, i):

        base_name = self.data_list[i]
        if self.format == "numpy":
            meta = self.read_numpy(os.path.join(self.data_dir, base_name))
        elif self.format == "hdf5":
            meta = self.read_hdf5(base_name)
        if meta == None:
            return (np.zeros(self.X_shape, dtype=self.dtype), np.zeros(self.Y_shape, dtype=self.dtype), base_name)

        sample = np.copy(meta["data"])
        itp_list = meta["itp"]
        its_list = meta["its"]

        sample = normalize(sample)
        if np.random.random() < 0.95:
            sample, itp_list, its_list, _ = self.random_shift(sample, itp_list, its_list, shift_range=self.shift_range)
            sample, itp_list, its_list, _ = self.stack_events(sample, itp_list, its_list, shift_range=self.shift_range)
            target = self.generate_label(sample, [itp_list, its_list])
            sample, target, itp_list, its_list = self.cut_window(sample, target, itp_list, its_list, self.select_range)
        else:
            ## noise
            assert self.X_shape[0] <= min(min(itp_list))
            sample = sample[: self.X_shape[0], ...]
            target = np.zeros(self.Y_shape).astype(self.dtype)
            itp_list = [[]]
            its_list = [[]]

        sample = normalize(sample)
        return (sample.astype(self.dtype), target.astype(self.dtype), base_name)

    def dataset(self, batch_size, num_parallel_calls=2, shuffle=True, drop_remainder=True):
        dataset = dataset_map(
            self,
            output_types=(self.dtype, self.dtype, "string"),
            output_shapes=(self.X_shape, self.Y_shape, None),
            num_parallel_calls=num_parallel_calls,
            shuffle=shuffle,
        )
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder).prefetch(batch_size * 2)
        return dataset


class DataReader_test(DataReader):
    def __init__(self, format="numpy", config=DataConfig(), **kwargs):

        super().__init__(format=format, config=config, **kwargs)

        self.select_range = [5000, 8000]

    def __getitem__(self, i):

        base_name = self.data_list[i]
        if self.format == "numpy":
            meta = self.read_numpy(os.path.join(self.data_dir, base_name))
        elif self.format == "hdf5":
            meta = self.read_hdf5(base_name)
        if meta == -1:
            return (np.zeros(self.Y_shape, dtype=self.dtype), np.zeros(self.X_shape, dtype=self.dtype), base_name)

        sample = np.copy(meta["data"])
        itp_list = meta["itp"]
        its_list = meta["its"]

        # sample, itp_list, its_list, _ = self.random_shift(sample, itp_list, its_list, shift_range=self.shift_range)
        target = self.generate_label(sample, [itp_list, its_list])
        sample, target, itp_list, its_list = self.cut_window(sample, target, itp_list, its_list, self.select_range)

        sample = normalize(sample)
        return (sample, target, base_name, itp_list, its_list)

    def dataset(self, batch_size, num_parallel_calls=2, shuffle=False, drop_remainder=False):
        dataset = dataset_map(
            self,
            output_types=(self.dtype, self.dtype, "string", "int64", "int64"),
            output_shapes=(self.X_shape, self.Y_shape, None, None, None),
            num_parallel_calls=num_parallel_calls,
            shuffle=shuffle,
        )
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder).prefetch(batch_size * 2)
        return dataset


class DataReader_pred(DataReader):
    def __init__(self, format="numpy", amplitude=True, config=DataConfig(), **kwargs):

        super().__init__(format=format, config=config, **kwargs)

        self.amplitude = amplitude
        self.X_shape = self.get_data_shape()

    def get_data_shape(self):
        base_name = self.data_list[0]
        if self.format == "numpy":
            meta = self.read_numpy(os.path.join(self.data_dir, base_name))
        elif self.format == "mseed":
            meta = self.read_mseed(os.path.join(self.data_dir, base_name))
        elif self.format == "sac":
            meta = self.read_sac(os.path.join(self.data_dir, base_name))
        elif self.format == "hdf5":
            meta = self.read_hdf5(base_name)
        return meta["data"].shape

    def adjust_missingchannels(self, data):
        tmp = np.max(np.abs(data), axis=0, keepdims=True)
        assert tmp.shape[-1] == data.shape[-1]
        if np.count_nonzero(tmp) > 0:
            data *= data.shape[-1] / np.count_nonzero(tmp)
        return data

    def __getitem__(self, i):

        base_name = self.data_list[i]

        if self.format == "numpy":
            meta = self.read_numpy(os.path.join(self.data_dir, base_name))
        elif self.format == "mseed":
            meta = self.read_mseed(os.path.join(self.data_dir, base_name))
        elif self.format == "sac":
            meta = self.read_sac(os.path.join(self.data_dir, base_name))
        elif self.format == "hdf5":
            meta = self.read_hdf5(base_name)
        else:
            raise (f"{self.format} does not support!")
        if meta == -1:
            return (np.zeros(self.X_shape, dtype=self.dtype), base_name)

        raw_amp = np.zeros(self.X_shape, dtype=self.dtype)
        raw_amp[: meta["data"].shape[0], ...] = meta["data"][: self.X_shape[0], ...]
        sample = np.zeros(self.X_shape, dtype=self.dtype)
        sample[: meta["data"].shape[0], ...] = normalize_long(meta["data"])[: self.X_shape[0], ...]
        if abs(meta["data"].shape[0] - self.X_shape[0]) > 1:
            logging.warning(f"Data length mismatch in {base_name}: {meta['data'].shape[0]} != {self.X_shape[0]}")

        if "t0" in meta:
            t0 = meta["t0"]
        else:
            t0 = "1970-01-01T00:00:00.000"

        if "station_id" in meta:
            station_id = meta["station_id"].split("/")[-1].rstrip("*")
        else:
            # station_id = base_name.split("/")[-1].rstrip("*")
            station_id = os.path.basename(base_name).rstrip("*")

        if np.isnan(sample).any() or np.isinf(sample).any():
            logging.warning(f"Data error: Nan or Inf found in {base_name}")
            sample[np.isnan(sample)] = 0
            sample[np.isinf(sample)] = 0

        # sample = self.adjust_missingchannels(sample)
        if self.amplitude:
            return (sample[: self.X_shape[0], ...], raw_amp[: self.X_shape[0], ...], base_name, t0, station_id)
        else:
            return (sample[: self.X_shape[0], ...], base_name, t0, station_id)

    def dataset(self, batch_size, num_parallel_calls=2, shuffle=False, drop_remainder=False):
        if self.amplitude:
            dataset = dataset_map(
                self,
                output_types=(self.dtype, self.dtype, "string", "string", "string"),
                output_shapes=(self.X_shape, self.X_shape, None, None, None),
                num_parallel_calls=num_parallel_calls,
                shuffle=shuffle,
            )
        else:
            dataset = dataset_map(
                self,
                output_types=(self.dtype, "string", "string", "string"),
                output_shapes=(self.X_shape, None, None, None),
                num_parallel_calls=num_parallel_calls,
                shuffle=shuffle,
            )
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder).prefetch(batch_size * 2)
        return dataset


class DataReader_mseed_array(DataReader):
    def __init__(self, stations, amplitude=True, remove_resp=True, config=DataConfig(), **kwargs):

        super().__init__(format="mseed", config=config, **kwargs)

        # self.stations = pd.read_json(stations)
        with open(stations, "r") as f:
            self.stations = json.load(f)
        print(pd.DataFrame.from_dict(self.stations, orient="index").to_string())

        self.amplitude = amplitude
        self.remove_resp = remove_resp
        self.X_shape = self.get_data_shape()

    def get_data_shape(self):
        fname = os.path.join(self.data_dir, self.data_list[0])
        meta = self.read_mseed_array(fname, self.stations, self.amplitude, self.remove_resp)
        return meta["data"].shape

    def __getitem__(self, i):

        fp = os.path.join(self.data_dir, self.data_list[i])
        # try:
        meta = self.read_mseed_array(fp, self.stations, self.amplitude, self.remove_resp)
        # except Exception as e:
        #     logging.error(f"Failed reading {fp}: {e}")
        #     if self.amplitude:
        #         return (np.zeros(self.X_shape).astype(self.dtype), np.zeros(self.X_shape).astype(self.dtype),
        #             [self.stations.iloc[i]["station"] for i in range(len(self.stations))], ["0" for i in range(len(self.stations))])
        #     else:
        #         return (np.zeros(self.X_shape).astype(self.dtype), ["" for i in range(len(self.stations))],
        #             [self.stations.iloc[i]["station"] for i in range(len(self.stations))])

        sample = np.zeros([len(meta["data"]), *self.X_shape[1:]], dtype=self.dtype)
        sample[:, : meta["data"].shape[1], :, :] = normalize_batch(meta["data"])[:, : self.X_shape[1], :, :]
        if np.isnan(sample).any() or np.isinf(sample).any():
            logging.warning(f"Data error: Nan or Inf found in {fp}")
            sample[np.isnan(sample)] = 0
            sample[np.isinf(sample)] = 0
        t0 = meta["t0"]
        base_name = meta["fname"]
        station_id = meta["station_id"]
        #         base_name = [self.stations.iloc[i]["station"]+"."+t0[i] for i in range(len(self.stations))]
        # base_name = [self.stations.iloc[i]["station"] for i in range(len(self.stations))]

        if self.amplitude:
            raw_amp = np.zeros([len(meta["raw_amp"]), *self.X_shape[1:]], dtype=self.dtype)
            raw_amp[:, : meta["raw_amp"].shape[1], :, :] = meta["raw_amp"][:, : self.X_shape[1], :, :]
            if np.isnan(raw_amp).any() or np.isinf(raw_amp).any():
                logging.warning(f"Data error: Nan or Inf found in {fp}")
                raw_amp[np.isnan(raw_amp)] = 0
                raw_amp[np.isinf(raw_amp)] = 0
            return (sample, raw_amp, base_name, t0, station_id)
        else:
            return (sample, base_name, t0, station_id)

    def dataset(self, num_parallel_calls=1, shuffle=False):
        if self.amplitude:
            dataset = dataset_map(
                self,
                output_types=(self.dtype, self.dtype, "string", "string", "string"),
                output_shapes=([None, *self.X_shape[1:]], [None, *self.X_shape[1:]], None, None, None),
                num_parallel_calls=num_parallel_calls,
            )
        else:
            dataset = dataset_map(
                self,
                output_types=(self.dtype, "string", "string", "string"),
                output_shapes=([None, *self.X_shape[1:]], None, None, None),
                num_parallel_calls=num_parallel_calls,
            )
        dataset = dataset.prefetch(1)
        #         dataset = dataset.prefetch(len(self.stations)*2)
        return dataset


###### test ########


def test_DataReader():
    import os
    import timeit

    import matplotlib.pyplot as plt

    if not os.path.exists("test_figures"):
        os.mkdir("test_figures")

    def plot_sample(sample, fname, label=None):
        plt.clf()
        plt.subplot(211)
        plt.plot(sample[:, 0, -1])
        if label is not None:
            plt.subplot(212)
            plt.plot(label[:, 0, 0])
            plt.plot(label[:, 0, 1])
            plt.plot(label[:, 0, 2])
        plt.savefig(f"test_figures/{fname.decode()}.png")

    def read(data_reader, batch=1):
        start_time = timeit.default_timer()
        if batch is None:
            dataset = data_reader.dataset(shuffle=False)
        else:
            dataset = data_reader.dataset(1, shuffle=False)
        sess = tf.compat.v1.Session()

        print(len(data_reader))
        print("-------", tf.data.Dataset.cardinality(dataset))
        num = 0
        x = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
        while True:
            num += 1
            # print(num)
            try:
                out = sess.run(x)
                if len(out) == 2:
                    sample, fname = out[0], out[1]
                    for i in range(len(sample)):
                        plot_sample(sample[i], fname[i])
                else:
                    sample, label, fname = out[0], out[1], out[2]
                    for i in range(len(sample)):
                        plot_sample(sample[i], fname[i], label[i])
            except tf.errors.OutOfRangeError:
                break
                print("End of dataset")
        print("Tensorflow Dataset:\nexecution time = ", timeit.default_timer() - start_time)

    data_reader = DataReader_train(data_list="test_data/selected_phases.csv", data_dir="test_data/data/")

    read(data_reader)

    data_reader = DataReader_train(format="hdf5", hdf5="test_data/data.h5", group="data")

    read(data_reader)

    data_reader = DataReader_test(data_list="test_data/selected_phases.csv", data_dir="test_data/data/")

    read(data_reader)

    data_reader = DataReader_test(format="hdf5", hdf5="test_data/data.h5", group="data")

    read(data_reader)

    data_reader = DataReader_pred(format="numpy", data_list="test_data/selected_phases.csv", data_dir="test_data/data/")

    read(data_reader)

    data_reader = DataReader_pred(
        format="mseed", data_list="test_data/mseed_station.csv", data_dir="test_data/waveforms/"
    )

    read(data_reader)

    data_reader = DataReader_pred(
        format="mseed", amplitude=True, data_list="test_data/mseed_station.csv", data_dir="test_data/waveforms/"
    )

    read(data_reader)

    data_reader = DataReader_mseed_array(
        data_list="test_data/mseed.csv",
        data_dir="test_data/waveforms/",
        stations="test_data/stations.csv",
        remove_resp=False,
    )

    read(data_reader, batch=None)


if __name__ == "__main__":

    test_DataReader()
