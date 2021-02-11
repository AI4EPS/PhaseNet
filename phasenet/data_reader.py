import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import os, threading, logging
import numpy as np
import pandas as pd
import scipy.interpolate
pd.options.mode.chained_assignment = None
import obspy
from tqdm import tqdm
from scipy.interpolate import interp1d
import s3fs


def py_func_decorator(output_types=None, output_shapes=None, name=None):
    def decorator(func):
        def call(*args, **kwargs):
            nonlocal output_shapes
            # flat_output_types = nest.flatten(output_types)
            flat_output_types = tf.nest.flatten(output_types)
            # flat_values = tf.py_func(
            flat_values = tf.numpy_function(
                func, 
                inp=args, 
                Tout=flat_output_types,
                name=name
            )
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
        dataset = dataset.shuffle(len(iterator))
    @py_func_decorator(output_types, output_shapes, name=name)
    def index_to_entry(idx):
        return iterator[idx]    
    return dataset.map(index_to_entry, num_parallel_calls=num_parallel_calls)


def normalize(data, window=3000):
    """
    data: nt, nch
    """
    shift = window//2
    nt, nch = data.shape

    ## std in slide windows
    data_pad = np.pad(data, ((window//2, window//2), (0,0)), mode="reflect")
    t = np.arange(0, nt, shift, dtype="int")
    std = np.zeros([len(t)+1, nch])
    mean = np.zeros([len(t)+1, nch])
    for i in range(1, len(std)):
        std[i, :] = np.std(data_pad[i*shift:i*shift+window, :], axis=0)
        mean[i, :] = np.mean(data_pad[i*shift:i*shift+window, :], axis=0)
        
    t = np.append(t, nt)
    # std[-1, :] = np.std(data_pad[-window:, :], axis=0)
    # mean[-1, :] = np.mean(data_pad[-window:, :], axis=0)
    std[-1, :], mean[-1, :] = std[-2, :], mean[-2, :]
    std[0, :], mean[0, :] = std[1, :], mean[1, :]
    std[std == 0] = 1

    ## normalize data with interplated std 
    t_interp = np.arange(nt, dtype="int")
    std_interp = interp1d(t, std, axis=0, kind="slinear")(t_interp)
    mean_interp = interp1d(t, mean, axis=0, kind="slinear")(t_interp)
    data = (data - mean_interp)/std_interp

    return data


def normalize_batch(data, window=3000):
    """
    data: nsta, nt, nch
    """
    shift = window//2
    nsta, nt, nch = data.shape
    
    ## std in slide windows
    data_pad = np.pad(data, ((0,0), (window//2, window//2), (0,0)), mode="reflect")
    t = np.arange(0, nt, shift, dtype="int")
    std = np.zeros([nsta, len(t)+1, nch])
    mean = np.zeros([nsta, len(t)+1, nch])
    for i in range(1, len(t)):
        std[:, i, :] = np.std(data_pad[:, i*shift:i*shift+window, :], axis=1)
        mean[:, i, :] = np.mean(data_pad[:, i*shift:i*shift+window, :], axis=1)
    
    t = np.append(t, nt)
    # std[:, -1, :] = np.std(data_pad[:, -window:, :], axis=1)
    # mean[:, -1, :] = np.mean(data_pad[:, -window:, :], axis=1)
    std[:, -1, :], mean[:, -1, :] = std[:, -2, :], mean[:, -2, :]
    std[:, 0, :], mean[:, 0, :] = std[:, 1, :], mean[:, 1, :]
    std[std == 0] = 1
    
    # ## normalize data with interplated std 
    t_interp = np.arange(nt, dtype="int")
    std_interp = interp1d(t, std, axis=1, kind="slinear")(t_interp)
    mean_interp = interp1d(t, mean, axis=1, kind="slinear")(t_interp)
    data = (data - mean_interp)/std_interp
    
    return data

class DataConfig():

    seed = 100
    use_seed = True
    n_channel = 3
    n_class = 3
    sampling_rate = 100
    dt = 1.0/sampling_rate
    X_shape = [3000, 1, n_channel]
    Y_shape = [3000, 1, n_class]
    min_event_gap = 3 * sampling_rate
    dtype = "float32"

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class DataReader(object):
    def __init__(self,
                 data_dir,
                 data_list,
                 mask_window,
                 queue_size,
                 coord,
                 config=DataConfig()):
        self.config = config
        tmp_list = pd.read_csv(data_list, header=0)
        self.data_list = tmp_list
        self.num_data = len(self.data_list)
        self.data_dir = data_dir
        self.queue_size = queue_size
        self.n_channel = config.n_channel
        self.n_class = config.n_class
        self.X_shape = config.X_shape
        self.Y_shape = config.Y_shape
        self.dt = config.dt
        self.min_event_gap = config.min_event_gap
        self.mask_window = int(mask_window * config.sampling_rate)
        self.coord = coord
        self.threads = []
        self.buffer = {}
        self.buffer_channels = {}
        self.add_placeholder()
    
    def add_placeholder(self):
        self.sample_placeholder = tf.compat.v1.placeholder(dtype=tf.float32, shape=self.config.X_shape)
        self.target_placeholder = tf.compat.v1.placeholder(dtype=tf.float32, shape=self.config.Y_shape)
        self.queue = tf.queue.PaddingFIFOQueue(self.queue_size,
                                               ['float32', 'float32'],
                                               shapes=[self.config.X_shape, self.config.Y_shape])
        self.enqueue = self.queue.enqueue([self.sample_placeholder, self.target_placeholder])

    def dequeue(self, num_elements):
        output = self.queue.dequeue_many(num_elements)
        return output

    def normalize(self, data):
        data -= np.mean(data, axis=0, keepdims=True)
        std_data = np.std(data, axis=0, keepdims=True)
        assert(std_data.shape[-1] == data.shape[-1])
        std_data[std_data == 0] = 1
        data /= std_data
        return data


    def generate_target(self, itp_list, its_list):

        target = np.zeros(self.Y_shape)
        itp_true = []
        its_true = []
        for itp, its in zip(itp_list, its_list):
            if (itp >= target.shape[0]) or (itp < 0):
                pass
            elif (itp-self.mask_window//2 >= 0) and (itp-self.mask_window//2 < target.shape[0]):
                target[itp-self.mask_window//2:itp+self.mask_window//2, 0, 1] = \
                        np.exp(-(np.arange(-self.mask_window//2,self.mask_window//2))**2/(2*(self.mask_window//4)**2))[:target.shape[0]-(itp-self.mask_window//2)]
                itp_true.append(itp)
            elif (itp-self.mask_window//2 < target.shape[0]):
                target[0:itp+self.mask_window//2, 0, 1] = \
                            np.exp(-(np.arange(0,itp+self.mask_window//2)-itp)**2/(2*(self.mask_window//4)**2))[:target.shape[0]-(itp-self.mask_window//2)]
                itp_true.append(itp)

            if (its >= target.shape[0]) or (its < 0):
                pass
            elif (its-self.mask_window//2 >= 0) and (its-self.mask_window//2 < target.shape[0]):
                target[its-self.mask_window//2:its+self.mask_window//2, 0, 2] = \
                        np.exp(-(np.arange(-self.mask_window//2,self.mask_window//2))**2/(2*(self.mask_window//4)**2))[:target.shape[0]-(its-self.mask_window//2)]
                its_true.append(its)
            elif (its-self.mask_window//2 < target.shape[0]):
                target[0:its+self.mask_window//2, 0, 2] = \
                        np.exp(-(np.arange(0,its+self.mask_window//2)-its)**2/(2*(self.mask_window//4)**2))[:target.shape[0]-(its-self.mask_window//2)]
                its_true.append(its)

        target[:, :, 0] = 1 - target[:, :, 1] - target[:, :, 2]
        return target, itp_true, its_true

    def thread_main(self, sess, n_threads=1, start=0):
        stop = False
        while not stop:
            index = list(range(start, self.num_data, n_threads))
            np.random.shuffle(index)
            for i in index:
                fname = os.path.join(self.data_dir, self.data_list.iloc[i]['fname'])
                try:
                    if fname not in self.buffer:
                        meta = np.load(fname)
                        self.buffer[fname] = {'data': meta['data'], 'itp': meta['itp'], 'its': meta['its']}
                    meta = self.buffer[fname]
                except:
                    logging.error("Failed reading {}".format(fname))
                    continue

                start_tp = meta['itp'].tolist()

                if self.coord.should_stop():
                    stop = True
                    break

                sample = np.zeros(self.X_shape)
                if np.random.random() < 0.95:
                    data = np.copy(meta['data'])
                    itp = meta['itp']
                    its = meta['its']
                    start_tp = itp

                    shift = np.random.randint(-(self.X_shape[0]-self.mask_window), min([its-start_tp, self.X_shape[0]])-self.mask_window)
                    sample[:, :, :] = data[start_tp+shift:start_tp+self.X_shape[0]+shift, np.newaxis, :]
                    itp_list = [itp-start_tp-shift]
                    its_list = [its-start_tp-shift]
                else:
                    sample[:, :, :] = np.copy(meta['data'][start_tp-self.X_shape[0]:start_tp, np.newaxis, :])
                    itp_list = []
                    its_list = []

                sample = self.normalize(sample)

                if (np.isnan(sample).any() or np.isinf(sample).any() or (not sample.any())):
                    continue

                target, _, _ = self.generate_target(itp_list, its_list)

                sess.run(self.enqueue, feed_dict={self.sample_placeholder: sample, self.target_placeholder: target})
        return 0

    def start_threads(self, sess, n_threads=8):
        for i in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess, n_threads, i))
            thread.daemon = True
            thread.start()
            self.threads.append(thread)
        return self.threads

class DataReader_test(DataReader):

    def add_placeholder(self):
        self.sample_placeholder = tf.compat.v1.placeholder(dtype=tf.float32, shape=None)
        self.target_placeholder = tf.compat.v1.placeholder(dtype=tf.float32, shape=None)
        self.fname_placeholder = tf.compat.v1.placeholder(dtype=tf.string, shape=None)
        self.itp_placeholder = tf.compat.v1.placeholder(dtype=tf.int32, shape=None)
        self.its_placeholder = tf.compat.v1.placeholder(dtype=tf.int32, shape=None)
        self.queue = tf.queue.PaddingFIFOQueue(self.queue_size,
                                               ['float32', 'float32', 'string', 'int32', 'int32'],
                                               shapes=[self.config.X_shape, self.config.Y_shape, [], [None], [None]])
        self.enqueue = self.queue.enqueue([self.sample_placeholder, self.target_placeholder, 
                                                                             self.fname_placeholder, 
                                                                             self.itp_placeholder, self.its_placeholder])

    def dequeue(self, num_elements):
        output = self.queue.dequeue_up_to(num_elements)
        return output

    def thread_main(self, sess, n_threads=1, start=0):

        index = list(range(start, self.num_data, n_threads))
        for i in index:
            fname = self.data_list.iloc[i]['fname']
            fp = os.path.join(self.data_dir, fname)
            try:
                if fp not in self.buffer:
                    meta = np.load(fp)
                    self.buffer[fp] = {'data': meta['data'], 'itp': meta['itp'], 'its': meta['its']}
                meta = self.buffer[fp]
            except:
                logging.error("Failed reading {}".format(fp))
                continue

            start_tp = meta['itp'].tolist()
            
            if self.coord.should_stop():
                break

            sample = np.zeros(self.X_shape)

            np.random.seed(self.config.seed+i)
            shift = np.random.randint(-(self.X_shape[0]-self.mask_window), min([meta['its'].tolist()-start_tp, self.X_shape[0]])-self.mask_window)
            sample[:, :, :] = np.copy(meta['data'][start_tp+shift:start_tp+self.X_shape[0]+shift, np.newaxis, :])
            itp_list = [meta['itp'].tolist()-start_tp-shift]
            its_list = [meta['its'].tolist()-start_tp-shift]

            sample = self.normalize(sample)

            if (np.isnan(sample).any() or np.isinf(sample).any() or (not sample.any())):
                continue

            target, itp_true, its_true = self.generate_target(itp_list, its_list)

            sess.run(self.enqueue, feed_dict={self.sample_placeholder: sample,
                                              self.target_placeholder: target,
                                              self.fname_placeholder: fname,
                                              self.itp_placeholder: itp_true,
                                              self.its_placeholder: its_true})

        return 0


class DataReader_pred(DataReader):

    def __init__(self,
                data_dir,
                data_list):

        tmp_list = pd.read_csv(data_list, header=0)
        self.data_list = tmp_list
        self.num_data = len(self.data_list)
        self.data_dir = data_dir
        self.dtype = "float32"
        self.X_shape = self.get_data_shape()

    def get_data_shape(self):
        fname = self.data_list.iloc[0]['fname']
        fp = os.path.join(self.data_dir, fname)
        data = np.load(fp)["data"][:,np.newaxis,:]
        return data.shape

    def adjust_missingchannels(self, data):
        tmp = np.max(np.abs(data), axis=0, keepdims=True)
        assert(tmp.shape[-1] == data.shape[-1])
        if np.count_nonzero(tmp) > 0:
            data *= data.shape[-1] / np.count_nonzero(tmp)
        return data

    def __len__(self):
        return self.num_data

    def __getitem__(self, i):
        fname = self.data_list.iloc[i]['fname']
        fp = os.path.join(self.data_dir, fname)
        
        try:
            meta = np.load(fp)
        except:
            logging.error("Failed reading {}".format(fname))
            return (np.zeros(self.X_shape, dtype=self.dtype), fname)

        shift = 0
        sample = normalize(meta['data'])[:, np.newaxis, :].astype(self.dtype)

        if not np.array_equal(np.array(sample.shape), np.array(self.X_shape)):
            logging.warning(f"Shape mismatch: {sample.shape} != {self.X_shape} in {fname}")
        tmp = np.zeros(self.X_shape, dtype=self.dtype)
        tmp[:sample.shape[0],0,:sample.shape[2]] = sample[:tmp.shape[0],0,:tmp.shape[2]]
        sample = tmp

        if np.isnan(sample).any() or np.isinf(sample).any():
            logging.warning(f"Data error: Nan or Inf found in {fname}")

        sample[np.isnan(sample)] = 0
        sample[np.isinf(sample)] = 0

        # sample = self.normalize(sample)
        sample = self.adjust_missingchannels(sample)

        return (sample, fname)

    def dataset(self, num_parallel=2):
        dataset = dataset_map(self, 
                              output_types=("float32", "string"),
                              output_shapes=(self.X_shape, None), 
                              num_parallel_calls=num_parallel)
        return dataset

class DataReader_mseed(DataReader):

    def __init__(self, data_list, stations, data_dir="", amplitude=False):

        self.data_list = pd.read_csv(data_list, header=0)
        self.num_data = len(self.data_list)
        self.data_dir = data_dir
        self.stations = pd.read_csv(stations, delimiter="\t")
        self.dtype = "float32"
        self.amplitude = amplitude
        self.X_shape = self.get_data_shape()
        
    def get_data_shape(self):
        if self.amplitude:
            (data, _, _) = self[0]
        else:
            (data, _) = self[0]
        return data.shape

    def read_mseed(self, fp):
        meta = obspy.read(fp)
        meta = meta.detrend("spline", order=2, dspline=5*meta[0].stats.sampling_rate)
        meta = meta.merge(fill_value=0)
        meta = meta.trim(min([st.stats.starttime for st in meta]), 
                         max([st.stats.endtime for st in meta]), 
                         pad=True, fill_value=0)

        if meta[0].stats.sampling_rate != 100:
            logging.warning(f"Sampling rate {meta[0].stats.sampling_rate} != 100 Hz")

        # order = ['3','2','1','E','N','Z']
        # order = {key: i for i, key in enumerate(order)}
        nsta = len(self.stations)
        nt = len(meta[0].data)
        data = np.zeros([nsta, nt, 3], dtype=self.dtype)
        if self.amplitude:
            raw = np.zeros([nsta, nt, 3], dtype=self.dtype)
        for i in range(nsta):
            sta = self.stations.iloc[i]["station"]
            comp = self.stations.iloc[i]["component"].split(",")
            resp = self.stations.iloc[i]["response"].split(",")
            for j in range(len(comp)):
                data[i, :, j] = meta.select(id=sta+comp[j])[0].data.astype(self.dtype) / float(resp[j])
                if self.amplitude:
                    # raw[i, :, j] = meta.select(id=sta+comp[j])[0].integrate().data.astype(self.dtype) / float(resp[j])
                    raw[i, :, j] = meta.select(id=sta+comp[j])[0].data.astype(self.dtype) / float(resp[j])
        if self.amplitude:
            return data, raw
        else:
            return data

    def __len__(self):
        return self.num_data

    def __getitem__(self, i):
        
        fname_base = self.data_list.iloc[i]['fname']
        fname = [fname_base.split('/')[-1].rstrip(".mseed")+"."+self.stations.iloc[i]["station"] for i in range(len(self.stations))]
        fp = os.path.join(self.data_dir, fname_base)
        try:
            if self.amplitude:
                data, raw = self.read_mseed(fp)
            else:
                data = self.read_mseed(fp)
        except Exception as e:
            logging.error(f"Failed reading {fname}: {e}")
            return (np.zeros(self.X_shape, dtype=self.dtype), fname)
        
        sample = normalize_batch(data)[:,:,np.newaxis,:]

        if self.amplitude:
            return (sample.astype(self.dtype), fname, raw[:,:,np.newaxis,:])
        else:
            return (sample.astype(self.dtype), fname)

    def dataset(self, num_parallel=2):
        if self.amplitude:
            dataset = dataset_map(self, 
                                output_types=("float32", "string", "float32"),
                                output_shapes=(self.X_shape, None, self.X_shape), 
                                num_parallel_calls=num_parallel)
        else:
            dataset = dataset_map(self, 
                                output_types=("float32", "string"),
                                output_shapes=(self.X_shape, None), 
                                num_parallel_calls=num_parallel)
        return dataset


class DataReader_s3(DataReader_mseed):

    def __init__(self, data_list, stations, s3_url, bucket="waveforms", amplitude=False):

        self.data_list = pd.read_csv(data_list, header=0)
        self.num_data = len(self.data_list)
        self.stations = pd.read_csv(stations, delimiter="\t")
        self.s3_url = s3_url
        self.bucket = bucket
        self.dtype = "float32"
        self.amplitude = amplitude
        self.X_shape = self.get_data_shape()

    def __getitem__(self, i):
        
        fname = self.data_list.iloc[i]['fname']
        fs = s3fs.S3FileSystem(anon=False, key="quakeflow", secret="quakeflow", use_ssl=False, client_kwargs={'endpoint_url': self.s3_url})

        try:
            with fs.open(self.bucket+"/"+fname, 'rb') as fp:
                data = self.read_mseed(fp)
        except Exception as e:
            logging.error(f"Failed reading {fname}: {e}")
            return (np.zeros(self.X_shape, dtype=self.dtype), fname)
        
        sample = normalize_batch(data)[:,:,np.newaxis,:]

        return (sample.astype(self.dtype), fname)


    def dataset(self, num_parallel=2):
        dataset = dataset_map(self, 
                              output_types=("float32", "string"),
                              output_shapes=(self.X_shape, None), 
                              num_parallel_calls=num_parallel)
        return dataset

# def test_DataReader_mseed(data_list, data_dir, stations):
#     print("test_DataReader_mseed:")
#     import timeit
#     data_reader = DataReader_mseed(
#         data_list = data_list,
#         data_dir = data_dir,
#         stations = stations)
    
#     start_time = timeit.default_timer()
#     dataset = dataset_map(data_reader, 
#                           output_types=("float32", "string"),
#                           output_shapes=(data_reader.X_shape, None), 
#                           num_parallel_calls=2)
#     dataset = dataset
#     # dataset = tf.data.Dataset.range(data_reader.num_data)
#     # dataset = dataset.interleave(lambda x: tf.data.Dataset.from_generator(data_reader.generator, 
#     #                                                                       output_types=(data_reader.dtype, "string"), 
#     #                                                                       output_shapes=(data_reader.X_shape, None), 
#     #                                                                       args=(x,)),
#     #                              cycle_length=data_reader.num_data,
#     #                              block_length=1,
#     #                              num_parallel_calls=data_reader.num_data)

#     sess = tf.compat.v1.Session()
    
#     print(len(data_reader))
#     print("-------", tf.data.Dataset.cardinality(dataset))
#     num = 0
#     # for x in tf.compat.v1.data.make_one_shot_iterator(dataset).get_next():
#     #     print(num)
#     #     num += 1
#     #     dum = sess.run(x)
#         # print(dum)
#     x = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
#     while True:
#         num+=1
#         print(num)
#         dum = sess.run(x)
    
#     print("Tensorflow Dataset:\nexecution time = ", timeit.default_timer() - start_time)


if __name__ == "__main__":
    pass
    ## debug
    # data_reader = DataReader_mseed(
    #   data_dir="/data/beroza/zhuwq/Project-PhaseNet-mseed/mseed/",
    #   data_list="/data/beroza/zhuwq/Project-PhaseNet-mseed/fname.txt",
    #   queue_size=20,
    #   coord=None)
    # data_reader.thread_main(None, n_threads=1, start=0)
    # pred_fn(args, data_reader, log_dir=args.output_dir)

    # test_DataReader_mseed()
    # from glob import glob
    # import os
    # import pandas as pd
    # mseed_dir = "../../quakeflow/mseed"
    # mseed_list = glob(os.path.join(mseed_dir, "*.mseed"), recursive=True)
    # print(f"Number of mseed: {len(mseed_list)}")

    # stations = pd.read_csv("../../quakeflow/stations.csv", delimiter="\t", index_col="station")
    # test_DataReader_mseed("../../quakeflow/mseed.lst", "../../quakeflow/", "../../quakeflow/stations.csv")
