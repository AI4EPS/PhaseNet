from __future__ import division
import os
import tensorflow as tf
import numpy as np
import logging
import scipy.interpolate
import pandas as pd
pd.options.mode.chained_assignment = None
import obspy
from tqdm import tqdm
from util import py_func_decorator, generator
import multiprocessing
import h5py

class DataReader():

  def __init__(self,
               data_dir,
               data_list,
               config):
    
    self.config = config
    tmp_list = pd.read_csv(data_list, header=0)
    self.data_list = tmp_list
    self.num_data = len(self.data_list)
    self.data_dir = data_dir
    self.n_channel = config.n_channel
    self.n_class = config.n_class
    self.X_shape = config.X_shape
    self.Y_shape = config.Y_shape
    self.min_event_gap = config.min_event_gap
    self.label_width = config.label_width
    self.dtype = config.dtype
    self.buffer = {}
    self.buffer_channels = {}

  def normalize(self, data):
    data -= np.mean(data, axis=0, keepdims=True)
    std_data = np.std(data, axis=0, keepdims=True)
    assert(std_data.shape[-1] == data.shape[-1])
    std_data[std_data == 0] = 1
    data /= std_data
    return data

  def adjust_missingchannels(self, data):
    tmp = np.max(np.abs(data), axis=0, keepdims=True)
    assert(tmp.shape[-1] == data.shape[-1])
    if np.count_nonzero(tmp) > 0:
      data *= data.shape[-1] / np.count_nonzero(tmp)
    return data

  def generate_label(self, itp_list, its_list):
    target = np.zeros(self.Y_shape, dtype=self.dtype)
    for itp, its in zip(itp_list, its_list):
      if (itp >= target.shape[0]) or (itp < 0):
        pass
      elif (itp-self.label_width//2 >= 0) and (itp-self.label_width//2 < target.shape[0]):
        target[itp-self.label_width//2:itp+self.label_width//2, 0, 1] = \
            np.exp(-(np.arange(-self.label_width//2,self.label_width//2))**2/(2*(self.label_width//4)**2))[:target.shape[0]-(itp-self.label_width//2)]
      elif (itp-self.label_width//2 < target.shape[0]):
        target[0:itp+self.label_width//2, 0, 1] = \
              np.exp(-(np.arange(0,itp+self.label_width//2)-itp)**2/(2*(self.label_width//4)**2))[:target.shape[0]-(itp-self.label_width//2)]
      if (its >= target.shape[0]) or (its < 0):
        pass
      elif (its-self.label_width//2 >= 0) and (its-self.label_width//2 < target.shape[0]):
        target[its-self.label_width//2:its+self.label_width//2, 0, 2] = \
            np.exp(-(np.arange(-self.label_width//2,self.label_width//2))**2/(2*(self.label_width//4)**2))[:target.shape[0]-(its-self.label_width//2)]
      elif (its-self.label_width//2 < target.shape[0]):
        target[0:its+self.label_width//2, 0, 2] = \
            np.exp(-(np.arange(0,its+self.label_width//2)-its)**2/(2*(self.label_width//4)**2))[:target.shape[0]-(its-self.label_width//2)]
    target[:, :, 0] = 1 - target[:, :, 1] - target[:, :, 2]
    return target

  def __len__(self):
    return self.num_data

  def __getitem__(self, i):
    fname = os.path.join(self.data_dir, self.data_list.iloc[i]['fname'])
    try:
      if fname not in self.buffer:
        meta = np.load(fname)
        self.buffer[fname] = {'data': meta['data'].astype(self.dtype), 'itp': meta['itp'], 'its': meta['its'], 'channels': meta['channels']}
      meta = self.buffer[fname]
    except:
      logging.error("Failed reading {}".format(fname))
      return (np.zeros(self.Y_shape, dtype=self.dtype), np.zeros(self.X_shape, dtype=self.dtype))

    channels = meta['channels'].tolist()
    start_tp = meta['itp'].tolist()

    sample = np.zeros(self.X_shape, dtype=self.dtype)
    if np.random.random() < 0.95:
      data = np.copy(meta['data'])
      itp = meta['itp']
      its = meta['its']
      start_tp = itp

      shift = np.random.randint(-(self.X_shape[0]-self.label_width), min([its-start_tp, self.X_shape[0]])-self.label_width)
      sample[:, :, :] = data[start_tp+shift:start_tp+self.X_shape[0]+shift, np.newaxis, :]
      itp_list = [itp-start_tp-shift]
      its_list = [its-start_tp-shift]
    else:
      sample[:, :, :] = np.copy(meta['data'])[start_tp-self.X_shape[0]:start_tp, np.newaxis, :]
      itp_list = []
      its_list = []

    sample = self.normalize(sample)
    sample = self.adjust_missingchannels(sample)

    if (np.isnan(sample).any() or np.isinf(sample).any() or (not sample.any())):
      return (np.zeros(self.Y_shape, dtype=self.dtype), np.zeros(self.X_shape, dtype=self.dtype))

    target = self.generate_label(itp_list, its_list)
    # time.sleep(0.5)
    return (sample, target)

class DataReader_valid(DataReader):

  def __getitem__(self, i):
    fname = os.path.join(self.data_dir, self.data_list.iloc[i]['fname'])
    try:
      if fname not in self.buffer:
        meta = np.load(fname)
        self.buffer[fname] = {'data': meta['data'].astype(self.dtype), 'itp': meta['itp'].astype("int32"), 'its': meta['its'].astype("int32"), 'channels': meta['channels']}
      meta = self.buffer[fname]
    except:
      logging.error("Failed reading {}".format(fname))
      return (np.zeros(self.Y_shape, dtype=self.dtype), np.zeros(self.X_shape, dtype=self.dtype))

    channels = meta['channels'].tolist()
    start_tp = meta['itp'].tolist()

    sample = np.zeros(self.X_shape, dtype=self.dtype)
    if np.random.random() < 0.95:
      data = np.copy(meta['data'])
      itp = meta['itp']
      its = meta['its']
      start_tp = itp

      shift = np.random.randint(-(self.X_shape[0]-self.label_width), min([its-start_tp, self.X_shape[0]])-self.label_width, dtype="int32")
      sample[:, :, :] = data[start_tp+shift:start_tp+self.X_shape[0]+shift, np.newaxis, :]
      itp_list = [itp-start_tp-shift]
      its_list = [its-start_tp-shift]
    else:
      sample[:, :, :] = np.copy(meta['data'])[start_tp-self.X_shape[0]:start_tp, np.newaxis, :]
      itp_list = []
      its_list = []

    sample = self.normalize(sample)
    sample = self.adjust_missingchannels(sample)

    if (np.isnan(sample).any() or np.isinf(sample).any() or (not sample.any())):
      return (np.zeros(self.Y_shape, dtype=self.dtype), np.zeros(self.X_shape, dtype=self.dtype))

    target = self.generate_label(itp_list, its_list)

    itp_list = np.array(itp_list, dtype="int32")
    its_list = np.array(its_list, dtype="int32")
    return (sample, target, itp_list, its_list)


class DataReader_pred(DataReader):

  def __init__(self,
               data_dir,
               data_list,
               config,
               input_length=9001):
    self.config = config
    tmp_list = pd.read_csv(data_list, header=0)
    self.data_list = tmp_list
    self.num_data = len(self.data_list)
    self.data_dir = data_dir
    self.X_shape = config.X_shape
    if input_length != config.X_shape[0]:
      logging.warning("Using input length: {}".format(input_length))
      self.X_shape = (input_length, config.X_shape[1], config.X_shape[2])
    self.dtype = config.dtype

  def __len__(self):
    return self.num_data

  def __getitem__(self, i):
    fname = self.data_list.iloc[i]['fname']
    fp = os.path.join(self.data_dir, fname)
    try:
      meta = np.load(fp)
    except:
      logging.error("Failed reading {}".format(fname))
      return (np.zeros(self.Y_shape, dtype=self.dtype), np.zeros(self.X_shape, dtype=self.dtype))

    shift = 0
    # sample = meta['data'][shift:shift+self.X_shape, np.newaxis, :]
    sample = meta['data'][:, np.newaxis, :].astype(self.dtype)
    if not np.array_equal(np.array(sample.shape), np.array(self.X_shape)):
      logging.warning(f"Shape mismatch: {sample.shape} != {self.X_shape} in {fname}")
      tmp = np.zeros(self.X_shape, dtype=self.dtype)
      tmp[:sample.shape[0],0,:sample.shape[2]] = sample[:tmp.shape[0],0,:tmp.shape[2]]
      sample = tmp

    if np.isnan(sample).any() or np.isinf(sample).any():
      logging.warning(f"Data error: Nan or Inf found in {fname}")
      sample[np.isnan(sample)] = 0
      sample[np.isinf(sample)] = 0

    sample = self.normalize(sample)
    sample = self.adjust_missingchannels(sample)
    return (sample, fname)


class DataReader_hdf5(DataReader):

  def __init__(self,
               hdf5,
               group,
               config):
    self.config = config
    self.h5_data = h5py.File(hdf5, 'r', libver='latest', swmr=True)[group]
    self.data_list = list(self.h5_data.keys())
    self.num_data = len(self.data_list)
    self.X_shape = config.X_shape
    self.Y_shape = config.Y_shape
    self.dtype = config.dtype

  def __len__(self):
    return self.num_data

  def __getitem__(self, i):
    fname = self.data_list[i]
    sample = self.h5_data[fname].value
    sample = sample[:,np.newaxis,:]

    if not np.array_equal(np.array(sample.shape), np.array(self.X_shape)):
      logging.warning(f"Shape mismatch: {sample.shape} != {self.X_shape} in {fname}")
      tmp = np.zeros(self.X_shape, dtype=self.dtype)
      tmp[:sample.shape[0],0,:sample.shape[2]] = sample[:tmp.shape[0],0,:tmp.shape[2]]
      sample = tmp

    if np.isnan(sample).any() or np.isinf(sample).any():
      logging.warning(f"Data error: Nan or Inf found in {fname}")
      sample[np.isnan(sample)] = 0
      sample[np.isinf(sample)] = 0

    sample = self.normalize(sample)
    sample = self.adjust_missingchannels(sample)
    return (sample, fname)

  def __call__(self, batch_size):
    dataset = generator(self, 
                    output_types=("float32", "string"),
                    output_shapes=(self.X_shape, None), 
                    num_parallel_calls=2)
    dataset = dataset.batch(batch_size).prefetch(batch_size*2)
    return dataset

class DataReader_mseed_v2(DataReader):

  def __init__(self,
               data_dir,
               data_list,
               batch_size,
               config):
    self.config = config
    tmp_list = pd.read_csv(data_list, header=0)
    self.data_list = tmp_list
    self.num_data = len(self.data_list)
    self.data_dir = data_dir
    self.X_shape = config.X_shape
    self.Y_shape = config.Y_shape
    self.dtype = config.dtype
    self.queue = multiprocessing.Queue()
    self.processes = []
    self.batch_size = batch_size

  ## mseed preprocessing here 
  def read_mseed(self, fp, channels):
    meta = obspy.read(fp)
    meta = meta.detrend("spline", order=2, dspline=5*meta[0].stats.sampling_rate)
    meta = meta.merge(fill_value=0)
    meta = meta.trim(min([st.stats.starttime for st in meta]), 
                     max([st.stats.endtime for st in meta]), 
                     pad=True, fill_value=0)
    nt = len(meta[0].data)
    ## if needed
    # meta = meta.interpolate(sampling_rate=100)
    data = [[] for ch in channels]
    for i, ch in enumerate(channels):
      tmp = meta.select(channel=ch)
      if len(tmp) == 1:
        data[i] = tmp[0].data.astype(self.dtype)
      elif len(tmp) == 0:
        logging.warning(f"Warning: Missing channel \"{ch}\" in {meta}")
        data[i] = np.zeros(nt, dtype=self.dtype)
      else:
        print(f"Error in {tmp}")
    data = np.vstack(data)
    data = data[:,:data.shape[-1] - (data.shape[-1] % self.X_shape[0])]
    data = np.hstack([data, np.zeros_like(data[:,:self.X_shape[0]//2]), data[:,:-self.X_shape[0]//2]])
    data = np.reshape(data, (3, -1, self.X_shape[0]))
    data = data.transpose(1,2,0)[:,:,np.newaxis,:]
    return data

  def _main(self, n_proc=1, start=0):
    index = list(range(start, self.num_data, n_proc))
    for i in index:
      fname = self.data_list.iloc[i]['fname']
      fp = os.path.join(self.data_dir, fname)
      E = self.data_list.iloc[i]['E']
      N = self.data_list.iloc[i]['N']
      Z = self.data_list.iloc[i]['Z']
      try:
        meta = self.read_mseed(fp, [E, N, Z])
      except Exception as e:
        logging.error("Failed reading {}".format(fname))
        print(e)
        continue
      for i in tqdm(range(0, meta.shape[0], self.batch_size), desc=f"{fp}"):
        sample = meta[i:i+self.batch_size]
        sample = self.normalize(sample)
        sample = self.adjust_missingchannels(sample)
        self.queue.put((sample, f"{fname}_{i*self.X_shape[0]}"))

  def start_processes(self, n_proc=2):
    for i in range(n_proc):
      process = multiprocessing.Process(target=self._main, args=(n_proc, i))
      process.daemon = True
      process.start()
      self.processes.append(process)
    return self.processes


class DataReader_mseed(DataReader):

  def __init__(self,
               data_dir,
               data_list,
               config):
    self.config = config
    tmp_list = pd.read_csv(data_list, header=0)
    self.data_list = tmp_list
    self.num_data = len(self.data_list)
    self.data_dir = data_dir
    self.X_shape = config.X_shape
    self.Y_shape = config.Y_shape
    self.dtype = config.dtype

  ## mseed preprocessing here 
  def read_mseed(self, fp, channels):
    meta = obspy.read(fp)
    meta = meta.detrend("spline", order=2, dspline=5*meta[0].stats.sampling_rate)
    meta = meta.merge(fill_value=0)
    meta = meta.trim(min([st.stats.starttime for st in meta]), 
                     max([st.stats.endtime for st in meta]), 
                     pad=True, fill_value=0)
    nt = len(meta[0].data)
    ## if needed
    # meta = meta.interpolate(sampling_rate=100)
    data = [[] for ch in channels]
    for i, ch in enumerate(channels):
      tmp = meta.select(channel=ch)
      if len(tmp) == 1:
        data[i] = tmp[0].data.astype(self.dtype)
      elif len(tmp) == 0:
        logging.warning(f"Warning: Missing channel \"{ch}\" in {meta}")
        data[i] = np.zeros(nt, dtype=self.dtype)
      else:
        print(f"Error in {tmp}")
    data = np.vstack(data)
    data = data[:,:data.shape[-1] - (data.shape[-1] % self.X_shape[0])]
    data = np.hstack([data, np.zeros_like(data[:,:self.X_shape[0]//2]), data[:,:-self.X_shape[0]//2]])
    data = np.reshape(data, (3, -1, self.X_shape[0]))
    data = data.transpose(1,2,0)[:,:,np.newaxis,:]
    return data

  def generator(self, i):
    fname = self.data_list.iloc[i]['fname']
    fp = os.path.join(self.data_dir, fname)
    E = self.data_list.iloc[i]['E']
    N = self.data_list.iloc[i]['N']
    Z = self.data_list.iloc[i]['Z']
    try:
      meta = self.read_mseed(fp, [E, N, Z])
    except Exception as e:
      logging.error("Failed reading {}".format(fname))
      print(e)
    for i in tqdm(range(meta.shape[0]), desc=f"{fp}"):
      sample = meta[i]
      sample = self.normalize(sample)
      sample = self.adjust_missingchannels(sample)
      yield (sample, f"{fname}_{i*self.X_shape[0]}")


def benchmark(dataset, num_batch=10):
  import time
  start_time = time.perf_counter()
  num = 0
  for sample in dataset:
    # time.sleep(0.5)
    num += 1
    if num > num_batch:
      break
  print("Execution time:", time.perf_counter() - start_time)


def test_DataReader():
  print("test_DataReader:")

  data_reader = DataReader(
    data_dir="dataset/waveform_train",
    data_list="dataset/waveform.csv")

  dataset = generator(data_reader, 
                      output_types=("float32", "float32",),
                      output_shapes=(data_reader.X_shape, data_reader.Y_shape), 
                      num_parallel_calls=None)

  print("Base case:")
  benchmark(dataset)
  print("Prefetch:")
  benchmark(dataset.prefetch(tf.data.experimental.AUTOTUNE))
  dataset = generator(data_reader, 
                      output_types=("float32", "float32",),
                      output_shapes=(data_reader.X_shape, data_reader.Y_shape), 
                      num_parallel_calls=4)
  print("Parallel calls:")
  benchmark(dataset)
  benchmark(dataset.prefetch(tf.data.experimental.AUTOTUNE))


def test_DataReader_valid():
  print("test_DataReader_valid:")

  data_reader = DataReader_valid(
    data_dir="dataset/waveform_train",
    data_list="dataset/waveform.csv")

  dataset = generator(data_reader, 
                      output_types=("float32", "float32", "int32", "int32"),
                      output_shapes=(data_reader.X_shape, data_reader.Y_shape, None, None), 
                      num_parallel_calls=None)
  
  print("Base case:")
  benchmark(dataset)
  print("Prefetch:")
  benchmark(dataset.prefetch(tf.data.experimental.AUTOTUNE))
  dataset = generator(data_reader, 
                      output_types=("float32", "float32", "int32", "int32"),
                      output_shapes=(data_reader.X_shape, data_reader.Y_shape, None, None), 
                      num_parallel_calls=4)
  print("Parallel calls:")
  benchmark(dataset)
  benchmark(dataset.prefetch(tf.data.experimental.AUTOTUNE))


def test_DataReader_pred():
  print("test_DataReader_pred:")

  data_reader = DataReader_pred(
    data_dir="dataset/waveform_train",
    data_list="dataset/waveform.csv")

  dataset = generator(data_reader, 
                      output_types=("float32", "string"),
                      output_shapes=(data_reader.X_shape, None), 
                      num_parallel_calls=None)

  print("Base case:")
  benchmark(dataset)
  print("Prefetch:")
  benchmark(dataset.prefetch(tf.data.experimental.AUTOTUNE))
  dataset = generator(data_reader, 
                      output_types=("float32", "string"),
                      output_shapes=(data_reader.X_shape, None), 
                      num_parallel_calls=4)
  print("Parallel calls:")
  benchmark(dataset)
  benchmark(dataset.prefetch(tf.data.experimental.AUTOTUNE))


def test_DataReader_hdf5():
  print("test_DataReader_hdf5:")
  data_reader = DataReader_hdf5(hdf5="dataset/data.hdf5", 
                                data_dir="data")

  dataset = generator(data_reader, 
                      output_types=("float32", "string"),
                      output_shapes=(data_reader.X_shape, None), 
                      num_parallel_calls=None)

  print("Base case:")
  benchmark(dataset)
  print("Prefetch:")
  benchmark(dataset.prefetch(tf.data.experimental.AUTOTUNE))
  dataset = generator(data_reader, 
                      output_types=("float32", "string"),
                      output_shapes=(data_reader.X_shape, None), 
                      num_parallel_calls=4)
  print("Parallel calls:")
  benchmark(dataset)
  benchmark(dataset.prefetch(tf.data.experimental.AUTOTUNE))


def test_DataReader_mseed_v2():
  print("test_DataReader_mseed_v2:")
  import timeit
  data_reader = DataReader_mseed_v2(
    data_list = "demo/fname.csv",
    data_dir = "demo/mseed/")
  data_reader.start_processes(2)
  start_time = timeit.default_timer()
  data = data_reader.queue.get(block=True, timeout=None)

  while True:
    try:
      data = data_reader.queue.get(block=True, timeout=10)
    except Exception as err:
      print(err)
      break

  print("Multiprocessing:\nexecution time = ", timeit.default_timer() - start_time)
  return data


def test_DataReader_mseed():
  print("test_DataReader_mseed:")
  import timeit
  data_reader = DataReader_mseed(
    data_list = "demo/fname.csv",
    data_dir = "demo/mseed/")
  
  start_time = timeit.default_timer()
  dataset = tf.data.Dataset.range(data_reader.num_data)
  dataset = dataset.interleave(lambda x: tf.data.Dataset.from_generator(data_reader.generator, 
                                                 output_types=(data_reader.dtype, "string"), 
                                                 output_shapes=(data_reader.X_shape, None), 
                                                 args=(x,)),
                    cycle_length=data_reader.num_data,
                    block_length=1,
                    num_parallel_calls=data_reader.num_data)
  dataset = dataset.batch(24)
  for x in dataset:
    # print(x[1])
    pass
  
  print("Tensorflow Dataset:\nexecution time = ", timeit.default_timer() - start_time)


if __name__ == "__main__":
  pass