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
from dataclasses import dataclass
import multiprocessing

@dataclass
class Config:
  seed = 100
  use_seed = False
  n_channel = 3
  n_class = 3
  sampling_rate = 100.0
  dt = 1.0/sampling_rate
  X_shape = (3000, 1, n_channel)
  Y_shape = (3000, 1, n_class)
  min_event_gap = 3 * sampling_rate
  label_width = 6
  dtype="float32"


class DataReader():

  def __init__(self,
               data_dir,
               data_list,
               config = Config()):
    
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

class DataReader_test(DataReader):

  def add_placeholder(self):
    self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
    self.target_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
    self.fname_placeholder = tf.placeholder(dtype=tf.string, shape=None)
    self.itp_placeholder = tf.placeholder(dtype=tf.int32, shape=None)
    self.its_placeholder = tf.placeholder(dtype=tf.int32, shape=None)
    self.queue = tf.PaddingFIFOQueue(self.queue_size,
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
          self.buffer[fp] = {'data': meta['data'], 'itp': meta['itp'], 'its': meta['its'], 'channels': meta['channels']}
        meta = self.buffer[fp]
      except:
        logging.error("Failed reading {}".format(fp))
        continue

      channels = meta['channels'].tolist()
      start_tp = meta['itp'].tolist()
      
      if self.coord.should_stop():
        break

      sample = np.zeros(self.X_shape)

      np.random.seed(self.config.seed+i)
      shift = np.random.randint(-(self.X_shape[0]-self.label_width), min([meta['its'].tolist()-start_tp, self.X_shape[0]])-self.label_width)
      sample[:, :, :] = np.copy(meta['data'][start_tp+shift:start_tp+self.X_shape[0]+shift, np.newaxis, :])
      itp_list = [meta['itp'].tolist()-start_tp-shift]
      its_list = [meta['its'].tolist()-start_tp-shift]

      sample = self.normalize(sample)
      sample = self.adjust_missingchannels(sample)

      if (np.isnan(sample).any() or np.isinf(sample).any() or (not sample.any())):
        continue

      target = np.zeros(self.Y_shape)
      itp_true = []
      its_true = []
      for itp, its in zip(itp_list, its_list):
        if (itp >= target.shape[0]) or (itp < 0):
          pass
        elif (itp-self.label_width//2 >= 0) and (itp-self.label_width//2 < target.shape[0]):
          target[itp-self.label_width//2:itp+self.label_width//2, 0, 1] = \
              np.exp(-(np.arange(-self.label_width//2,self.label_width//2))**2/(2*(self.label_width//4)**2))[:target.shape[0]-(itp-self.label_width//2)]
          itp_true.append(itp)
        elif (itp-self.label_width//2 < target.shape[0]):
          target[0:itp+self.label_width//2, 0, 1] = \
              np.exp(-(np.arange(0,itp+self.label_width//2)-itp)**2/(2*(self.label_width//4)**2))[:target.shape[0]-(itp-self.label_width//2)]
          itp_true.append(itp)

        if (its >= target.shape[0]) or (its < 0):
          pass
        elif (its-self.label_width//2 >= 0) and (its-self.label_width//2 < target.shape[0]):
          target[its-self.label_width//2:its+self.label_width//2, 0, 2] = \
              np.exp(-(np.arange(-self.label_width//2,self.label_width//2))**2/(2*(self.label_width//4)**2))[:target.shape[0]-(its-self.label_width//2)]
          its_true.append(its)
        elif (its-self.label_width//2 < target.shape[0]):
          target[0:its+self.label_width//2, 0, 2] = \
              np.exp(-(np.arange(0,its+self.label_width//2)-its)**2/(2*(self.label_width//4)**2))[:target.shape[0]-(its-self.label_width//2)]
          its_true.append(its)
      target[:, :, 0] = 1 - target[:, :, 1] - target[:, :, 2]

      sess.run(self.enqueue, feed_dict={self.sample_placeholder: sample,
                                        self.target_placeholder: target,
                                        self.fname_placeholder: fname,
                                        self.itp_placeholder: itp_true,
                                        self.its_placeholder: its_true})
    return 0


class DataReader_pred(DataReader):

  def __init__(self,
               data_dir,
               data_list,
               queue_size,
               coord,
               input_length=None,
               config=Config()):
    self.config = config
    tmp_list = pd.read_csv(data_list, header=0)
    self.data_list = tmp_list
    self.num_data = len(self.data_list)
    self.data_dir = data_dir
    self.queue_size = queue_size
    self.X_shape = config.X_shape
    self.Y_shape = config.Y_shape
    if input_length is not None:
      logging.warning("Using input length: {}".format(input_length))
      self.X_shape[0] = input_length
      self.Y_shape[0] = input_length

    self.coord = coord
    self.threads = []
    self.add_placeholder()
  
  def add_placeholder(self):
    self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
    self.fname_placeholder = tf.placeholder(dtype=tf.string, shape=None)
    self.queue = tf.PaddingFIFOQueue(self.queue_size,
                                     ['float32', 'string'],
                                     shapes=[self.config.X_shape, []])

    self.enqueue = self.queue.enqueue([self.sample_placeholder, 
                                       self.fname_placeholder])

  def dequeue(self, num_elements):
    output = self.queue.dequeue_up_to(num_elements)
    return output

  def thread_main(self, sess, n_threads=1, start=0):
    index = list(range(start, self.num_data, n_threads))
    for i in index:
      fname = self.data_list.iloc[i]['fname']
      fp = os.path.join(self.data_dir, fname)
      try:
        meta = np.load(fp)
      except:
        logging.error("Failed reading {}".format(fname))
        continue
      shift = 0
      # sample = meta['data'][shift:shift+self.X_shape, np.newaxis, :]
      sample = meta['data'][:, np.newaxis, :]
      if not np.array_equal(np.array(sample.shape), np.array(self.X_shape)):
        logging.warning(f"Shape mismatch: {sample.shape} != {self.X_shape} in {fname}")
        tmp = np.zeros(self.X_shape)
        tmp[:sample.shape[0],0,:sample.shape[2]] = sample[:tmp.shape[0],0,:tmp.shape[2]]
        sample = tmp

      if np.isnan(sample).any() or np.isinf(sample).any():
        logging.warning(f"Data error: Nan or Inf found in {fname}")
        sample[np.isnan(sample)] = 0
        sample[np.isinf(sample)] = 0

      sample = self.normalize(sample)
      sample = self.adjust_missingchannels(sample)
      sess.run(self.enqueue, feed_dict={self.sample_placeholder: sample,
                                        self.fname_placeholder: fname})

class DataReader_hdf5(DataReader):

  def __init__(self,
               file,
               config=Config()):
    self.config = config
    self.num_data = len(self.data_list)
    self.X_shape = config.X_shape
  
  def add_placeholder(self):
    self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
    self.fname_placeholder = tf.placeholder(dtype=tf.string, shape=None)
    self.queue = tf.PaddingFIFOQueue(self.queue_size,
                                     ['float32', 'string'],
                                     shapes=[self.config.X_shape, []])

    self.enqueue = self.queue.enqueue([self.sample_placeholder, 
                                       self.fname_placeholder])

  def dequeue(self, num_elements):
    output = self.queue.dequeue_up_to(num_elements)
    return output

  def thread_main(self, sess, n_threads=1, start=0):
    index = list(range(start, self.num_data, n_threads))
    for i in index:
      fname = self.data_list.iloc[i]['fname']
      fp = os.path.join(self.data_dir, fname)
      try:
        meta = np.load(fp)
      except:
        logging.error("Failed reading {}".format(fname))
        continue
      shift = 0
      # sample = meta['data'][shift:shift+self.X_shape, np.newaxis, :]
      sample = meta['data'][:, np.newaxis, :]
      if not np.array_equal(np.array(sample.shape), np.array(self.X_shape)):
        logging.warning(f"Shape mismatch: {sample.shape} != {self.X_shape} in {fname}")
        tmp = np.zeros(self.X_shape)
        tmp[:sample.shape[0],0,:sample.shape[2]] = sample[:tmp.shape[0],0,:tmp.shape[2]]
        sample = tmp

      if np.isnan(sample).any() or np.isinf(sample).any():
        logging.warning(f"Data error: Nan or Inf found in {fname}")
        sample[np.isnan(sample)] = 0
        sample[np.isinf(sample)] = 0

      sample = self.normalize(sample)
      sample = self.adjust_missingchannels(sample)
      sess.run(self.enqueue, feed_dict={self.sample_placeholder: sample,
                                        self.fname_placeholder: fname})


class DataReader_mseed_v2(DataReader):

  def __init__(self,
               data_dir,
               data_list,
               batch_size = 32,
               config=Config()):
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
               config=Config()):
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


def test_DataReader():
  import time 

  data_reader = DataReader(
    data_dir="dataset/waveform_train",
    data_list="dataset/waveform.csv")

  def benchmark(dataset, num_batch=10):
    start_time = time.perf_counter()
    num = 0
    for sample in dataset:
      # time.sleep(0.5)
      num += 1
      if num > num_batch:
        break
    print("Execution time:", time.perf_counter() - start_time)

  dataset = generator(data_reader, 
                      output_types=(tf.float32, tf.float32),
                      output_shapes=(data_reader.X_shape, data_reader.Y_shape), 
                      num_parallel_calls=None)

  print("Base case:")
  benchmark(dataset)
  print("Prefetch:")
  benchmark(dataset.prefetch(tf.data.experimental.AUTOTUNE))
  dataset = generator(data_reader, 
                      output_types=(tf.float32, tf.float32),
                      output_shapes=(data_reader.X_shape, data_reader.Y_shape), 
                      num_parallel_calls=4)
  print("Parallel calls:")
  benchmark(dataset)
  benchmark(dataset.prefetch(tf.data.experimental.AUTOTUNE))

def test_DataReader_mseed_v2():
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


