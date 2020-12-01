from __future__ import division
import os
import threading
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import pandas as pd
import logging
import scipy.interpolate
pd.options.mode.chained_assignment = None
import obspy
from tqdm import tqdm


class Config():
  seed = 100
  use_seed = False
  n_channel = 3
  n_class = 3
  num_repeat_noise = 1
  sampling_rate = 100
  dt = 1.0/sampling_rate
  X_shape = [3000, 1, n_channel]
  Y_shape = [3000, 1, n_class]
  min_event_gap = 3 * sampling_rate


class DataReader(object):

  def __init__(self,
               data_dir,
               data_list,
               mask_window,
               queue_size,
               coord,
               config=Config()):
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

  def adjust_missingchannels(self, data):
    tmp = np.max(np.abs(data), axis=0, keepdims=True)
    assert(tmp.shape[-1] == data.shape[-1])
    if np.count_nonzero(tmp) > 0:
      data *= data.shape[-1] / np.count_nonzero(tmp)
    return data

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
            self.buffer[fname] = {'data': meta['data'], 'itp': meta['itp'], 'its': meta['its'], 'channels': meta['channels']}
          meta = self.buffer[fname]
        except:
          logging.error("Failed reading {}".format(fname))
          continue

        channels = meta['channels'].tolist()
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
        sample = self.adjust_missingchannels(sample)

        if (np.isnan(sample).any() or np.isinf(sample).any() or (not sample.any())):
          continue

        target = np.zeros(self.Y_shape)
        for itp, its in zip(itp_list, its_list):
          if (itp >= target.shape[0]) or (itp < 0):
            pass
          elif (itp-self.mask_window//2 >= 0) and (itp-self.mask_window//2 < target.shape[0]):
            target[itp-self.mask_window//2:itp+self.mask_window//2, 0, 1] = \
                np.exp(-(np.arange(-self.mask_window//2,self.mask_window//2))**2/(2*(self.mask_window//4)**2))[:target.shape[0]-(itp-self.mask_window//2)]
          elif (itp-self.mask_window//2 < target.shape[0]):
            target[0:itp+self.mask_window//2, 0, 1] = \
                 np.exp(-(np.arange(0,itp+self.mask_window//2)-itp)**2/(2*(self.mask_window//4)**2))[:target.shape[0]-(itp-self.mask_window//2)]
          if (its >= target.shape[0]) or (its < 0):
            pass
          elif (its-self.mask_window//2 >= 0) and (its-self.mask_window//2 < target.shape[0]):
            target[its-self.mask_window//2:its+self.mask_window//2, 0, 2] = \
                np.exp(-(np.arange(-self.mask_window//2,self.mask_window//2))**2/(2*(self.mask_window//4)**2))[:target.shape[0]-(its-self.mask_window//2)]
          elif (its-self.mask_window//2 < target.shape[0]):
            target[0:its+self.mask_window//2, 0, 2] = \
                np.exp(-(np.arange(0,its+self.mask_window//2)-its)**2/(2*(self.mask_window//4)**2))[:target.shape[0]-(its-self.mask_window//2)]
        target[:, :, 0] = 1 - target[:, :, 1] - target[:, :, 2]

        sess.run(self.enqueue, feed_dict={self.sample_placeholder: sample,
                                          self.target_placeholder: target})
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
      shift = np.random.randint(-(self.X_shape[0]-self.mask_window), min([meta['its'].tolist()-start_tp, self.X_shape[0]])-self.mask_window)
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


class DataReader_mseed(DataReader):

  def __init__(self,
               data_dir,
               data_list,
               queue_size,
               coord,
               input_length=3000,
               config=Config()):
    self.config = config
    tmp_list = pd.read_csv(data_list, header=0)
    self.data_list = tmp_list
    self.num_data = len(self.data_list)
    self.data_dir = data_dir
    self.queue_size = queue_size
    self.X_shape = config.X_shape
    self.Y_shape = config.Y_shape
    self.input_length = config.X_shape[0]
    if input_length is not None:
      logging.warning("Using input length: {}".format(input_length))
      self.X_shape[0] = input_length
      self.Y_shape[0] = input_length
      self.input_length = input_length

    self.coord = coord
    self.threads = []
    self.add_placeholder()
  
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


  ## mseed preprocessing here 
  def read_mseed(self, fp, channels):

    meta = obspy.read(fp)
    meta = meta.detrend('constant')
    meta = meta.merge(fill_value=0)
    meta = meta.trim(min([st.stats.starttime for st in meta]), 
                     max([st.stats.endtime for st in meta]), 
                     pad=True, fill_value=0)
    nt = len(meta[0].data)

    ## can test small sampling rate for longer distance
    # meta = meta.interpolate(sampling_rate=100)

    data = [[] for ch in channels]
    for i, ch in enumerate(channels):
      tmp = meta.select(channel=ch)
      if len(tmp) == 1:
        data[i] = tmp[0].data
      elif len(tmp) == 0:
        print(f"Warning: Missing channel \"{ch}\" in {meta}")
        data[i] = np.zeros(nt)
      else:
        print(f"Error in {tmp}")
    data = np.vstack(data)
  
    pad_width = int((np.ceil((data.shape[1] - 1) / self.input_length))*self.input_length - data.shape[1])
    if pad_width == -1:
      data = data[:,:-1]
    else:
      data = np.pad(data, ((0,0), (0, pad_width)), 'constant', constant_values=(0,0))
    
    data = np.hstack([data, np.zeros_like(data[:,:self.input_length//2]), data[:,:-self.input_length//2]])
    data = np.reshape(data, (3, -1, self.input_length))
    data = data.transpose(1,2,0)[:,:,np.newaxis,:]

    return data


  def thread_main(self, sess, n_threads=1, start=0):
    index = list(range(start, self.num_data, n_threads))
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
      for i in tqdm(range(meta.shape[0]), desc=f"{fp}"):
        sample = meta[i]
        sample = self.normalize(sample)
        sample = self.adjust_missingchannels(sample)
        sess.run(self.enqueue, feed_dict={self.sample_placeholder: sample,
                                          self.fname_placeholder: f"{fname}_{i*self.input_length}"})

if __name__ == "__main__":
  ## debug
  data_reader = DataReader_mseed(
    data_dir="/data/beroza/zhuwq/Project-PhaseNet-mseed/mseed/",
    data_list="/data/beroza/zhuwq/Project-PhaseNet-mseed/fname.txt",
    queue_size=20,
    coord=None)
  data_reader.thread_main(None, n_threads=1, start=0)
# pred_fn(args, data_reader, log_dir=args.output_dir)

