from __future__ import division
import os
import threading
import tensorflow as tf
import numpy as np
import pandas as pd
import logging
pd.options.mode.chained_assignment = None


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
    self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=self.config.X_shape)
    self.target_placeholder = tf.placeholder(dtype=tf.float32, shape=self.config.Y_shape)
    self.queue = tf.PaddingFIFOQueue(self.queue_size,
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

        if self.coord.should_stop():
          stop = True
          break

        sample = np.zeros(self.X_shape)
        sample[:, :, :] = np.copy(meta['data'])
        itp = meta['itp']
        its = meta['its']

        sample = self.normalize(sample)

        if (np.isnan(sample).any() or np.isinf(sample).any() or (not sample.any())):
          continue

        target = np.zeros(self.Y_shape)

        if (itp-self.mask_window//2 >= target.shape[0]) or (itp+self.mask_window//2 < 0):
          pass
        elif (itp-self.mask_window//2 >= 0) and (itp-self.mask_window//2 < target.shape[0]):
          target[itp-self.mask_window//2:itp+self.mask_window//2, 0, 1] = np.exp(-(np.arange(
                  itp-self.mask_window//2,itp+self.mask_window//2)-itp)**2/(2*(self.mask_window//4)**2))[:target.shape[0]-(itp-self.mask_window//2)]
        elif (itp-self.mask_window//2 < target.shape[0]):
          target[0:itp+self.mask_window//2, 0, 1] = np.exp(-(np.arange(
                  0,itp+self.mask_window//2)-itp)**2/(2*(self.mask_window//4)**2))[:target.shape[0]-(itp-self.mask_window//2)]
        if (its-self.mask_window//2 >= target.shape[0]) or (its+self.mask_window//2 < 0):
          pass
        elif (its-self.mask_window//2 >= 0) and (its-self.mask_window//2 < target.shape[0]):
          target[its-self.mask_window//2:its+self.mask_window//2, 0, 2] = np.exp(-(np.arange(
                  its-self.mask_window//2,its+self.mask_window//2)-its)**2/(2*(self.mask_window//4)**2))[:target.shape[0]-(its-self.mask_window//2)]
        elif (its-self.mask_window//2 < target.shape[0]):
          target[0:its+self.mask_window//2, 0, 2] = np.exp(-(np.arange(
                  0,its+self.mask_window//2)-its)**2/(2*(self.mask_window//4)**2))[:target.shape[0]-(its-self.mask_window//2)]
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


class DataReader_valid(DataReader):

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
      fname = os.path.join(self.data_dir, self.data_list.iloc[i]['fname'])
      try:
        if fname not in self.buffer:
          meta = np.load(fname)
          self.buffer[fname] = {'data': meta['data'], 'itp': meta['itp'], 'its': meta['its']}
        meta = self.buffer[fname]
      except:
        logging.error("Failed reading {}".format(fname))
        continue
      
      if self.coord.should_stop():
        break

      sample = np.zeros(self.X_shape)
      sample[:, :, :] = np.copy(meta['data'])
      itp = meta['itp']
      its = meta['its']

      sample = self.normalize(sample)

      if (np.isnan(sample).any() or np.isinf(sample).any() or (not sample.any())):
        continue

      target = np.zeros(self.Y_shape)

      if (itp-self.mask_window//2 >= target.shape[0]) or (itp+self.mask_window//2 < 0):
        pass
      elif (itp-self.mask_window//2 >= 0) and (itp-self.mask_window//2 < target.shape[0]):
        target[itp-self.mask_window//2:itp+self.mask_window//2, 0, 1] = np.exp(-(np.arange(
                itp-self.mask_window//2,itp+self.mask_window//2)-itp)**2/(2*(self.mask_window//4)**2))[:target.shape[0]-(itp-self.mask_window//2)]
      elif (itp-self.mask_window//2 < target.shape[0]):
        target[0:itp+self.mask_window//2, 0, 1] = np.exp(-(np.arange(
                0,itp+self.mask_window//2)-itp)**2/(2*(self.mask_window//4)**2))[:target.shape[0]-(itp-self.mask_window//2)]

      if (its-self.mask_window//2 >= target.shape[0]) or (its+self.mask_window//2 < 0):
        pass
      elif (its-self.mask_window//2 >= 0) and (its-self.mask_window//2 < target.shape[0]):
        target[its-self.mask_window//2:its+self.mask_window//2, 0, 2] = np.exp(-(np.arange(
                its-self.mask_window//2,its+self.mask_window//2)-its)**2/(2*(self.mask_window//4)**2))[:target.shape[0]-(its-self.mask_window//2)]
      elif (its-self.mask_window//2 < target.shape[0]):
        target[0:its+self.mask_window//2, 0, 2] = np.exp(-(np.arange(
                0,its+self.mask_window//2)-its)**2/(2*(self.mask_window//4)**2))[:target.shape[0]-(its-self.mask_window//2)]

      target[:, :, 0] = 1 - target[:, :, 1] - target[:, :, 2]

      sess.run(self.enqueue, feed_dict={self.sample_placeholder: sample,
                                        self.target_placeholder: target,
                                        self.fname_placeholder: fname,
                                        self.itp_placeholder: itp,
                                        self.its_placeholder: its})
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
    self.enqueue = self.queue.enqueue([self.sample_placeholder, self.fname_placeholder])

  def dequeue(self, num_elements):
    output = self.queue.dequeue_up_to(num_elements)
    return output

  def thread_main(self, sess, n_threads=1, start=0):
    index = list(range(start, self.num_data, n_threads))
    for i in index:
      fname = os.path.join(self.data_dir, self.data_list.iloc[i]['fname'])
      try:
        meta = np.load(fname)
      except:
        logging.error("Failed reading {}".format(fname))
        continue

      sample = meta['data'][:, np.newaxis, :]
      if np.array(sample.shape).all() != np.array(self.X_shape).all():
        logging.error("{}: shape {} is not same as input shape {}!".format(fname, sample.shape, self.X_shape))
        continue

      if np.isnan(sample).any() or np.isinf(sample).any():
        logging.warning("Data error: {}\nReplacing nan and inf with zeros".format(fname))
        sample[np.isnan(sample)] = 0
        sample[np.isinf(sample)] = 0

      sample = self.normalize(sample)
      sess.run(self.enqueue, feed_dict={self.sample_placeholder: sample,
                                        self.fname_placeholder: fname})


if __name__ == "__main__":
  pass

