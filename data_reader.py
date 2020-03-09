from __future__ import division
import os
import threading
import tensorflow as tf
import numpy as np
import pandas as pd
import logging
import scipy.interpolate
from tqdm import tqdm
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
               config=Config(),
               use_seed=False):
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
    self.use_seed = use_seed
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

  def stretch(self, data, itp, its, ratio=1):
    nt = data.shape[0]
    t = np.linspace(0, 1, nt)
    t_new = np.linspace(0, 1, int(round(nt*ratio)))
    f = scipy.interpolate.interp1d(t, data, axis=0)
    data_new = f(t_new)
    return data_new, int(round(itp*ratio)), int(round(its*ratio))

  def scale_amplitude(self, data):
    tmp = np.random.uniform(0, 1)
    if tmp < 0.2:
      data *= np.random.uniform(1, 3)
    elif tmp < 0.4:
      data /= np.random.uniform(1, 3)
    return data

  def drop_channel(self, data, prob=0.5):
    if np.random.uniform(0, 1) < prob:
      c1 = np.random.choice([0, 1])
      c2 = np.random.choice([0, 1])
      c3 = np.random.choice([0, 1])
      if c1 + c2 + c3 > 0:
        data[..., np.array([c1, c2, c3]) == 0] = 0
        # data *= 3/(c1+c2+c3)
      else:
        data[..., np.random.choice([0,1,2], size=2, replace=False)] = 0
        # data *= 3/(c1+c2+c3)
    return data

  def interplate(self, data, itp, its, start_tp, ratio=2, prob=0.5):
    if (np.random.uniform(0, 1) < prob) and ((its-itp)*ratio < self.X_shape[0]*0.8):
      nt = data.shape[0]
      t = np.linspace(0, 1, nt)
      f = scipy.interpolate.interp1d(t, data, axis=0)
      t_new = np.linspace(0, 1, round(nt*ratio))
      data_new = f(t_new)
      return data_new, round(itp*ratio), round(its*ratio), round(start_tp*ratio)
    else:
      return data, itp, its, start_tp

  def add_noise(self, data, channels, prob=0.3):
    if np.random.uniform(0, 1) < prob:
      # if channels not in self.buffer_channels:
      #   self.buffer_channels[channels] = self.data_list[self.data_list['channels']==channels]
      # fname = os.path.join(self.data_dir, self.buffer_channels[channels].sample(n=1).iloc[0]['fname'])
      fname = os.path.join(self.data_dir, self.data_list.sample(n=1).iloc[0]['fname'])
      try:
        if fname not in self.buffer:
          meta = np.load(fname)
          self.buffer[fname] = {'data': meta['data'], 'itp': meta['itp'], 'its': meta['its'], 
                                'snr': meta['snr'],  'channels': meta['channels']}
        meta = self.buffer[fname]
      except:
        logging.error("Failed reading {} in func add_noise".format(fname))
        return data
      # data += self.normalize(np.copy(meta['data'][:self.X_shape[0], np.newaxis, :])) * np.random.uniform(10, 100)
      data += self.normalize(np.copy(meta['data'][:self.X_shape[0], np.newaxis, :])) * np.random.chisquare(100)
    return data

  def adjust_missingchannels(self, data):
    tmp = np.max(np.abs(data), axis=0, keepdims=True)
    assert(tmp.shape[-1] == data.shape[-1])
    if np.count_nonzero(tmp) > 0:
      data *= data.shape[-1] / np.count_nonzero(tmp)
    return data

  def add_event(self, data, itp_list, its_list, channels, normalize=True, prob=0.5):
    # while np.random.uniform(0, 1) < prob:
    if np.random.uniform(0, 1) < prob:
      shift = None
      # if channels not in self.buffer_channels:
      #   self.buffer_channels[channels] = self.data_list[self.data_list['channels']==channels]
      # fname = os.path.join(self.data_dir, self.buffer_channels[channels].sample(n=1).iloc[0]['fname'])
      fname = os.path.join(self.data_dir, self.data_list.sample(n=1).iloc[0]['fname'])
      try:
        if fname not in self.buffer:
          meta = np.load(fname)
          self.buffer[fname] = {'data': meta['data'], 'itp': meta['itp'], 'its': meta['its'], 
                                'snr': meta['snr'], 'channels': meta['channels']}
        meta = self.buffer[fname]
      except:
        logging.error("Failed reading {} in func add_event".format(fname))
        # continue
        return data, itp_list, its_list

      start_tp = meta['itp'].tolist()
      itp = meta['itp'].tolist() - start_tp
      its = meta['its'].tolist() - start_tp

      if (max(its_list) - itp + self.mask_window + self.min_event_gap >= self.X_shape[0]-self.mask_window) \
         and (its - min(itp_list) + self.mask_window + self.min_event_gap >= min([its, self.X_shape[0]]) - self.mask_window):
        # continue
        return data, itp_list, its_list
      elif max(its_list) - itp + self.mask_window + self.min_event_gap >= self.X_shape[0]-self.mask_window:
        shift = np.random.randint(its - min(itp_list)+self.mask_window + self.min_event_gap, min([its, self.X_shape[0]])-self.mask_window)
      elif its - min(itp_list) + self.mask_window + self.min_event_gap >= min([its, self.X_shape[0]]) - self.mask_window:
        shift = -np.random.randint(max(its_list) - itp + self.mask_window + self.min_event_gap, self.X_shape[0] - self.mask_window)
      else:
        shift = np.random.choice([-np.random.randint(max(its_list) - itp + self.mask_window + self.min_event_gap, self.X_shape[0]-self.mask_window), 
                                  np.random.randint(its - min(itp_list)+self.mask_window + self.min_event_gap, min([its, self.X_shape[0]])-self.mask_window)])
      if normalize:
        ## chisquare
        # data += self.normalize(np.copy(meta['data'][start_tp+shift:start_tp+self.X_shape[0]+shift, np.newaxis, :])) * np.random.chisquare(1)
        ## uniform
        data += self.normalize(np.copy(meta['data'][start_tp+shift:start_tp+self.X_shape[0]+shift, np.newaxis, :])) * np.random.uniform(1, 10)
      else:
        data += np.copy(meta['data'][start_tp+shift:start_tp+self.X_shape[0]+shift, np.newaxis, :])

      itp_list.append(itp-shift)
      its_list.append(its-shift)
    return data, itp_list, its_list

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
            self.buffer[fname] = {'data': meta['data'], 'itp': meta['itp'], 'its': meta['its'], 
                                  'snr': meta['snr'], 'channels': meta['channels']}
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
        itp_list = []
        its_list = []
        if self.use_seed:
          np.random.seed(self.config.seed+i)
          
      ############### base case ###############
        data = np.copy(meta['data'])
        itp = meta['itp']
        its = meta['its']
        shift = np.random.randint(-(self.X_shape[0]-self.mask_window), min([its-start_tp, self.X_shape[0]])-self.mask_window)
        sample[:, :, :] = data[start_tp+shift:start_tp+self.X_shape[0]+shift, np.newaxis, :]
        itp_list.append(itp-start_tp-shift)
        its_list.append(its-start_tp-shift)
        sample = self.normalize(sample)

      ############### case 1: random shift  ###############
        # data = np.copy(meta['data'])
        # itp = meta['itp']
        # its = meta['its']
        # ## wrong shifting
        # # shift = np.random.randint(-1500, -1000)
        # shift = -1000
        # sample[:, :, :] = data[start_tp+shift:start_tp+self.X_shape[0]+shift, np.newaxis, :]
        # itp_list.append(itp-start_tp-shift)
        # its_list.append(its-start_tp-shift)
        # sample = self.normalize(sample)

      ############### case 2: stack events  ###############
        # data = np.copy(meta['data'])
        # itp = meta['itp']
        # its = meta['its']
        # shift = np.random.randint(-(self.X_shape[0]-self.mask_window), min([its-start_tp, self.X_shape[0]])-self.mask_window)
        # sample[:, :, :] = data[start_tp+shift:start_tp+self.X_shape[0]+shift, np.newaxis, :]
        # itp_list.append(itp-start_tp-shift)
        # its_list.append(its-start_tp-shift)
        # sample = self.normalize(sample)
        # sample, itp_list, its_list = self.add_event(sample, itp_list, its_list, channels, normalize=True, prob=0.5)
        # sample = self.normalize(sample)

      ############### case 3: stack noise   ###############
        # data = np.copy(meta['data'])
        # itp = meta['itp']
        # its = meta['its']
        # shift = np.random.randint(-(self.X_shape[0]-self.mask_window), min([its-start_tp, self.X_shape[0]])-self.mask_window)
        # sample[:, :, :] = data[start_tp+shift:start_tp+self.X_shape[0]+shift, np.newaxis, :]
        # itp_list = [itp-start_tp-shift]
        # its_list = [its-start_tp-shift]
        # sample = self.add_noise(sample, channels, prob=0.3)
        # sample = self.normalize(sample)

      ############### case 4: strech events  ###############
        # data = np.copy(meta['data'])
        # itp = meta['itp']
        # its = meta['its']
        # data, itp, its, start_tp = self.interplate(data, itp.tolist(), its.tolist(), start_tp, ratio=2, prob=0.5)
        # shift = np.random.randint(-(self.X_shape[0]-self.mask_window), min([its-start_tp, self.X_shape[0]])-self.mask_window)
        # sample[:, :, :] = data[start_tp+shift:start_tp+self.X_shape[0]+shift, np.newaxis, :]
        # itp_list = [itp-start_tp-shift]
        # its_list = [its-start_tp-shift]
        # sample = self.normalize(sample)

      ############### case 5: channel drop  ###############
        # data = np.copy(meta['data'])
        # itp = meta['itp']
        # its = meta['its']
        # shift = np.random.randint(-(self.X_shape[0]-self.mask_window), min([its-start_tp, self.X_shape[0]])-self.mask_window)
        # sample[:, :, :] = data[start_tp+shift:start_tp+self.X_shape[0]+shift, np.newaxis, :]
        # itp_list = [itp-start_tp-shift]
        # its_list = [its-start_tp-shift]
        # sample = self.normalize(sample)

        # if len(channels.split('_')) == 3:
        #   sample = self.drop_channel(sample, prob=0.3)
        # sample = self.adjust_missingchannels(sample)

      ############### case 6: pure noise  ###############
        # if np.random.uniform(0,1) >= 0.2: ## zero means no noise
        #   data = np.copy(meta['data'])
        #   itp = meta['itp']
        #   its = meta['its']
        #   shift = np.random.randint(-(self.X_shape[0]-self.mask_window), min([its-start_tp, self.X_shape[0]])-self.mask_window)
        #   sample[:, :, :] = data[start_tp+shift:start_tp+self.X_shape[0]+shift, np.newaxis, :]
        #   itp_list = [itp-start_tp-shift]
        #   its_list = [its-start_tp-shift]
        # else:
        #   sample[:, :, :] = np.copy(meta['data'][start_tp-self.X_shape[0]:start_tp, np.newaxis, :])
        #   if np.random.uniform(0, 1) > 0.5:
        #     sample[:np.random.randint(0, 2000), :, :] = 0
        #   itp_list = []
        #   its_list = []
        # sample = self.normalize(sample)


        ## common
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


class DataReader_valid(DataReader):

  def thread_main(self, sess, n_threads=1, start=0):
    stop = False
    while not stop:
      index = list(range(start, self.num_data, n_threads))
      for i in index:
        fname = os.path.join(self.data_dir, self.data_list.iloc[i]['fname'])
        try:
          if fname not in self.buffer:
            meta = np.load(fname)
            self.buffer[fname] = {'data': meta['data'], 'itp': meta['itp'], 'its': meta['its'], 
                                  'snr': meta['snr'], 'channels': meta['channels']}
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
        itp_list = []
        its_list = []
        np.random.seed(self.config.seed+i)

      ############### base case ###############
        data = np.copy(meta['data'])
        itp = meta['itp']
        its = meta['its']
        shift = np.random.randint(-(self.X_shape[0]-self.mask_window), min([its-start_tp, self.X_shape[0]])-self.mask_window)
        sample[:, :, :] = data[start_tp+shift:start_tp+self.X_shape[0]+shift, np.newaxis, :]
        itp_list.append(itp-start_tp-shift)
        its_list.append(its-start_tp-shift)
        sample = self.normalize(sample)


        ## common
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

  def __init__(self,
               data_dir,
               data_list,
               mask_window,
               queue_size,
               coord,
               config=Config()):
    self.config = config
    tmp_list = pd.read_csv(data_list, header=0)

    ## case 3: stack noise
    # tmp_list = tmp_list[(20*np.log10(tmp_list['snr'])>20)]
    # tmp_list = tmp_list[(20*np.log10(tmp_list['snr'])<20)]
     
    ## case 5: channel drop
    # tmp_list = tmp_list[(20*np.log10(tmp_list['snr'])>20)]

    ## case 6: test time augmentation
    # tmp_list = tmp_list[(tmp_list['its-itp'])>5*100]
    # tmp_list = tmp_list[(tmp_list['distance'])>40]

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
      itp_list = []
      its_list = []
      np.random.seed(self.config.seed+i)

      ############### base case  ###############
      shift = np.random.randint(-(self.X_shape[0]-self.mask_window), min([meta['its'].tolist()-start_tp, self.X_shape[0]])-self.mask_window)
      sample[:, :, :] = np.copy(meta['data'][start_tp+shift:start_tp+self.X_shape[0]+shift, np.newaxis, :])
      itp_list.append(meta['itp'].tolist()-start_tp-shift)
      its_list.append(meta['its'].tolist()-start_tp-shift)
      sample = self.normalize(sample)      

      ############### case 1: random shift  ###############
      # shift = np.random.randint(-(self.X_shape[0]-self.mask_window), min([meta['its'].tolist()-start_tp, self.X_shape[0]])-self.mask_window)
      # sample[:, :, :] = np.copy(meta['data'][start_tp+shift:start_tp+self.X_shape[0]+shift, np.newaxis, :])
      # itp_list.append(meta['itp'].tolist()-start_tp-shift)
      # its_list.append(meta['its'].tolist()-start_tp-shift)

      # for shift in tqdm(range(-3000, 1000, 20)):
      #   sample[:, :, :] = np.copy(meta['data'][start_tp+shift:start_tp+self.X_shape[0]+shift, np.newaxis, :])
      #   itp_list = [meta['itp'].tolist()-start_tp-shift]
      #   its_list = [meta['its'].tolist()-start_tp-shift]
      #   sample = self.normalize(sample)
      #   if (np.isnan(sample).any() or np.isinf(sample).any() or (not sample.any())):
      #     continue
      #   target = np.zeros(self.Y_shape)
      #   itp_true = []
      #   its_true = []
      #   for itp, its in zip(itp_list, its_list):
      #     if (itp >= target.shape[0]) or (itp < 0):
      #       pass
      #     elif (itp-self.mask_window//2 >= 0) and (itp-self.mask_window//2 < target.shape[0]):
      #       target[itp-self.mask_window//2:itp+self.mask_window//2, 0, 1] = \
      #           np.exp(-(np.arange(-self.mask_window//2,self.mask_window//2))**2/(2*(self.mask_window//4)**2))[:target.shape[0]-(itp-self.mask_window//2)]
      #       itp_true.append(itp)
      #     elif (itp-self.mask_window//2 < target.shape[0]):
      #       target[0:itp+self.mask_window//2, 0, 1] = \
      #           np.exp(-(np.arange(0,itp+self.mask_window//2)-itp)**2/(2*(self.mask_window//4)**2))[:target.shape[0]-(itp-self.mask_window//2)]
      #       itp_true.append(itp)

      #     if (its >= target.shape[0]) or (its < 0):
      #       pass
      #     elif (its-self.mask_window//2 >= 0) and (its-self.mask_window//2 < target.shape[0]):
      #       target[its-self.mask_window//2:its+self.mask_window//2, 0, 2] = \
      #           np.exp(-(np.arange(-self.mask_window//2,self.mask_window//2))**2/(2*(self.mask_window//4)**2))[:target.shape[0]-(its-self.mask_window//2)]
      #       its_true.append(its)
      #     elif (its-self.mask_window//2 < target.shape[0]):
      #       target[0:its+self.mask_window//2, 0, 2] = \
      #           np.exp(-(np.arange(0,its+self.mask_window//2)-its)**2/(2*(self.mask_window//4)**2))[:target.shape[0]-(its-self.mask_window//2)]
      #       its_true.append(its)
      #   target[:, :, 0] = 1 - target[:, :, 1] - target[:, :, 2]

      #   sess.run(self.enqueue, feed_dict={self.sample_placeholder: sample,
      #                                     self.target_placeholder: target,
      #                                     self.fname_placeholder: fname+f"_{abs(shift-1000):04d}",
      #                                     self.itp_placeholder: itp_true,
      #                                     self.its_placeholder: its_true})



      ############### case 2: stack event  ###############
      # # for shift in [-1000, -1500]:
      # for shift, ratio in zip([-500, -1500], [3, 1]):
      #   sample[:, :, :] += self.normalize(np.copy(meta['data'][start_tp+shift:start_tp+self.X_shape[0]+shift, np.newaxis, :])) * ratio#np.random.uniform(1, 10)
      #   itp_list.append(meta['itp'].tolist()-start_tp-shift)
      #   its_list.append(meta['its'].tolist()-start_tp-shift)
      # sample = self.normalize(sample)      

      ############### case 3: stack noise  ###############
      ############### case 4: time stretch ###############
      # shift = np.random.randint(-(self.X_shape[0]-self.mask_window), min([meta['its'].tolist()-start_tp, self.X_shape[0]])-self.mask_window)
      # sample[:, :, :] = np.copy(meta['data'][start_tp+shift:start_tp+self.X_shape[0]+shift, np.newaxis, :])
      # itp_list = [meta['itp'].tolist()-start_tp-shift]
      # its_list = [meta['its'].tolist()-start_tp-shift]
      # sample = self.normalize(sample)      

      ############### case 5: drop channel  ###############
      # shift = np.random.randint(-(self.X_shape[0]-self.mask_window), min([meta['its'].tolist()-start_tp, self.X_shape[0]])-self.mask_window)
      # sample[:, :, :] = np.copy(meta['data'][start_tp+shift:start_tp+self.X_shape[0]+shift, np.newaxis, :])
      # itp_list.append(meta['itp'].tolist()-start_tp-shift)
      # its_list.append(meta['its'].tolist()-start_tp-shift)
      # sample = self.normalize(sample)
 
      # sample[:, :, 0] = 0 ## no E
      # sample[:, :, 1] = 0 ## no N
      # sample[:, :, 2] = 0 ## no Z
      # # only E
      # sample[:, :, 1] = 0; sample[:, :, 2] = 0
      # # only N
      # sample[:, :, 0] = 0; sample[:, :, 2] = 0
      # # only Z
      # sample[:, :, 0] = 0; sample[:, :, 1] = 0
  
      # if len(channels.split('_')) == 3:
      #   sample = self.drop_channel(sample, prob=0.5)
      # sample = self.adjust_missingchannels(sample)

      ############### case 6: pure noise ###############
      # shift = -3000
      # sample[:, :, :] = np.copy(meta['data'][start_tp+shift:start_tp+self.X_shape[0]+shift, np.newaxis, :])
      # sample[:1000, :, :] = 0
      # dum_tp = meta['itp'].tolist()-start_tp-shift 
      # dum_ts = meta['its'].tolist()-start_tp-shift
      # if self.mask_window//2 < dum_tp < 3000 - self.mask_window//2:
      #   itp_list.append(dum_tp)
      # if self.mask_window//2 < dum_ts < 3000 - self.mask_window//2:
      #   its_list.append(dum_ts)
      # sample = self.normalize(sample)      


      ############### case 7: test time augmentation ###############
      # shift = 500
      # tmp_data = np.copy(meta['data'])
      # tmp_itp = meta['itp'].tolist()
      # tmp_its = meta['its'].tolist()
      # data, itp, its = self.stretch(tmp_data, tmp_itp, tmp_its, 1/2)
      # sample[:, :, :] = data[shift:self.X_shape[0]+shift, np.newaxis, :]
      # itp_list.append(itp-shift)
      # its_list.append(its-shift)
      # sample = self.normalize(sample)   

      #########
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
      if np.array(sample.shape).all() != np.array(self.X_shape).all():
        logging.error("{}: shape {} is not same as input shape {}!".format(fname, sample.shape, self.X_shape))
        continue

      if np.isnan(sample).any() or np.isinf(sample).any():
        logging.warning("Data error: {}\nReplacing nan and inf with zeros".format(fname))
        sample[np.isnan(sample)] = 0
        sample[np.isinf(sample)] = 0

      sample = self.normalize(sample)
      sample = self.adjust_missingchannels(sample)
      sess.run(self.enqueue, feed_dict={self.sample_placeholder: sample,
                                        self.fname_placeholder: fname})


if __name__ == "__main__":
  pass

