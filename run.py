import numpy as np
import tensorflow as tf
import argparse
import os
import time
import logging
from unet import UNet
from data_reader import Config, DataReader, DataReader_valid, DataReader_pred
from util import *
from tqdm import tqdm
import multiprocessing
from functools import partial

def read_flags():
  """Returns flags"""

  parser = argparse.ArgumentParser()

  parser.add_argument("--mode",
                      default="train",
                      help="train/valid/test/debug")

  parser.add_argument("--epochs",
                      default=100,
                      type=int,
                      help="Number of epochs (default: 10)")

  parser.add_argument("--batch_size",
                      default=200,
                      type=int,
                      help="Batch size")

  parser.add_argument("--learning_rate",
                      default=0.01,
                      type=float,
                      help="learning rate")

  parser.add_argument("--decay_step",
                      default=-1,
                      type=int,
                      help="decay step")

  parser.add_argument("--decay_rate",
                      default=0.95,
                      type=float,
                      help="decay rate")

  parser.add_argument("--momentum",
                      default=0.9,
                      type=float,
                      help="momentum")

  parser.add_argument("--filters_root",
                      default=8,
                      type=int,
                      help="filters root")

  parser.add_argument("--depth",
                      default=5,
                      type=int,
                      help="depth")

  parser.add_argument("--kernel_size",
                      nargs="+",
                      type=int,
                      default=[7, 1],
                      help="kernel size")

  parser.add_argument("--pool_size",
                      nargs="+",
                      type=int,
                      default=[4, 1],
                      help="pool size")

  parser.add_argument("--drop_rate",
                      default=0,
                      type=float,
                      help="drop out rate")

  parser.add_argument("--dilation_rate",
                      nargs="+",
                      type=int,
                      default=[1, 1],
                      help="dilation_rate")

  parser.add_argument("--loss_type",
                      default="cross_entropy",
                      help="loss type: cross_entropy, IOU, mean_squared")

  parser.add_argument("--weight_decay",
                      default=0,
                      type=float,
                      help="weight decay")

  parser.add_argument("--optimizer",
                      default="adam",
                      help="optimizer: adam, momentum")

  parser.add_argument("--summary",
                      default=True,
                      type=bool,
                      help="summary")

  parser.add_argument("--class_weights",
                      nargs="+",
                      default=[1, 1, 1],
                      type=float,
                      help="class weights")

  parser.add_argument("--logdir",
                      default="log",
                      help="Tensorboard log directory (default: log)")

  parser.add_argument("--ckdir",
                      default=None,
                      help="Checkpoint directory (default: None)")

  parser.add_argument("--plot_number",
                      default=10,
                      type=int,
                      help="plotting trainning result")

  parser.add_argument("--fpred",
                      default="preds.npz",
                      help="ouput file name of test data")

  flags = parser.parse_args()
  return flags


def set_config(flags, data_reader):
  config = Config()

  config.X_shape = data_reader.X_shape
  config.n_channel = config.X_shape[-1]
  config.Y_shape = data_reader.Y_shape
  config.n_class = config.Y_shape[-1]

  config.depths = flags.depth
  config.filters_root = flags.filters_root
  config.kernel_size = flags.kernel_size
  config.pool_size = flags.pool_size
  config.dilation_rate = flags.dilation_rate
  config.batch_size = flags.batch_size
  config.class_weights = flags.class_weights
  config.loss_type = flags.loss_type
  config.weight_decay = flags.weight_decay
  config.optimizer = flags.optimizer

  config.learning_rate = flags.learning_rate
  if (flags.decay_step == -1) and (flags.mode == 'train'):
    config.decay_step = data_reader.num_data // flags.batch_size
  else:
    config.decay_step = flags.decay_step
  config.decay_rate = flags.decay_rate
  config.momentum = flags.momentum

  config.summary = flags.summary
  config.drop_rate = flags.drop_rate
  config.class_weights = flags.class_weights

  return config


def train_fn(flags, data_reader):
  current_time = time.strftime("%m%d%H%M%S")
  logging.info("Training log: {}".format(current_time))
  log_dir = os.path.join(flags.logdir, current_time)
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  fig_dir = os.path.join(log_dir, 'figures')
  if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

  config = set_config(flags, data_reader)
  with open(os.path.join(log_dir, 'config.log'), 'w') as fp:
    fp.write('\n'.join("%s: %s" % item for item in vars(config).items()))

  with tf.name_scope('Input_Batch'):
    batch = data_reader.dequeue(flags.batch_size)

  model = UNet(config, input_batch=batch)
  sess_config = tf.ConfigProto()
  sess_config.gpu_options.allow_growth = True
  sess_config.log_device_placement = False

  with tf.Session(config=sess_config) as sess:

    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
    init = tf.global_variables_initializer()
    sess.run(init)

    if flags.ckdir is not None:
      logging.info("restoring models...")
      latest_check_point = tf.train.latest_checkpoint(flags.ckdir)
      saver.restore(sess, latest_check_point)

    threads = data_reader.start_threads(sess, n_threads=20)
    flog = open(os.path.join(log_dir, 'loss.log'), 'w')
    total_step = 0
    mean_loss = 0
    for epoch in range(flags.epochs):
      progressbar = tqdm(range(0, data_reader.num_data, flags.batch_size), desc="epoch {}".format(epoch))
      for step in progressbar:
        # X_batch, Y_batch = sess.run(batch)
        # loss_batch, pred_batch, logits_batch = model.train_on_batch(
        #                       sess, X_batch, Y_batch, summary_writer, flags.drop_rate)
        loss_batch = model.train_on_batch(sess, summary_writer, flags.drop_rate)
        if epoch < 1:
          mean_loss = loss_batch
        else:
          total_step += 1
          mean_loss += (loss_batch-mean_loss)/total_step
        progressbar.set_description("{}: epoch {}, loss={:.6f}, mean={:.6f}".format(log_dir.split("/")[-1], epoch, loss_batch, mean_loss))
        flog.write("epoch: {}, step: {}, loss: {}, mean loss: {}\n".format(epoch, step//flags.batch_size, loss_batch, mean_loss))
        flog.flush()

      loss_batch, pred_batch, logits_batch, X_batch, Y_batch = model.train_on_batch(sess, summary_writer, flags.drop_rate, raw_data=True)
      plot_result(epoch, flags.plot_number, fig_dir,
                  pred_batch, X_batch, Y_batch)
      saver.save(sess, os.path.join(log_dir, "model_{}.ckpt".format(epoch)))
    flog.close()
    data_reader.coord.request_stop()
    for t in threads:
      t.join()
    sess.run(data_reader.queue.close())
  return 0

def valid_fn(flags, data_reader, fig_dir=None, result_dir=None, save_result=True):
  current_time = time.strftime("%m%d%H%M%S")
  logging.info("{} log: {}".format(flags.mode, current_time))
  log_dir = os.path.join(flags.logdir, flags.mode, current_time)
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  if fig_dir is None:
    fig_dir = os.path.join(log_dir, 'figures')
    if not os.path.exists(fig_dir):
      os.makedirs(fig_dir)
  if (save_result == True) and (result_dir is None):
    result_dir = os.path.join(log_dir, 'results')
    if not os.path.exists(result_dir):
      os.makedirs(result_dir)

  config = set_config(flags, data_reader)
  with open(os.path.join(log_dir, 'config.log'), 'w') as fp:
    fp.write('\n'.join("%s: %s" % item for item in vars(config).items()))

  with tf.name_scope('Input_Batch'):
    batch = data_reader.dequeue(flags.batch_size)

  model = UNet(config, input_batch=batch, mode='valid')
  sess_config = tf.ConfigProto()
  sess_config.gpu_options.allow_growth = True
  sess_config.log_device_placement = False

  with tf.Session(config=sess_config) as sess:

    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
    init = tf.global_variables_initializer()
    sess.run(init)

    logging.info("restoring models...")
    latest_check_point = tf.train.latest_checkpoint(flags.ckdir)
    saver.restore(sess, latest_check_point)
    
    threads = data_reader.start_threads(sess, n_threads=20)
    flog = open(os.path.join(log_dir, 'loss.log'), 'w')
    total_step = 0
    mean_loss = 0
    picks = []
    itp = []
    its = []
    progressbar = tqdm(range(0, data_reader.num_data-flags.batch_size, flags.batch_size), desc=flags.mode)
    pool = multiprocessing.Pool(multiprocessing.cpu_count()*2)
    for step in progressbar:

      loss_batch, pred_batch, X_batch, Y_batch, \
      fname_batch, itp_batch, its_batch = model.valid_on_batch(sess, summary_writer)
      total_step += 1
      mean_loss += (loss_batch-mean_loss)/total_step
      progressbar.set_description("{}, loss={:.6f}, mean loss={:6f}".format(flags.mode, loss_batch, mean_loss))
      flog.write("step: {}, loss: {}\n".format(step, loss_batch))
      flog.flush()

      itp_batch = clean_queue(itp_batch)
      its_batch = clean_queue(its_batch)
      picks_batch = pool.map(partial(postprocessing_thread,
                               pred = pred_batch,
                               X = X_batch,
                               Y = Y_batch,
                               itp = itp_batch,
                               its = its_batch,
                               fname = fname_batch,
                               result_dir = None,
                               fig_dir = None),
                       range(len(pred_batch)))
      picks.extend(picks_batch)
      itp.extend(itp_batch)
      its.extend(its_batch)

    ## final batch
    for t in threads:
      t.join()
    sess.run(data_reader.queue.close())
    loss_batch, pred_batch, X_batch, Y_batch, \
    fname_batch, itp_batch, its_batch = model.valid_on_batch(sess, summary_writer)

    itp_batch = clean_queue(itp_batch)
    its_batch = clean_queue(its_batch)
    picks_batch = pool.map(partial(postprocessing_thread,
                              pred = pred_batch,
                              X = X_batch,
                              Y = Y_batch,
                              itp = itp_batch,
                              its = its_batch,
                              fname = fname_batch,
                              result_dir = None,
                              fig_dir = None),
                      range(len(pred_batch)))
    picks.extend(picks_batch)
    itp.extend(itp_batch)
    its.extend(its_batch)
    pool.close()

    metrics_p, metrics_s = calculate_metrics(picks, itp, its, tol=0.1)
    flog.write("P-phase: Precision={}, Recall={}, F1={}\n".format(metrics_p[0], metrics_p[1], metrics_p[2]))
    flog.write("S-phase: Precision={}, Recall={}, F1={}\n".format(metrics_s[0], metrics_s[1], metrics_s[2]))
    flog.close()

  return 0

def pred_fn(flags, data_reader, fig_dir=None, result_dir=None, save_result=True):
  current_time = time.strftime("%m%d%H%M%S")
  log_dir = os.path.join(flags.logdir, "pred", current_time)
  logging.info("Pred log: %s" % log_dir)
  logging.info("Dataset size: {}".format(data_reader.num_data))
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  if fig_dir is None:
    fig_dir = os.path.join(log_dir, 'figures')
    if not os.path.exists(fig_dir):
      os.makedirs(fig_dir)
  if (save_result == True) and (result_dir is None):
    result_dir = os.path.join(log_dir, 'results')
    if not os.path.exists(result_dir):
      os.makedirs(result_dir)

  config = set_config(flags, data_reader)
  with open(os.path.join(log_dir, 'config.log'), 'w') as fp:
    fp.write('\n'.join("%s: %s" % item for item in vars(config).items()))

  with tf.name_scope('Input_Batch'):
    batch = data_reader.dequeue(flags.batch_size)

  model = UNet(config, batch, "pred")
  sess_config = tf.ConfigProto()
  sess_config.gpu_options.allow_growth = True
  sess_config.log_device_placement = False

  with tf.Session(config=sess_config) as sess:

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
    init = tf.global_variables_initializer()
    sess.run(init)

    logging.info("restoring models...")
    latest_check_point = tf.train.latest_checkpoint(flags.ckdir)
    saver.restore(sess, latest_check_point)

    threads = data_reader.start_threads(sess, n_threads=20)
    picks = []
    fname = []
    pool = multiprocessing.Pool(multiprocessing.cpu_count()*2)
    for step in tqdm(range(0, data_reader.num_data-flags.batch_size, flags.batch_size), desc="Pred"):
      pred_batch, X_batch, fname_batch = sess.run([model.preds, batch[0], batch[1]], 
                                                   feed_dict={model.drop_rate: 0,
                                                              model.is_training: False})
      
      picks_batch = pool.map(partial(postprocessing_thread,
                                      pred = pred_batch,
                                      X = X_batch,
                                      fname = fname_batch,
                                      result_dir = result_dir,
                                      fig_dir = fig_dir),
                              range(len(pred_batch)))
      picks.extend(picks_batch)
      fname.extend(fname_batch)

    for t in threads:
      t.join()
    sess.run(data_reader.queue.close())
    pred_batch, X_batch, fname_batch = sess.run([model.preds, batch[0], batch[1]],
                                                  feed_dict={model.drop_rate: 0,
                                                            model.is_training: False})
    picks_batch = pool.map(partial(postprocessing_thread,
                              pred = pred_batch,
                              X = X_batch,
                              fname = fname_batch,
                              result_dir = result_dir,
                              fig_dir = fig_dir),
                      range(len(pred_batch)))
    picks.extend(picks_batch)
    fname.extend(fname_batch)

    if save_result:
      np.savez(os.path.join(log_dir, "picks.npz"), picks=picks, fname=fname)

  return 0


def main(flags):

  logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
  coord = tf.train.Coordinator()

  if flags.mode == "train":
    with tf.name_scope('create_inputs'):
      data_reader = DataReader(
          # data_dir="../Dataset/NPZ_PS/HNE_HNN_HNZ/",
          # data_list="../Dataset/NPZ_PS/HNE_HNN_HNZ.csv",
          data_dir="../Dataset/NPZ_PS/",
          data_list="../Dataset/NPZ_PS/selected_channels_train.csv",
          mask_window=0.4,
          queue_size=flags.batch_size*3,
          coord=coord)
    train_fn(flags, data_reader)
  
  elif flags.mode == "valid" or flags.mode == "test":
    with tf.name_scope('create_inputs'):
      data_reader = DataReader_valid(
          data_dir="../Dataset2018/NPZ_PS/HNE_HNN_HNZ/",
          data_list="../Dataset2018/NPZ_PS/HNE_HNN_HNZ.csv",
          mask_window=0.4,
          queue_size=flags.batch_size*3,
          coord=coord)
    valid_fn(flags, data_reader)

  elif flags.mode == "debug":
    with tf.name_scope('create_inputs'):
      data_reader = DataReader(
          data_dir="../Dataset/NPZ_PS/",
          data_list="../Dataset/NPZ_PS/selected_channels_train.csv",
          mask_window=0.4,
          queue_size=flags.batch_size*3,
          coord=coord)
    valid_fn(flags, data_reader)

  elif flags.mode == "pred":
    with tf.name_scope('create_inputs'):
      data_reader = DataReader_pred(
          # data_dir="../Dataset2018/NPZ_PS/EHE_EHN_EHZ/",
          # data_list="../Dataset2018/NPZ_PS/EHE_EHN_EHZ.csv",
          data_dir="../Demo/PhaseNet/",
          data_list="../Demo/PhaseNet.csv",
          queue_size=flags.batch_size*3,
          coord=coord)
    pred_fn(flags, data_reader)

  else:
    print("mode should be: train, valid, test, pred or debug")

  coord.request_stop()
  coord.join()

  return


if __name__ == '__main__':
  flags = read_flags()
  main(flags)
