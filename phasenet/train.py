import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import argparse, os, time, logging
from tqdm import tqdm
import pandas as pd
import threading
import multiprocessing
from functools import partial
import pickle
from model import UNet, ModelConfig
from data_reader import DataReader, DataReader_test
from util import *

def read_args():

  parser = argparse.ArgumentParser()
  parser.add_argument("--mode", default="train", help="train/train_valid/test/debug")
  parser.add_argument("--epochs", default=100, type=int, help="number of epochs (default: 10)")
  parser.add_argument("--batch_size", default=20, type=int, help="batch size")
  parser.add_argument("--learning_rate", default=0.01, type=float, help="learning rate")
  parser.add_argument("--decay_step", default=-1, type=int, help="decay step")
  parser.add_argument("--decay_rate", default=0.9, type=float, help="decay rate")
  parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
  parser.add_argument("--filters_root", default=8, type=int, help="filters root")
  parser.add_argument("--depth", default=5, type=int, help="depth")
  parser.add_argument("--kernel_size", nargs="+", type=int, default=[7, 1], help="kernel size")
  parser.add_argument("--pool_size", nargs="+", type=int, default=[4, 1], help="pool size")
  parser.add_argument("--drop_rate", default=0, type=float, help="drop out rate")
  parser.add_argument("--dilation_rate", nargs="+", type=int, default=[1, 1], help="dilation rate")
  parser.add_argument("--loss_type", default="cross_entropy", help="loss type: cross_entropy, IOU, mean_squared")
  parser.add_argument("--weight_decay", default=0, type=float, help="weight decay")
  parser.add_argument("--optimizer", default="adam", help="optimizer: adam, momentum")
  parser.add_argument("--summary", default=True, type=bool, help="summary")
  parser.add_argument("--class_weights", nargs="+", default=[1, 1, 1], type=float, help="class weights")
  parser.add_argument("--log_dir", default="log", help="Tensorboard log directory (default: log)")
  parser.add_argument("--model_dir", default=None, help="Checkpoint directory (default: None)")
  parser.add_argument("--num_plots", default=10, type=int, help="Plotting trainning results")
  parser.add_argument("--tp_prob", default=0.3, type=float, help="Probability threshold for P pick")
  parser.add_argument("--ts_prob", default=0.3, type=float, help="Probability threshold for S pick")
  parser.add_argument("--train_dir", default="./dataset/waveform_train/", help="Input file directory")
  parser.add_argument("--train_list", default="./dataset/waveform.csv", help="Input csv file")
  parser.add_argument("--valid_dir", default=None, help="Input file directory")
  parser.add_argument("--valid_list", default=None, help="Input csv file")
  parser.add_argument("--test_dir", default=None, help="Input file directory")
  parser.add_argument("--test_list", default=None, help="Input csv file")
  parser.add_argument("--plot_figure", action="store_true", help="If plot figure for test")
  parser.add_argument("--save_prob", action="store_true", help="If save result for test")
  args = parser.parse_args()
  return args

def train_fn(args, data_reader, data_reader_valid=None):
  current_time = time.strftime("%y%m%d-%H%M%S")
  log_dir = os.path.join(args.log_dir, current_time)
  logging.info("Training log: {}".format(log_dir))
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  figure_dir = os.path.join(log_dir, 'figures')
  if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)

  with tf.compat.v1.name_scope('Input_Batch'):
    batch = data_reader.dequeue(args.batch_size)
    if data_reader_valid is not None:
      batch_valid = data_reader_valid.dequeue(args.batch_size)

  config = ModelConfig(X_shape=data_reader.X_shape, Y_shape=data_reader.Y_shape)
  if args.decay_step == -1:
    args.decay_step = data_reader.num_data // args.batch_size
  config.update_args(args)
  with open(os.path.join(log_dir, 'config.log'), 'w') as fp:
    fp.write('\n'.join("%s: %s" % item for item in vars(config).items()))

  model = UNet(config)
  sess_config = tf.compat.v1.ConfigProto()
  sess_config.gpu_options.allow_growth = True
  # sess_config.log_device_placement = False

  with tf.compat.v1.Session(config=sess_config) as sess:

    summary_writer = tf.compat.v1.summary.FileWriter(log_dir, sess.graph)
    saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=5)
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)

    if args.model_dir is not None:
      logging.info("restoring models...")
      latest_check_point = tf.train.latest_checkpoint(args.model_dir)
      saver.restore(sess, latest_check_point)

    threads = data_reader.start_threads(sess, n_threads=multiprocessing.cpu_count())
    if data_reader_valid is not None:
      threads_valid = data_reader_valid.start_threads(sess, n_threads=multiprocessing.cpu_count())
    flog = open(os.path.join(log_dir, 'loss.log'), 'w')
    total_step = 0
    mean_loss = 0
    if args.plot_figure:
      num_pool = multiprocessing.cpu_count()*2
    elif args.save_prob:
      num_pool = multiprocessing.cpu_count()
    else:
      num_pool = 2
    pool = multiprocessing.Pool(num_pool)

    for epoch in range(args.epochs):
      progressbar = tqdm(range(0, data_reader.num_data, args.batch_size), desc="{}: epoch {}".format(log_dir.split("/")[-1], epoch))
      for step in progressbar:
        X_batch, Y_batch = sess.run(batch)
        loss_batch = model.train_on_batch(sess, X_batch, Y_batch, summary_writer, args.drop_rate)
        if epoch < 1:
          mean_loss = loss_batch
        else:
          total_step += 1
          mean_loss += (loss_batch-mean_loss)/total_step
        progressbar.set_description("{}: epoch {}, loss={:.6f}, mean={:.6f}".format(log_dir.split("/")[-1], epoch, loss_batch, mean_loss))
      flog.write("epoch: {}, mean loss: {}\n".format(epoch, mean_loss))
      
      if data_reader_valid is not None:
        valid_step = 0
        valid_loss = 0
        progressbar = tqdm(range(0, data_reader_valid.num_data, args.batch_size), desc="Valid:")
        for step in progressbar:
          X_batch, Y_batch = sess.run(batch_valid)
          loss_batch, preds_batch = model.valid_on_batch(sess, X_batch, Y_batch, summary_writer)
          valid_step += 1
          valid_loss += (loss_batch-valid_loss)/valid_step
          progressbar.set_description("valid, loss={:.6f}, mean={:.6f}".format(loss_batch, valid_loss))
        flog.write("Valid: mean loss: {}\n".format(valid_loss))
      else:
        loss_batch, preds_batch = model.valid_on_batch(sess, X_batch, Y_batch, summary_writer)
      # loss_batch, pred_batch, logits_batch, X_batch, Y_batch = model.train_on_batch(sess, summary_writer, args.drop_rate, raw_data=True)
      try: ## IO Error on cluster
        flog.flush()
        pool.map(partial(plot_result_thread,
                        pred = preds_batch,
                        X = X_batch,
                        Y = Y_batch,
                        fname = ["{:03d}_{:03d}".format(epoch, x).encode() for x in range(args.num_plots)],
                        figure_dir = figure_dir),
                range(args.num_plots))
        saver.save(sess, os.path.join(log_dir, "model_{}.ckpt".format(epoch)))
      except:
        pass
    flog.close()
    pool.close()
    data_reader.coord.request_stop()
    try:
      data_reader.coord.join(threads, stop_grace_period_secs=10, ignore_live_threads=True)
      if data_reader_valid is not None:
        data_reader_valid.coord.join(threads_valid, stop_grace_period_secs=10, ignore_live_threads=True)
    except:
      pass
    sess.run(data_reader.queue.close(cancel_pending_enqueues=True))
    if data_reader_valid is not None:
      sess.run(data_reader_valid.queue.close(cancel_pending_enqueues=True))
      
  return 0

def test_fn(args, data_reader, figure_dir=None, result_dir=None):
  current_time = time.strftime("%y%m%d-%H%M%S")
  logging.info("{} log: {}".format(args.mode, current_time))
  log_dir = os.path.join(args.log_dir, args.mode, current_time)
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  if (args.plot_figure == True ) and (figure_dir is None):
    figure_dir = os.path.join(log_dir, 'figures')
    if not os.path.exists(figure_dir):
      os.makedirs(figure_dir)
  if (args.save_prob == True) and (result_dir is None):
    result_dir = os.path.join(log_dir, 'results')
    if not os.path.exists(result_dir):
      os.makedirs(result_dir)

  with tf.compat.v1.name_scope('Input_Batch'):
    batch = data_reader.dequeue(args.batch_size)

  config = ModelConfig(X_shape=data_reader.X_shape, Y_shape=data_reader.Y_shape)
  if args.decay_step == -1:
    args.decay_step = data_reader.num_data // args.batch_size
  config.update_args(args)
  with open(os.path.join(log_dir, 'config.log'), 'w') as fp:
    fp.write('\n'.join("%s: %s" % item for item in vars(config).items()))

  model = UNet(config, input_batch=batch, mode='valid')
  sess_config = tf.compat.v1.ConfigProto()
  sess_config.gpu_options.allow_growth = True
  # sess_config.log_device_placement = False

  with tf.compat.v1.Session(config=sess_config) as sess:

    summary_writer = tf.compat.v1.summary.FileWriter(log_dir, sess.graph)
    saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=5)
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)

    logging.info("restoring models...")
    latest_check_point = tf.train.latest_checkpoint(args.model_dir)
    saver.restore(sess, latest_check_point)
    
    threads = data_reader.start_threads(sess, n_threads=8)
    flog = open(os.path.join(log_dir, 'loss.log'), 'w')
    total_step = 0
    mean_loss = 0
    picks = []
    itp = []
    its = []
    progressbar = tqdm(range(0, data_reader.num_data, args.batch_size), desc=args.mode)
    if args.plot_figure:
      num_pool = multiprocessing.cpu_count()*2
    elif args.save_prob:
      num_pool = multiprocessing.cpu_count()
    else:
      num_pool = 2
    pool = multiprocessing.Pool(num_pool)
    for step in progressbar:
      
      if step + args.batch_size >= data_reader.num_data:
        for t in threads:
          t.join()
        sess.run(data_reader.queue.close())

      loss_batch, pred_batch, X_batch, Y_batch, \
      fname_batch, itp_batch, its_batch = model.test_on_batch(sess, summary_writer)
      total_step += 1
      mean_loss += (loss_batch-mean_loss)/total_step
      progressbar.set_description("{}, loss={:.6f}, mean loss={:6f}".format(args.mode, loss_batch, mean_loss))

      itp_batch = clean_queue(itp_batch)
      its_batch = clean_queue(its_batch)
      picks_batch = pool.map(partial(postprocessing_thread,
                               pred = pred_batch,
                               X = X_batch,
                               Y = Y_batch,
                               itp = itp_batch,
                               its = its_batch,
                               fname = fname_batch,
                               result_dir = result_dir,
                               figure_dir = figure_dir),
                       range(len(pred_batch)))
      picks.extend(picks_batch)
      itp.extend(itp_batch)
      its.extend(its_batch)

    flog.write("mean loss: {}\n".format(mean_loss))
    metrics_p, metrics_s = calculate_metrics(picks, itp, its, tol=0.1)
    flog.write("P-phase: Precision={}, Recall={}, F1={}\n".format(metrics_p[0], metrics_p[1], metrics_p[2]))
    flog.write("S-phase: Precision={}, Recall={}, F1={}\n".format(metrics_s[0], metrics_s[1], metrics_s[2]))
    flog.close()

  return 0

def main(args):

  logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
  coord = tf.train.Coordinator()

  if (args.mode == "train") or (args.mode == "train_valid"):
    with tf.compat.v1.name_scope('create_inputs'):
      data_reader = DataReader(
          data_dir=args.train_dir,
          data_list=args.train_list,
          mask_window=0.4,
          queue_size=args.batch_size*3,
          coord=coord)
      if args.mode == "train_valid":
        data_reader_valid = DataReader(
            data_dir=args.valid_dir,
            data_list=args.valid_list,
            mask_window=0.4,
            queue_size=args.batch_size*2,
            coord=coord)
        logging.info("Dataset size: train {}, valid {}".format(data_reader.num_data, data_reader_valid.num_data))
      else:
      	data_reader_valid = None
      	logging.info("Dataset size: train {}".format(data_reader.num_data))
    train_fn(args, data_reader, data_reader_valid)
  
  elif args.mode == "test":
    with tf.compat.v1.name_scope('create_inputs'):
      data_reader = DataReader_test(
          data_dir=args.test_dir,
          data_list=args.test_list,
          mask_window=0.4,
          queue_size=args.batch_size*10,
          coord=coord)
    test_fn(args, data_reader)

  else:
    print("mode should be: train, train_valid, pred or debug")

  return


if __name__ == '__main__':
  args = read_args()
  main(args)
