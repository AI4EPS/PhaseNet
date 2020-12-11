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
from data_reader import DataReader_pred, DataReader_mseed, DataReader_s3
from util import *

def read_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", help="Checkpoint directory (default: None)")
    parser.add_argument("--data_dir", default="", help="Input file directory")
    parser.add_argument("--data_list", default="", help="Input csv file")
    parser.add_argument("--result_dir", default="results", help="Output directory")
    parser.add_argument("--result_name", default="picks.csv", help="Output file")
    parser.add_argument("--batch_size", default=20, type=int, help="batch size")
    parser.add_argument("--tp_prob", default=0.3, type=float, help="Probability threshold for P pick")
    parser.add_argument("--ts_prob", default=0.3, type=float, help="Probability threshold for S pick")
    parser.add_argument("--input_mseed", action="store_true", help="mseed format")
    parser.add_argument("--input_s3", action="store_true", help="s3 format")
    parser.add_argument("--s3_url", default="localhost:9000", help="s3 url")
    parser.add_argument("--stations", default="", help="seismic station info")
    parser.add_argument("--plot_figure", action="store_true", help="If plot figure for test")
    parser.add_argument("--save_prob", action="store_true", help="If save result for test")
    args = parser.parse_args()
    return args


def pred_fn(args, data_reader, figure_dir=None, prob_dir=None, log_dir=None):
    current_time = time.strftime("%y%m%d-%H%M%S")
    if log_dir is None:
        log_dir = os.path.join(args.log_dir, "pred", current_time)
    logging.info("Pred log: %s" % log_dir)
    logging.info("Dataset size: {}".format(data_reader.num_data))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if (args.plot_figure == True) and (figure_dir is None):
        figure_dir = os.path.join(log_dir, 'figures')
        if not os.path.exists(figure_dir):
            os.makedirs(figure_dir)
    if (args.save_prob == True) and (prob_dir is None):
        prob_dir = os.path.join(log_dir, 'results')
        if not os.path.exists(prob_dir):
            os.makedirs(prob_dir)

    with tf.compat.v1.name_scope('Input_Batch'):
        dataset = data_reader.dataset().prefetch(10)
        if args.input_mseed:
            batch_size = 1
        else:
            batch_size = args.batch_size
            dataset = dataset.batch(batch_size)
        batch = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()

    config = ModelConfig(X_shape=data_reader.X_shape)
    with open(os.path.join(log_dir, 'config.log'), 'w') as fp:
        fp.write('\n'.join("%s: %s" % item for item in vars(config).items()))

    model = UNet(config=config, input_batch=batch, mode="pred")
    sess_config = tf.compat.v1.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    # sess_config.log_device_placement = False

    with tf.compat.v1.Session(config=sess_config) as sess:

        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=5)
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        latest_check_point = tf.train.latest_checkpoint(args.model_dir)
        logging.info(f"restoring model {latest_check_point}")
        saver.restore(sess, latest_check_point)

        if args.plot_figure:
            num_pool = multiprocessing.cpu_count()*2
        elif args.save_prob:
            num_pool = multiprocessing.cpu_count()
        else:
            num_pool = 2
        pool = multiprocessing.Pool(num_pool)

        fclog = open(os.path.join(log_dir, args.result_fname+'.csv'), 'w')
        fclog.write("fname,itp,tp_prob,its,ts_prob\n") 
        picks = {}

        for step in tqdm(range(0, data_reader.num_data, batch_size), desc="Pred"):
            pred_batch, X_batch, fname_batch = sess.run([model.preds, batch[0], batch[1]], 
                                                         feed_dict={model.drop_rate: 0,
                                                                    model.is_training: False})
            if args.input_mseed:
                fname_batch = [(fname_batch.decode().split('/')[-1].rstrip(".mseed")+"."+data_reader.stations.iloc[i]["station"]).encode() 
                               for i in range(len(pred_batch))]
            picks_batch = pool.map(partial(postprocessing_thread,
                                           pred = pred_batch,
                                           X = X_batch,
                                           fname = fname_batch,
                                           result_dir = prob_dir,
                                           figure_dir = figure_dir,
                                           args=args), range(len(pred_batch)))

            for i in range(len(fname_batch)):
                row = "{},[{}],[{}],[{}],[{}]".format(fname_batch[i].decode(), " ".join(map(str,picks_batch[i][0][0])), " ".join(map(str,picks_batch[i][0][1])),
                                            " ".join(map(str,picks_batch[i][1][0])), " ".join(map(str,picks_batch[i][1][1])))
                fclog.write(row+"\n")
                picks[fname_batch[i].decode()]={"itp":picks_batch[i][0][0], "tp_prob":picks_batch[i][0][1], "its":picks_batch[i][1][0], "ts_prob":picks_batch[i][1][1]}
            
            fclog.flush()

    fclog.close()
    with open(os.path.join(log_dir, args.result_fname+'.pkl'), 'wb') as fp:
        pickle.dump(picks, fp)
    print("Done")

    return 0


def main(args):

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    with tf.compat.v1.name_scope('create_inputs'):

        if args.input_mseed:
            data_reader = DataReader_mseed(data_dir=args.data_dir,
                                           data_list=args.data_list,
                                           stations=args.stations)
        elif args.input_s3:
            args.input_mseed = True
            from minio import Minio            
            s3_client = Minio(f'{args.s3_url}',
                        access_key='quakeflow',
                        secret_key='quakeflow',
                        secure=False)
            data_reader = DataReader_s3(data_dir=args.data_dir,
                                           data_list=args.data_list,
                                           stations=args.stations,
                                           s3_client=s3_client,
                                           bucket="waveforms")
        else:
            data_reader = DataReader_pred(data_dir=args.data_dir,
                                          data_list=args.data_list)
        
        pred_fn(args, data_reader, log_dir=args.result_dir)

    return


if __name__ == '__main__':
    args = read_args()
    main(args)
