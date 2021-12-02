import argparse
import logging
import multiprocessing
import os
import pickle
import time
from functools import partial

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from .data_reader import DataReader_mseed_array, DataReader_pred
from .model import ModelConfig, UNet
from .postprocess import (
    extract_amplitude,
    extract_picks,
    save_picks,
    save_picks_json,
    save_prob_h5,
)
from .visulization import plot_waveform

tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def pred_fn(data_reader, figure_dir=None, prob_dir=None, log_dir=None, **kwargs):
    # ----------------------------------------------------------
    #               define default arguments
    # ----------------------------------------------------------
    kwargs.setdefault('batch_size', 20)
    kwargs.setdefault('model_dir', '')
    kwargs.setdefault('data_dir', '')
    kwargs.setdefault('hdf5_file', 'data.h5')
    kwargs.setdefault('hdf5_group', 'data')
    kwargs.setdefault('min_p_prob', 0.6)
    kwargs.setdefault('min_s_prob', 0.6)
    kwargs.setdefault('mpd', 50)
    kwargs.setdefault('amplitude', False)
    kwargs.setdefault('format', 'hdf5')
    kwargs.setdefault('s3_url', 'localhost:9000')
    kwargs.setdefault('stations', '')
    # ----------------------------------------------------------
    #      cheap trick to reuse most of the original code
    # ----------------------------------------------------------
    args = AttrDict(kwargs)
    # ----------------------------------------------------------
    current_time = time.strftime("%y%m%d-%H%M%S")
    if log_dir is None:
        log_dir = os.path.join(args.log_dir, "pred", current_time)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.info("Pred log: %s" % log_dir)
    logging.info("Dataset size: {}".format(data_reader.num_data))

    with tf.compat.v1.name_scope('Input_Batch'):
        if args.format == "mseed_array":
            batch_size = 1
        else:
            batch_size = args.batch_size
        dataset = data_reader.dataset(batch_size)
        batch = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()

    config = ModelConfig(X_shape=data_reader.X_shape)
    with open(os.path.join(log_dir, 'config.log'), 'w') as fp:
        fp.write('\n'.join("%s: %s" % item for item in vars(config).items()))

    model = UNet(config=config, input_batch=batch, mode="pred")
    # model = UNet(config=config, mode="pred")
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

        predictions, fnames, picks = [], [], []
        amps = [] if args.amplitude else None

        for _ in tqdm(range(0, data_reader.num_data, batch_size), desc="Pred"):
            if args.amplitude:
                pred_batch, X_batch, amp_batch, fname_batch, t0_batch = sess.run(
                    [model.preds, batch[0], batch[1], batch[2], batch[3]],
                    feed_dict={model.drop_rate: 0, model.is_training: False},
                )
            #    X_batch, amp_batch, fname_batch, t0_batch = sess.run([batch[0], batch[1], batch[2], batch[3]])
            else:
                pred_batch, X_batch, fname_batch, t0_batch = sess.run(
                    [model.preds, batch[0], batch[1], batch[2]],
                    feed_dict={model.drop_rate: 0, model.is_training: False},
                )

            picks_ = extract_picks(preds=pred_batch, fnames=fname_batch, t0=t0_batch, config=args)
            picks.extend(picks_)
            if args.amplitude:
                amps_ = extract_amplitude(amp_batch, picks_)
                amps.extend(amps_)


            # store the batch predictions
            predictions.extend(pred_batch)
            fnames.extend(fname_batch)

        # convert lists to numpy arrays
        predictions = np.float32(predictions).squeeze()
        fnames = list(np.asarray(fnames).astype('U'))

        # order the outputs
        ordered_proba = np.zeros_like(predictions, dtype=np.float32)
        ordered_picks = np.zeros((data_reader.num_data, 2, 2), dtype=object)
        for i in range(data_reader.num_data):
            sample_name = f'sample{i}'
            idx = fnames.index(sample_name)
            ordered_proba[i, ...] = predictions[idx, ...].squeeze()
            ordered_picks[i, 0, 0] = np.array(picks[idx].p_idx).squeeze()
            ordered_picks[i, 0, 1] = np.array(picks[idx].p_prob).squeeze()
            ordered_picks[i, 1, 0] = np.array(picks[idx].s_idx).squeeze()
            ordered_picks[i, 1, 1] = np.array(picks[idx].s_prob).squeeze()
    return ordered_proba, ordered_picks

