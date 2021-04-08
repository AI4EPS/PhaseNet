import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import argparse, os, time, logging
from tqdm import tqdm
import pandas as pd
import multiprocessing
from functools import partial
import pickle
from model import UNet, ModelConfig
from data_reader import DataReader_pred, DataReader_mseed_array
from postprocess import extract_picks, save_picks, save_picks_json, extract_amplitude

def read_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=20, type=int, help="batch size")
    parser.add_argument("--model_dir", help="Checkpoint directory (default: None)")
    parser.add_argument("--data_dir", default="", help="Input file directory")
    parser.add_argument("--data_list", default="", help="Input csv file")
    parser.add_argument("--result_dir", default="results", help="Output directory")
    parser.add_argument("--result_fname", default="picks.csv", help="Output file")
    parser.add_argument("--min_p_prob", default=0.3, type=float, help="Probability threshold for P pick")
    parser.add_argument("--min_s_prob", default=0.3, type=float, help="Probability threshold for S pick")
    parser.add_argument("--amplitude", action="store_true", help="if return amplitude value")
    parser.add_argument("--format", default="numpy", help="input format")
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

        picks = []
        amps = [] if args.amplitude else None

        for _ in tqdm(range(0, data_reader.num_data, batch_size), desc="Pred"):
            if args.amplitude:
                pred_batch, X_batch, amp_batch, fname_batch, t0_batch = sess.run([model.preds, batch[0], batch[1], batch[2], batch[3]], 
                                                                                 feed_dict={model.drop_rate: 0, model.is_training: False})
            else:
                pred_batch, X_batch, fname_batch, t0_batch = sess.run([model.preds, batch[0], batch[1], batch[2]], 
                                                                      feed_dict={model.drop_rate: 0, model.is_training: False})

            picks_ = extract_picks(preds=pred_batch, fnames=fname_batch, t0=t0_batch)
            picks.extend(picks_)
            if args.amplitude:
                amps_ = extract_amplitude(amp_batch, picks_)
                amps.extend(amps_)

        save_picks(picks, args.result_dir, amps=amps)
        save_picks_json(picks, args.result_dir, dt=data_reader.dt, amps=amps)

    print(f"Done with {sum([len(x) for pick in picks for x in pick.p_idx])} P-picks and {sum([len(x) for pick in picks for x in pick.s_idx])} S-picks")
    return 0


def main(args):

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    with tf.compat.v1.name_scope('create_inputs'):

        if args.format == "mseed_array":
            data_reader = DataReader_mseed_array(data_dir=args.data_dir,
                                                 data_list=args.data_list,
                                                 stations=args.stations,
                                                 amplitude=args.amplitude)
        else:
            data_reader = DataReader_pred(format=args.format,
                                          data_dir=args.data_dir,
                                          data_list=args.data_list,
                                          amplitude=args.amplitude)
        
        pred_fn(args, data_reader, log_dir=args.result_dir)

    return

if __name__ == '__main__':
    args = read_args()
    main(args)
