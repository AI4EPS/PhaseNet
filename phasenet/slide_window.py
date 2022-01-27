import os
from collections import defaultdict, namedtuple
from datetime import datetime, timedelta
from json import dumps

import numpy as np
import tensorflow as tf

from model import ModelConfig, UNet
from postprocess import extract_amplitude, extract_picks
import pandas as pd

tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))

# load model
model = UNet(mode="pred")
sess_config = tf.compat.v1.ConfigProto()
sess_config.gpu_options.allow_growth = True

sess = tf.compat.v1.Session(config=sess_config)
saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
init = tf.compat.v1.global_variables_initializer()
sess.run(init)
latest_check_point = tf.train.latest_checkpoint(f"{PROJECT_ROOT}/model/190703-214543")
print(f"restoring model {latest_check_point}")
saver.restore(sess, latest_check_point)


def calc_timestamp(timestamp, sec):
    timestamp = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%f") + timedelta(seconds=sec)
    return timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

def format_picks(picks, dt):
    picks_ = []
    for pick in picks:
        for idxs, probs in zip(pick.p_idx, pick.p_prob):
            for idx, prob in zip(idxs, probs):
                picks_.append(
                    {
                        "id": pick.fname,
                        "timestamp": calc_timestamp(pick.t0, float(idx) * dt),
                        "prob": prob,
                        "type": "p",
                    }
                )
        for idxs, probs in zip(pick.s_idx, pick.s_prob):
            for idx, prob in zip(idxs, probs):
                picks_.append(
                    {
                        "id": pick.fname,
                        "timestamp": calc_timestamp(pick.t0, float(idx) * dt),
                        "prob": prob,
                        "type": "s",
                    }
                )
    return picks_


vec = np.random.randn(10, 3000, 1, 3) ## batch, nt, dummy_dim, channel
dt = 0.01

feed = {model.X: vec, model.drop_rate: 0, model.is_training: False}
preds = sess.run(model.preds, feed_dict=feed)

picks = extract_picks(preds, fnames=None, station_ids=None, t0=None)
picks = format_picks(picks, dt)


picks = pd.DataFrame(picks)
print(picks)