import os
from collections import defaultdict, namedtuple
from datetime import datetime, timedelta
from json import dumps

import numpy as np
import tensorflow as tf

from model import ModelConfig, UNet
from postprocess import extract_amplitude, extract_picks
import pandas as pd
import obspy


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


stream = obspy.read()
stream = stream.sort() ## Assume it is NPZ sorted
assert(len(stream) == 3)
data = []
for trace in stream:
    data.append(trace.data)
data = np.array(data).T
assert(data.shape[-1] == 3)

# data_id = stream[0].get_id()[:-1]
# timestamp = stream[0].stats.starttime.datetime.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

data = np.stack([data for i in range(10)]) ## Assume 10 windows
data = data[:,:,np.newaxis,:] ## batch, nt, dummy_dim, channel
print(f"{data.shape = }")
data = (data - data.mean(axis=1, keepdims=True))/data.std(axis=1, keepdims=True)

feed = {model.X: data, model.drop_rate: 0, model.is_training: False}
preds = sess.run(model.preds, feed_dict=feed)

picks = extract_picks(preds, fnames=None, station_ids=None, t0=None)
picks = format_picks(picks, dt=0.01)


picks = pd.DataFrame(picks)
print(picks)