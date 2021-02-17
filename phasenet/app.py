import requests
from fastapi import FastAPI
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from model import UNet, ModelConfig
from postprocess import extract_picks, extract_amplitude
from data_reader import normalize_batch
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import List, Any

app = FastAPI()

## load model
config = ModelConfig(X_shape=[3000, 1, 3])
model = UNet(config=config, mode="pred")
sess_config = tf.compat.v1.ConfigProto()
sess_config.gpu_options.allow_growth = True

sess = tf.compat.v1.Session(config=sess_config)
saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
init = tf.compat.v1.global_variables_initializer()
sess.run(init)
latest_check_point = tf.train.latest_checkpoint("../model/190703-214543")
print(f"restoring model {latest_check_point}")
saver.restore(sess, latest_check_point)

# GMMA API Endpoint
GMMA_API_URL = 'http://localhost:8001'

def preprocess(data):
    data = normalize_batch(data)
    if len(data.shape) == 3:
        data = data[:,:,np.newaxis,:]
    return data

def calc_timestamp(timestamp, sec):
    timestamp = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%f") + timedelta(seconds=sec)
    return timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

def format_picks(picks, dt, amplitudes):
    picks_ = []
    for pick, amplitude in zip(picks, amplitudes):
        for idxs, probs, amps in zip(pick.p_idx, pick.p_prob, amplitude.p_amp):
            for idx, prob, amp in zip(idxs, probs, amps):
                picks_.append({"id": pick.fname, 
                               "timestamp":calc_timestamp(pick.t0, float(idx)*dt), 
                               "prob": prob, 
                               "amp": amp,
                               "type": "p"})
        for idxs, probs, amps in zip(pick.s_idx, pick.s_prob, amplitude.s_amp):
            for idx, prob, amp in zip(idxs, probs, amps):
                picks_.append({"id": pick.fname, 
                               "timestamp":calc_timestamp(pick.t0, float(idx)*dt), 
                               "prob": prob, 
                               "amp": amp,
                               "type": "s"})
    return picks_

def get_prediction(data):

    vec = np.array(data.vec)
    vec = preprocess(vec)

    feed = {model.X: vec,
            model.drop_rate: 0,
            model.is_training: False}
    preds = sess.run(model.preds, feed_dict=feed)

    picks = extract_picks(preds, fnames=data.id, t0=data.timestamp)
    amps = extract_amplitude(vec, picks)
    picks = format_picks(picks, data.dt, amps)
    return picks
 
class Data(BaseModel):
    id: List[str]
    timestamp: List[str]
    vec: List[List[List[float]]]
    dt: float = 0.01

@app.get('/predict')
def predict(data: Data):

    picks = get_prediction(data)

    return picks

@app.get('/predict2gmma')
def predict(data: Data):

    picks = get_prediction(data)

    # TODO
    # push prediction results to Kafka

    try:
        catalog = requests.get(f'{GMMA_API_URL}/predict', json={"picks": picks})
        print(catalog.json())
        return catalog.json()
    except Exception as error:
        print(error)
    return {}

