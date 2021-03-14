from kafka import KafkaProducer
from json import dumps
import os
from scipy.interpolate import interp1d
from typing import List, Any, List, Union, Dict, AnyStr
from datetime import datetime, timedelta
from pydantic import BaseModel
from postprocess import extract_picks, extract_amplitude
from model import UNet, ModelConfig
import requests
from fastapi import FastAPI, Request

import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
JSONObject = Dict[AnyStr, Any]
JSONArray = List[Any]
JSONStructure = Union[JSONArray, JSONObject]

app = FastAPI()

# load model
config = ModelConfig(X_shape=[3000, 1, 3])
model = UNet(config=config, mode="pred")
sess_config = tf.compat.v1.ConfigProto()
sess_config.gpu_options.allow_growth = True

sess = tf.compat.v1.Session(config=sess_config)
saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
init = tf.compat.v1.global_variables_initializer()
sess.run(init)
latest_check_point = tf.train.latest_checkpoint(f"{PROJECT_ROOT}/model/190703-214543")
print(f"restoring model {latest_check_point}")
saver.restore(sess, latest_check_point)

# GMMA API Endpoint
GMMA_API_URL = 'http://localhost:8001'

# Kafak producer
use_kafka = False
# BROKER_URL = 'localhost:9092'
# BROKER_URL = 'my-kafka-headless:9092'

try:
    print('Connecting to k8s kafka')
    BROKER_URL = 'quakeflow-kafka-headless:9092'
    producer = KafkaProducer(bootstrap_servers=[BROKER_URL],
                             key_serializer=lambda x: dumps(x).encode('utf-8'),
                             value_serializer=lambda x: dumps(x).encode('utf-8'))
    use_kafka = True
except BaseException:
    print('k8s Kafka connection error')

try:
    print('Connecting to local kafka')
    producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                             key_serializer=lambda x: dumps(x).encode('utf-8'),
                             value_serializer=lambda x: dumps(x).encode('utf-8'))
    use_kafka = True
except BaseException:
    print('local Kafka connection error')


def normalize_batch(data, window=3000):
    """
    data: nsta, nt, nch
    """
    shift = window // 2
    nsta, nt, nch = data.shape

    # std in slide windows
    data_pad = np.pad(data, ((0, 0), (window // 2, window // 2), (0, 0)), mode="reflect")
    t = np.arange(0, nt, shift, dtype="int")
    std = np.zeros([nsta, len(t) + 1, nch])
    mean = np.zeros([nsta, len(t) + 1, nch])
    for i in range(1, len(t)):
        std[:, i, :] = np.std(data_pad[:, i * shift:i * shift + window, :], axis=1)
        mean[:, i, :] = np.mean(data_pad[:, i * shift:i * shift + window, :], axis=1)

    t = np.append(t, nt)
    # std[:, -1, :] = np.std(data_pad[:, -window:, :], axis=1)
    # mean[:, -1, :] = np.mean(data_pad[:, -window:, :], axis=1)
    std[:, -1, :], mean[:, -1, :] = std[:, -2, :], mean[:, -2, :]
    std[:, 0, :], mean[:, 0, :] = std[:, 1, :], mean[:, 1, :]
    std[std == 0] = 1

    # ## normalize data with interplated std
    t_interp = np.arange(nt, dtype="int")
    std_interp = interp1d(t, std, axis=1, kind="slinear")(t_interp)
    mean_interp = interp1d(t, mean, axis=1, kind="slinear")(t_interp)
    data = (data - mean_interp) / std_interp

    return data


def preprocess(data):
    raw = data.copy()
    data = normalize_batch(data)
    if len(data.shape) == 3:
        data = data[:, :, np.newaxis, :]
        raw = raw[:, :, np.newaxis, :]
    return data, raw


def calc_timestamp(timestamp, sec):
    timestamp = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%f") + timedelta(seconds=sec)
    return timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]


def format_picks(picks, dt, amplitudes):
    picks_ = []
    for pick, amplitude in zip(picks, amplitudes):
        for idxs, probs, amps in zip(pick.p_idx, pick.p_prob, amplitude.p_amp):
            for idx, prob, amp in zip(idxs, probs, amps):
                picks_.append({"id": pick.fname,
                               "timestamp": calc_timestamp(pick.t0, float(idx) * dt),
                               "prob": prob,
                               "amp": amp,
                               "type": "p"})
        for idxs, probs, amps in zip(pick.s_idx, pick.s_prob, amplitude.s_amp):
            for idx, prob, amp in zip(idxs, probs, amps):
                picks_.append({"id": pick.fname,
                               "timestamp": calc_timestamp(pick.t0, float(idx) * dt),
                               "prob": prob,
                               "amp": amp,
                               "type": "s"})
    return picks_


def get_prediction(data):

    vec = np.array(data.vec)
    vec, vec_raw = preprocess(vec)

    feed = {model.X: vec,
            model.drop_rate: 0,
            model.is_training: False}
    preds = sess.run(model.preds, feed_dict=feed)

    picks = extract_picks(preds, fnames=data.id, t0=data.timestamp)
    amps = extract_amplitude(vec_raw, picks)
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
    if use_kafka:
        for pick in picks:
            producer.send('phasenet_picks', key=pick["id"], value=pick)

    try:
        catalog = requests.get(f'{GMMA_API_URL}/predict', json={"picks": picks})
        print(catalog.json())
        return catalog.json()
    except Exception as error:
        print(error)

    return {}


@app.get('/test')
def predict(data: JSONStructure):
    print(data)
    # picks = get_prediction(data)

    return {}
