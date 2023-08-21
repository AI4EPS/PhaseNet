import os
from collections import defaultdict, namedtuple
from datetime import datetime, timedelta
from json import dumps
from typing import Any, AnyStr, Dict, List, NamedTuple, Union, Optional

import numpy as np
import requests
import tensorflow as tf
from fastapi import FastAPI, WebSocket
from kafka import KafkaProducer
from pydantic import BaseModel
from scipy.interpolate import interp1d

from model import ModelConfig, UNet
from postprocess import extract_picks

tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
JSONObject = Dict[AnyStr, Any]
JSONArray = List[Any]
JSONStructure = Union[JSONArray, JSONObject]

app = FastAPI()
X_SHAPE = [3000, 1, 3]
SAMPLING_RATE = 100

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

# GAMMA API Endpoint
GAMMA_API_URL = "http://gamma-api:8001"
# GAMMA_API_URL = 'http://localhost:8001'
# GAMMA_API_URL = "http://gamma.quakeflow.com"
# GAMMA_API_URL = "http://127.0.0.1:8001"

# Kafak producer
use_kafka = False

try:
    print("Connecting to k8s kafka")
    BROKER_URL = "quakeflow-kafka-headless:9092"
    # BROKER_URL = "34.83.137.139:9094"
    producer = KafkaProducer(
        bootstrap_servers=[BROKER_URL],
        key_serializer=lambda x: dumps(x).encode("utf-8"),
        value_serializer=lambda x: dumps(x).encode("utf-8"),
    )
    use_kafka = True
    print("k8s kafka connection success!")
except BaseException:
    print("k8s Kafka connection error")
    try:
        print("Connecting to local kafka")
        producer = KafkaProducer(
            bootstrap_servers=["localhost:9092"],
            key_serializer=lambda x: dumps(x).encode("utf-8"),
            value_serializer=lambda x: dumps(x).encode("utf-8"),
        )
        use_kafka = True
        print("local kafka connection success!")
    except BaseException:
        print("local Kafka connection error")
print(f"Kafka status: {use_kafka}")


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
        std[:, i, :] = np.std(data_pad[:, i * shift : i * shift + window, :], axis=1)
        mean[:, i, :] = np.mean(data_pad[:, i * shift : i * shift + window, :], axis=1)

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
                picks_.append(
                    {
                        "id": pick.fname,
                        "timestamp": calc_timestamp(pick.t0, float(idx) * dt),
                        "prob": prob,
                        "amp": amp,
                        "type": "p",
                    }
                )
        for idxs, probs, amps in zip(pick.s_idx, pick.s_prob, amplitude.s_amp):
            for idx, prob, amp in zip(idxs, probs, amps):
                picks_.append(
                    {
                        "id": pick.fname,
                        "timestamp": calc_timestamp(pick.t0, float(idx) * dt),
                        "prob": prob,
                        "amp": amp,
                        "type": "s",
                    }
                )
    return picks_


def format_data(data):
    # chn2idx = {"ENZ": {"E":0, "N":1, "Z":2},
    #            "123": {"3":0, "2":1, "1":2},
    #            "12Z": {"1":0, "2":1, "Z":2}}
    chn2idx = {"E": 0, "N": 1, "Z": 2, "3": 0, "2": 1, "1": 2}
    Data = NamedTuple("data", [("id", list), ("timestamp", list), ("vec", list), ("dt", float)])

    # Group by station
    chn_ = defaultdict(list)
    t0_ = defaultdict(list)
    vv_ = defaultdict(list)
    for i in range(len(data.id)):
        key = data.id[i][:-1]
        chn_[key].append(data.id[i][-1])
        t0_[key].append(datetime.strptime(data.timestamp[i], "%Y-%m-%dT%H:%M:%S.%f").timestamp() * SAMPLING_RATE)
        vv_[key].append(np.array(data.vec[i]))

    # Merge to Data tuple
    id_ = []
    timestamp_ = []
    vec_ = []
    for k in chn_:
        id_.append(k)
        min_t0 = min(t0_[k])
        timestamp_.append(datetime.fromtimestamp(min_t0 / SAMPLING_RATE).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3])
        vec = np.zeros([X_SHAPE[0], X_SHAPE[-1]])
        for i in range(len(chn_[k])):
            # vec[int(t0_[k][i]-min_t0):len(vv_[k][i]), chn2idx[chn_[k][i]]] = vv_[k][i][int(t0_[k][i]-min_t0):X_SHAPE[0]] - np.mean(vv_[k][i])
            shift = int(t0_[k][i] - min_t0)
            vec[shift : len(vv_[k][i]) + shift, chn2idx[chn_[k][i]]] = vv_[k][i][: X_SHAPE[0] - shift] - np.mean(
                vv_[k][i][: X_SHAPE[0] - shift]
            )
        vec_.append(vec.tolist())

    return Data(id=id_, timestamp=timestamp_, vec=vec_, dt=1 / SAMPLING_RATE)
    # return {"id": id_, "timestamp": timestamp_, "vec": vec_, "dt":1 / SAMPLING_RATE}


def get_prediction(data, return_preds=False):
    vec = np.array(data.vec)
    vec, vec_raw = preprocess(vec)

    feed = {model.X: vec, model.drop_rate: 0, model.is_training: False}
    preds = sess.run(model.preds, feed_dict=feed)

    picks = extract_picks(preds, station_ids=data.id, begin_times=data.timestamp, waveforms=vec_raw)

    picks = [
        {k: v for k, v in pick.items() if k in ["station_id", "phase_time", "phase_score", "phase_type", "dt"]}
        for pick in picks
    ]

    if return_preds:
        return picks, preds

    return picks


class Data(BaseModel):
    # id: Union[List[str], str]
    # timestamp: Union[List[str], str]
    # vec: Union[List[List[List[float]]], List[List[float]]]
    id: List[str]
    timestamp: List[Union[str, float, datetime]]
    vec: Union[List[List[List[float]]], List[List[float]]]

    dt: Optional[float] = 0.01
    ## gamma
    stations: Optional[List[Dict[str, Union[float, str]]]] = None
    config: Optional[Dict[str, Union[List[float], List[int], List[str], float, int, str]]] = None


# @app.on_event("startup")
# def set_default_executor():
#     from concurrent.futures import ThreadPoolExecutor
#     import asyncio
#
#     loop = asyncio.get_running_loop()
#     loop.set_default_executor(
#         ThreadPoolExecutor(max_workers=2)
#     )


@app.post("/predict")
def predict(data: Data):
    picks = get_prediction(data)

    return picks


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_json()
        # data = json.loads(data)
        data = Data(**data)
        picks = get_prediction(data)
        await websocket.send_json(picks)
        print("PhaseNet Updating...")


@app.post("/predict_prob")
def predict(data: Data):
    picks, preds = get_prediction(data, True)

    return picks, preds.tolist()


@app.post("/predict_phasenet2gamma")
def predict(data: Data):
    picks = get_prediction(data)

    # if use_kafka:
    #     print("Push picks to kafka...")
    #     for pick in picks:
    #         producer.send("phasenet_picks", key=pick["id"], value=pick)
    try:
        catalog = requests.post(
            f"{GAMMA_API_URL}/predict", json={"picks": picks, "stations": data.stations, "config": data.config}
        )
        print(catalog.json()["catalog"])
        return catalog.json()
    except Exception as error:
        print(error)

    return {}


@app.post("/predict_phasenet2gamma2ui")
def predict(data: Data):
    picks = get_prediction(data)

    try:
        catalog = requests.post(
            f"{GAMMA_API_URL}/predict", json={"picks": picks, "stations": data.stations, "config": data.config}
        )
        print(catalog.json()["catalog"])
        return catalog.json()
    except Exception as error:
        print(error)

    if use_kafka:
        print("Push picks to kafka...")
        for pick in picks:
            producer.send("phasenet_picks", key=pick["id"], value=pick)
        print("Push waveform to kafka...")
        for id, timestamp, vec in zip(data.id, data.timestamp, data.vec):
            producer.send("waveform_phasenet", key=id, value={"timestamp": timestamp, "vec": vec, "dt": data.dt})

    return {}


@app.post("/predict_stream_phasenet2gamma")
def predict(data: Data):
    data = format_data(data)
    # for i in range(len(data.id)):
    #     plt.clf()
    #     plt.subplot(311)
    #     plt.plot(np.array(data.vec)[i, :, 0])
    #     plt.subplot(312)
    #     plt.plot(np.array(data.vec)[i, :, 1])
    #     plt.subplot(313)
    #     plt.plot(np.array(data.vec)[i, :, 2])
    #     plt.savefig(f"{data.id[i]}.png")

    picks = get_prediction(data)

    return_value = {}
    try:
        catalog = requests.post(f"{GAMMA_API_URL}/predict_stream", json={"picks": picks})
        print("GMMA:", catalog.json()["catalog"])
        return_value = catalog.json()
    except Exception as error:
        print(error)

    if use_kafka:
        print("Push picks to kafka...")
        for pick in picks:
            producer.send("phasenet_picks", key=pick["id"], value=pick)
        print("Push waveform to kafka...")
        for id, timestamp, vec in zip(data.id, data.timestamp, data.vec):
            producer.send("waveform_phasenet", key=id, value={"timestamp": timestamp, "vec": vec, "dt": data.dt})

    return return_value


@app.get("/healthz")
def healthz():
    return {"status": "ok"}
