
import os
import numpy as np
from collections import namedtuple
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
import logging
from .detect_peaks import detect_peaks

def extract_picks(preds, fnames=None, t0=None, config=None):

    if preds.shape[-1] == 4:
        record = namedtuple("phase", ["fname", "t0", "p_idx", "p_prob", "s_idx", "s_prob", "ps_idx", "ps_prob"])
    else:
        record = namedtuple("phase", ["fname", "t0", "p_idx", "p_prob", "s_idx", "s_prob"])

    picks = []
    for i, pred in enumerate(preds):

        if config is None:
            mph_p, mph_s, mpd = 0.3, 0.3, 50
        else:
            mph_p, mph_s, mpd = config.min_p_prob, config.min_s_prob, config.mpd

        if (fnames is None):
            fname = f"{i:04d}"
        else:
            if isinstance(fnames[i], str):
                fname = fnames[i]
            else:
                fname = fnames[i].decode()
            
        if (t0 is None):
            start_time = "0"
        else:
            if isinstance(t0[i], str):
                start_time = t0[i]
            else:
                start_time = t0[i].decode()

        p_idx, p_prob, s_idx, s_prob = [], [], [], []
        for j in range(pred.shape[1]):
            p_idx_, p_prob_ = detect_peaks(pred[:,j,1], mph=mph_p, mpd=mpd, show=False)
            s_idx_, s_prob_ = detect_peaks(pred[:,j,2], mph=mph_s, mpd=mpd, show=False)
            p_idx.append(list(p_idx_))
            p_prob.append(list(p_prob_))
            s_idx.append(list(s_idx_))
            s_prob.append(list(s_prob_))

        if pred.shape[-1] == 4:
            ps_idx, ps_prob = detect_peaks(pred[:,0,3], mph=0.3, mpd=mpd, show=False)
            picks.append(record(fname, start_time, list(p_idx), list(p_prob), list(s_idx), list(s_prob), list(ps_idx), list(ps_prob)))
        else:
            picks.append(record(fname, start_time, list(p_idx), list(p_prob), list(s_idx), list(s_prob)))

    return picks


def extract_amplitude(data, picks, window_p=10, window_s=5, config=None):
    record = namedtuple("amplitude", ["p_amp", "s_amp"])
    dt = 0.01 if config is None else config.dt
    window_p = int(window_p/dt)
    window_s = int(window_s/dt)
    amps = []
    for i, (da, pi) in enumerate(zip(data, picks)):
        p_amp, s_amp = [], []
        for j in range(da.shape[1]):
            amp = np.max(np.abs(da[:,j,:]), axis=-1)
            #amp = np.median(np.abs(da[:,j,:]), axis=-1)
            #amp = np.linalg.norm(da[:,j,:], axis=-1)
            tmp = []
            for k in range(len(pi.p_idx[j])-1):
                tmp.append(np.max(amp[pi.p_idx[j][k]:min(pi.p_idx[j][k]+window_p, pi.p_idx[j][k+1])]))
            if len(pi.p_idx[j]) >= 1:
                tmp.append(np.max(amp[pi.p_idx[j][-1]:pi.p_idx[j][-1]+window_p]))
            p_amp.append(tmp)
            tmp = []
            for k in range(len(pi.s_idx[j])-1):
                tmp.append(np.max(amp[pi.s_idx[j][k]:min(pi.s_idx[j][k]+window_s, pi.s_idx[j][k+1])]))
            if len(pi.s_idx[j]) >= 1:
                tmp.append(np.max(amp[pi.s_idx[j][-1]:pi.s_idx[j][-1]+window_s]))
            s_amp.append(tmp)
        amps.append(record(p_amp, s_amp))
    return amps


def save_picks(picks, output_dir, amps=None, fname=None):
    if fname is None:
        fname = "picks.csv"

    int2s = lambda x: ",".join(["["+",".join(map(str, i))+"]" for i in x])
    flt2s = lambda x: ",".join(["["+",".join(map("{:0.3f}".format, i))+"]" for i in x])
    sci2s = lambda x: ",".join(["["+",".join(map("{:0.3e}".format, i))+"]" for i in x])
    if amps is None:
        if hasattr(picks[0], "ps_idx"):
            with open(os.path.join(output_dir, fname), "w") as fp:
                fp.write("fname\tt0\tp_idx\tp_prob\ts_idx\ts_prob\tps_idx\tps_prob\n")
                for pick in picks:
                    fp.write(f"{pick.fname}\t{pick.t0}\t{int2s(pick.p_idx)}\t{flt2s(pick.p_prob)}\t{int2s(pick.s_idx)}\t{flt2s(pick.s_prob)}\t{int2s(pick.ps_idx)}\t{flt2s(pick.ps_prob)}\n")
                fp.close()
        else:
            with open(os.path.join(output_dir, fname), "w") as fp:
                fp.write("fname\tt0\tp_idx\tp_prob\ts_idx\ts_prob\n")
                for pick in picks:
                    fp.write(f"{pick.fname}\t{pick.t0}\t{int2s(pick.p_idx)}\t{flt2s(pick.p_prob)}\t{int2s(pick.s_idx)}\t{flt2s(pick.s_prob)}\n")
                fp.close()
    else:
        with open(os.path.join(output_dir, fname), "w") as fp:
            fp.write("fname\tt0\tp_idx\tp_prob\ts_idx\ts_prob\tp_amp\ts_amp\n")
            for pick, amp in zip(picks, amps):
                fp.write(f"{pick.fname}\t{pick.t0}\t{int2s(pick.p_idx)}\t{flt2s(pick.p_prob)}\t{int2s(pick.s_idx)}\t{flt2s(pick.s_prob)}\t{sci2s(amp.p_amp)}\t{sci2s(amp.s_amp)}\n")
            fp.close()

    return 0


def calc_timestamp(timestamp, sec):
    timestamp = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%f") + timedelta(seconds=sec)
    return timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

    
def save_picks_json(picks, output_dir, dt=0.01, amps=None, fname=None):
    if fname is None:
        fname = "picks.json"

    picks_ = []
    if amps is None:
        for pick in picks:
            for idxs, probs in zip(pick.p_idx, pick.p_prob):
                for idx, prob in zip(idxs, probs):
                    picks_.append({"id": pick.fname, 
                                "timestamp":calc_timestamp(pick.t0, float(idx)*dt), 
                                "prob": prob.astype(float), 
                                "type": "p"})
            for idxs, probs in zip(pick.s_idx, pick.s_prob):
                for idx, prob in zip(idxs, probs):
                    picks_.append({"id": pick.fname, 
                                "timestamp":calc_timestamp(pick.t0, float(idx)*dt), 
                                "prob": prob.astype(float), 
                                "type": "s"})
    else:
        for pick, amplitude in zip(picks, amps):
            for idxs, probs, amps in zip(pick.p_idx, pick.p_prob, amplitude.p_amp):
                for idx, prob, amp in zip(idxs, probs, amps):
                    picks_.append({"id": pick.fname, 
                                "timestamp":calc_timestamp(pick.t0, float(idx)*dt), 
                                "prob": prob.astype(float), 
                                "amp": amp.astype(float),
                                "type": "p"})
            for idxs, probs, amps in zip(pick.s_idx, pick.s_prob, amplitude.s_amp):
                for idx, prob, amp in zip(idxs, probs, amps):
                    picks_.append({"id": pick.fname, 
                                "timestamp":calc_timestamp(pick.t0, float(idx)*dt), 
                                "prob": prob.astype(float), 
                                "amp": amp.astype(float),
                                "type": "s"})
    with open(os.path.join(output_dir, fname), "w") as fp:
        json.dump(picks_, fp)

    return 0


def convert_true_picks(fname, itp, its, itps=None):
    true_picks = []
    if itps is None:
        record = namedtuple("phase", ["fname", "p_idx", "s_idx"])
        for i in range(len(fname)):
            true_picks.append(record(fname[i].decode(), itp[i], its[i]))
    else:
        record = namedtuple("phase", ["fname", "p_idx", "s_idx", "ps_idx"])
        for i in range(len(fname)):
            true_picks.append(record(fname[i].decode(), itp[i], its[i], itps[i]))

    return true_picks


def calc_metrics(nTP, nP, nT):
    '''
    nTP: true positive
    nP: number of positive picks
    nT: number of true picks
    '''
    precision = nTP / nP
    recall = nTP / nT
    f1 = 2* precision * recall / (precision + recall)
    return [precision, recall, f1]

def calc_performance(picks, true_picks, tol=3.0, dt=1.0):
    assert(len(picks) == len(true_picks))
    logging.info("Total records: {}".format(len(picks)))

    count = lambda picks: sum([len(x) for x in picks])
    metrics = {}
    for phase in true_picks[0]._fields:
        if phase == "fname":
            continue
        true_positive, positive, true = 0, 0, 0
        residual = []
        for i in range(len(true_picks)):
            true += count(getattr(true_picks[i], phase))
            positive += count(getattr(picks[i], phase))
            # print(i, phase, getattr(picks[i], phase), getattr(true_picks[i], phase))
            diff = dt*(np.array(getattr(picks[i], phase))[:,np.newaxis,:] - np.array(getattr(true_picks[i], phase))[:,:,np.newaxis])
            residual.extend(list(diff[np.abs(diff) <= tol]))
            true_positive += np.sum(np.abs(diff) <= tol)
        metrics[phase] = calc_metrics(true_positive, positive, true)

        logging.info(f"{phase}-phase:")
        logging.info(f"True={true}, Positive={positive}, True Positive={true_positive}")
        logging.info(f"Precision={metrics[phase][0]:.3f}, Recall={metrics[phase][1]:.3f}, F1={metrics[phase][2]:.3f}")
        logging.info(f"Residual mean={np.mean(residual):.4f}, std={np.std(residual):.4f}")

    return metrics


def save_prob_h5(probs, fnames, output_h5):
    if fnames is None:
        fnames = [f"{i:04d}" for i in range(len(probs))]
    elif type(fnames[0]) is bytes:
        fnames = [f.decode().rstrip(".npz") for f in fnames]
    else:
        fnames = [f.rstrip(".npz") for f in fnames]
    for prob, fname in zip(probs, fnames):
        output_h5.create_dataset(fname, data=prob, dtype="float32")
    return 0

def save_prob(probs, fnames, prob_dir):
    if fnames is None:
        fnames = [f"{i:04d}" for i in range(len(probs))]
    elif type(fnames[0]) is bytes:
        fnames = [f.decode().rstrip(".npz") for f in fnames]
    else:
        fnames = [f.rstrip(".npz") for f in fnames]
    for prob, fname in zip(probs, fnames):
        np.savez(os.path.join(prob_dir, fname+".npz"), prob=prob)
    return 0
