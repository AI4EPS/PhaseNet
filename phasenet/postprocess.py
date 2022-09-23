import json
import logging
import os
from collections import namedtuple
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
from detect_peaks import detect_peaks

# def extract_picks(preds, fnames=None, station_ids=None, t0=None, config=None):

#     if preds.shape[-1] == 4:
#         record = namedtuple("phase", ["fname", "station_id", "t0", "p_idx", "p_prob", "s_idx", "s_prob", "ps_idx", "ps_prob"])
#     else:
#         record = namedtuple("phase", ["fname", "station_id", "t0", "p_idx", "p_prob", "s_idx", "s_prob"])

#     picks = []
#     for i, pred in enumerate(preds):

#         if config is None:
#             mph_p, mph_s, mpd = 0.3, 0.3, 50
#         else:
#             mph_p, mph_s, mpd = config.min_p_prob, config.min_s_prob, config.mpd

#         if (fnames is None):
#             fname = f"{i:04d}"
#         else:
#             if isinstance(fnames[i], str):
#                 fname = fnames[i]
#             else:
#                 fname = fnames[i].decode()

#         if (station_ids is None):
#             station_id = f"{i:04d}"
#         else:
#             if isinstance(station_ids[i], str):
#                 station_id = station_ids[i]
#             else:
#                 station_id = station_ids[i].decode()

#         if (t0 is None):
#             start_time = "1970-01-01T00:00:00.000"
#         else:
#             if isinstance(t0[i], str):
#                 start_time = t0[i]
#             else:
#                 start_time = t0[i].decode()

#         p_idx, p_prob, s_idx, s_prob = [], [], [], []
#         for j in range(pred.shape[1]):
#             p_idx_, p_prob_ = detect_peaks(pred[:,j,1], mph=mph_p, mpd=mpd, show=False)
#             s_idx_, s_prob_ = detect_peaks(pred[:,j,2], mph=mph_s, mpd=mpd, show=False)
#             p_idx.append(list(p_idx_))
#             p_prob.append(list(p_prob_))
#             s_idx.append(list(s_idx_))
#             s_prob.append(list(s_prob_))

#         if pred.shape[-1] == 4:
#             ps_idx, ps_prob = detect_peaks(pred[:,0,3], mph=0.3, mpd=mpd, show=False)
#             picks.append(record(fname, station_id, start_time, list(p_idx), list(p_prob), list(s_idx), list(s_prob), list(ps_idx), list(ps_prob)))
#         else:
#             picks.append(record(fname, station_id, start_time, list(p_idx), list(p_prob), list(s_idx), list(s_prob)))

#     return picks


def extract_picks(
    preds,
    file_names=None,
    begin_times=None,
    station_ids=None,
    dt=0.01,
    phases=["P", "S"],
    config=None,
    waveforms=None,
    use_amplitude=False,
    upload_waveform=False,
):
    """Extract picks from prediction results.
    Args:
        preds ([type]): [Nb, Nt, Ns, Nc] "batch, time, station, channel"
        file_names ([type], optional): [Nb]. Defaults to None.
        station_ids ([type], optional): [Ns]. Defaults to None.
        t0 ([type], optional): [Nb]. Defaults to None.
        config ([type], optional): [description]. Defaults to None.

    Returns:
        picks [type]: {file_name, station_id, pick_time, pick_prob, pick_type}
    """

    mph = {}
    if config is None:
        for x in phases:
            mph[x] = 0.3
        mpd = 50
        ## upload waveform
        pre_idx = int(1 / dt)
        post_idx = int(4 / dt)
    else:
        mph["P"] = config.min_p_prob
        mph["S"] = config.min_s_prob
        mph["PS"] = 0.3
        mpd = config.mpd
        pre_idx = int(config.pre_sec / dt)
        post_idx = int(config.post_sec / dt)

    Nb, Nt, Ns, Nc = preds.shape

    if file_names is None:
        file_names = [f"{i:04d}" for i in range(Nb)]
    elif not (isinstance(file_names, np.ndarray) or isinstance(file_names, list)):
        if isinstance(file_names, bytes):
            file_names = file_names.decode()
        file_names = [file_names] * Nb
    else:
        file_names = [x.decode() if isinstance(x, bytes) else x for x in file_names]

    if begin_times is None:
        begin_times = ["1970-01-01T00:00:00.000+00:00"] * Nb
    else:
        begin_times = [x.decode() if isinstance(x, bytes) else x for x in begin_times]

    picks = []
    for i in range(Nb):

        file_name = file_names[i]
        begin_time = datetime.fromisoformat(begin_times[i])

        for j in range(Ns):
            if (station_ids is None) or (len(station_ids[i]) == 0):
                station_id = f"{j:04d}"
            else:
                station_id = station_ids[i].decode() if isinstance(station_ids[i], bytes) else station_ids[i]

            if (waveforms is not None) and use_amplitude:
                amp = np.max(np.abs(waveforms[i, :, j, :]), axis=-1)  ## amplitude over three channelspy
            for k in range(Nc - 1):  # 0-th channel noise
                idxs, probs = detect_peaks(preds[i, :, j, k + 1], mph=mph[phases[k]], mpd=mpd, show=False)
                for l, (phase_index, phase_prob) in enumerate(zip(idxs, probs)):
                    pick_time = begin_time + timedelta(seconds=phase_index * dt)
                    pick = {
                        "file_name": file_name,
                        "station_id": station_id,
                        "begin_time": begin_time.isoformat(timespec="milliseconds"),
                        "phase_index": int(phase_index),
                        "phase_time": pick_time.isoformat(timespec="milliseconds"),
                        "phase_score": round(phase_prob, 3),
                        "phase_type": phases[k],
                        "dt": dt,
                    }

                    ## process waveform
                    if waveforms is not None:
                        tmp = np.zeros((pre_idx + post_idx, 3))
                        lo = phase_index - pre_idx
                        hi = phase_index + post_idx
                        insert_idx = 0
                        if lo < 0:
                            lo = 0
                            insert_idx = -lo
                        if hi > Nt:
                            hi = Nt
                        tmp[insert_idx : insert_idx + hi - lo, :] = waveforms[i, lo:hi, j, :]
                        if upload_waveform:
                            pick["waveform"] = tmp.tolist()
                            pick["_id"] = f"{pick['station_id']}_{pick['timestamp']}_{pick['type']}"
                        if use_amplitude:
                            next_pick = idxs[l + 1] if l < len(idxs) - 1 else (phase_index + post_idx * 3)
                            pick["phase_amp"] = np.max(
                                amp[phase_index : min(phase_index + post_idx * 3, next_pick)]
                            ).item()  ## peak amplitude

                    picks.append(pick)

    return picks


def extract_amplitude(data, picks, window_p=10, window_s=5, config=None):
    record = namedtuple("amplitude", ["p_amp", "s_amp"])
    dt = 0.01 if config is None else config.dt
    window_p = int(window_p / dt)
    window_s = int(window_s / dt)
    amps = []
    for i, (da, pi) in enumerate(zip(data, picks)):
        p_amp, s_amp = [], []
        for j in range(da.shape[1]):
            amp = np.max(np.abs(da[:, j, :]), axis=-1)
            # amp = np.median(np.abs(da[:,j,:]), axis=-1)
            # amp = np.linalg.norm(da[:,j,:], axis=-1)
            tmp = []
            for k in range(len(pi.p_idx[j]) - 1):
                tmp.append(np.max(amp[pi.p_idx[j][k] : min(pi.p_idx[j][k] + window_p, pi.p_idx[j][k + 1])]))
            if len(pi.p_idx[j]) >= 1:
                tmp.append(np.max(amp[pi.p_idx[j][-1] : pi.p_idx[j][-1] + window_p]))
            p_amp.append(tmp)
            tmp = []
            for k in range(len(pi.s_idx[j]) - 1):
                tmp.append(np.max(amp[pi.s_idx[j][k] : min(pi.s_idx[j][k] + window_s, pi.s_idx[j][k + 1])]))
            if len(pi.s_idx[j]) >= 1:
                tmp.append(np.max(amp[pi.s_idx[j][-1] : pi.s_idx[j][-1] + window_s]))
            s_amp.append(tmp)
        amps.append(record(p_amp, s_amp))
    return amps


def save_picks(picks, output_dir, amps=None, fname=None):
    if fname is None:
        fname = "picks.csv"

    int2s = lambda x: ",".join(["[" + ",".join(map(str, i)) + "]" for i in x])
    flt2s = lambda x: ",".join(["[" + ",".join(map("{:0.3f}".format, i)) + "]" for i in x])
    sci2s = lambda x: ",".join(["[" + ",".join(map("{:0.3e}".format, i)) + "]" for i in x])
    if amps is None:
        if hasattr(picks[0], "ps_idx"):
            with open(os.path.join(output_dir, fname), "w") as fp:
                fp.write("fname\tt0\tp_idx\tp_prob\ts_idx\ts_prob\tps_idx\tps_prob\n")
                for pick in picks:
                    fp.write(
                        f"{pick.fname}\t{pick.t0}\t{int2s(pick.p_idx)}\t{flt2s(pick.p_prob)}\t{int2s(pick.s_idx)}\t{flt2s(pick.s_prob)}\t{int2s(pick.ps_idx)}\t{flt2s(pick.ps_prob)}\n"
                    )
                fp.close()
        else:
            with open(os.path.join(output_dir, fname), "w") as fp:
                fp.write("fname\tt0\tp_idx\tp_prob\ts_idx\ts_prob\n")
                for pick in picks:
                    fp.write(
                        f"{pick.fname}\t{pick.t0}\t{int2s(pick.p_idx)}\t{flt2s(pick.p_prob)}\t{int2s(pick.s_idx)}\t{flt2s(pick.s_prob)}\n"
                    )
                fp.close()
    else:
        with open(os.path.join(output_dir, fname), "w") as fp:
            fp.write("fname\tt0\tp_idx\tp_prob\ts_idx\ts_prob\tp_amp\ts_amp\n")
            for pick, amp in zip(picks, amps):
                fp.write(
                    f"{pick.fname}\t{pick.t0}\t{int2s(pick.p_idx)}\t{flt2s(pick.p_prob)}\t{int2s(pick.s_idx)}\t{flt2s(pick.s_prob)}\t{sci2s(amp.p_amp)}\t{sci2s(amp.s_amp)}\n"
                )
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
                    picks_.append(
                        {
                            "id": pick.station_id,
                            "timestamp": calc_timestamp(pick.t0, float(idx) * dt),
                            "prob": prob.astype(float),
                            "type": "p",
                        }
                    )
            for idxs, probs in zip(pick.s_idx, pick.s_prob):
                for idx, prob in zip(idxs, probs):
                    picks_.append(
                        {
                            "id": pick.station_id,
                            "timestamp": calc_timestamp(pick.t0, float(idx) * dt),
                            "prob": prob.astype(float),
                            "type": "s",
                        }
                    )
    else:
        for pick, amplitude in zip(picks, amps):
            for idxs, probs, amps in zip(pick.p_idx, pick.p_prob, amplitude.p_amp):
                for idx, prob, amp in zip(idxs, probs, amps):
                    picks_.append(
                        {
                            "id": pick.station_id,
                            "timestamp": calc_timestamp(pick.t0, float(idx) * dt),
                            "prob": prob.astype(float),
                            "amp": amp.astype(float),
                            "type": "p",
                        }
                    )
            for idxs, probs, amps in zip(pick.s_idx, pick.s_prob, amplitude.s_amp):
                for idx, prob, amp in zip(idxs, probs, amps):
                    picks_.append(
                        {
                            "id": pick.station_id,
                            "timestamp": calc_timestamp(pick.t0, float(idx) * dt),
                            "prob": prob.astype(float),
                            "amp": amp.astype(float),
                            "type": "s",
                        }
                    )
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
    """
    nTP: true positive
    nP: number of positive picks
    nT: number of true picks
    """
    precision = nTP / nP
    recall = nTP / nT
    f1 = 2 * precision * recall / (precision + recall)
    return [precision, recall, f1]


def calc_performance(picks, true_picks, tol=3.0, dt=1.0):
    assert len(picks) == len(true_picks)
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
            diff = dt * (
                np.array(getattr(picks[i], phase))[:, np.newaxis, :]
                - np.array(getattr(true_picks[i], phase))[:, :, np.newaxis]
            )
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
        np.savez(os.path.join(prob_dir, fname + ".npz"), prob=prob)
    return 0
