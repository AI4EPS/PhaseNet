
import os
import numpy as np
from collections import namedtuple
from detect_peaks import detect_peaks


def extract_picks(preds, fnames=None, t0=None, config=None):

    if preds.shape[-1] == 4:
        record = namedtuple("phase", ["fname", "t0", "idx_p", "prob_p", "idx_s", "prob_s", "idx_ps", "prob_ps"])
    else:
        record = namedtuple("phase", ["fname", "t0", "idx_p", "prob_p", "idx_s", "prob_s"])

    picks = []
    for i, pred in enumerate(preds):

        if config is None:
            mph_p, mph_s, mpd = 0.3, 0.3, 30
        else:
            mph_p, mph_s, mpd = config.min_prob_p, config.min_prob_s, 0.5/config.dt

        if (fnames is None):
            fname = f"{i:04d}"
        else:
            fname = fnames[i].decode()
            
        if (t0 is None):
            start_time = "0"
        else:
            start_time = t0[i].decode()

        idx_p, prob_p, idx_s, prob_s = [], [], [], []
        for j in range(pred.shape[1]):
            idx_p_, prob_p_ = detect_peaks(pred[:,j,1], mph=mph_p, mpd=mpd, show=False)
            idx_s_, prob_s_ = detect_peaks(pred[:,j,2], mph=mph_s, mpd=mpd, show=False)
            idx_p.append(list(idx_p_))
            prob_p.append(list(prob_p_))
            idx_s.append(list(idx_s_))
            prob_s.append(list(prob_s_))

        if pred.shape[-1] == 4:
            idx_ps, prob_ps = detect_peaks(pred[:,0,3], mph=mph, mpd=mpd, show=False)
            picks.append(record(fname, start_time, list(idx_p), list(prob_p), list(idx_s), list(prob_s), list(idx_ps), list(prob_ps)))
        else:
            picks.append(record(fname, start_time, list(idx_p), list(prob_p), list(idx_s), list(prob_s)))

    return picks


def extract_amplitude(data, picks, window_p=8, window_s=5, config=None):
    record = namedtuple("amplitude", ["amp_p", "amp_s"])
    dt = 0.01 if config is None else config.dt
    window_p = int(window_p/dt)
    window_s = int(window_s/dt)
    amps = []
    for i, (da, pi) in enumerate(zip(data, picks)):
        amp_p, amp_s = [], []
        for j in range(da.shape[1]):
            amp = np.max(np.abs(da[:,j,:]), axis=-1)
            tmp = []
            for k in range(len(pi.idx_p[j])-1):
                tmp.append(np.max(amp[pi.idx_p[j][k]:min(pi.idx_p[j][k]+window_p, pi.idx_p[j][k+1])]))
            if len(pi.idx_p[j]) >= 1:
                tmp.append(np.max(amp[pi.idx_p[j][-1]:pi.idx_p[j][-1]+window_p]))
            amp_p.append(tmp)
            tmp = []
            for k in range(len(pi.idx_s[j])-1):
                tmp.append(np.max(amp[pi.idx_s[j][k]:min(pi.idx_s[j][k]+window_s, pi.idx_s[j][k+1])]))
            if len(pi.idx_s[j]) >= 1:
                tmp.append(np.max(amp[pi.idx_s[j][-1]:pi.idx_s[j][-1]+window_s]))
            amp_s.append(tmp)
        amps.append(record(amp_p, amp_s))
    return amps

def save_picks(picks, output_dir, amps=None):

    int2s = lambda x: ",".join(["["+",".join(map(str, i))+"]" for i in x])
    flt2s = lambda x: ",".join(["["+",".join(map("{:0.3f}".format, i))+"]" for i in x])
    sci2s = lambda x: ",".join(["["+",".join(map("{:0.3e}".format, i))+"]" for i in x])
    if amps is None:
        if hasattr(picks[0], "idx_ps"):
            with open(os.path.join(output_dir, "picks.csv"), "w") as fp:
                fp.write("fname\tt0\tidx_p\tprob_p\tidx_s\tprob_s\tidx_ps\tprob_ps\n")
                for pick in picks:
                    fp.write(f"{pick.fname}\t{pick.t0}\t{int2s(pick.idx_p)}\t{flt2s(pick.prob_p)}\t{int2s(pick.idx_s)}\t{flt2s(pick.prob_s)}\t{int2s(pick.idx_ps)}\t{flt2s(pick.prob_ps)}\n")
                fp.close()
        else:
            with open(os.path.join(output_dir, "picks.csv"), "w") as fp:
                fp.write("fname\tt0\tidx_p\tprob_p\tidx_s\tprob_s\n")
                for pick in picks:
                    fp.write(f"{pick.fname}\t{pick.t0}\t{int2s(pick.idx_p)}\t{flt2s(pick.prob_p)}\t{int2s(pick.idx_s)}\t{flt2s(pick.prob_s)}\n")
                fp.close()
    else:
        with open(os.path.join(output_dir, "picks.csv"), "w") as fp:
            fp.write("fname\tt0\tidx_p\tprob_p\tidx_s\tprob_s\tamp_p\tamp_s\n")
            for pick, amp in zip(picks, amps):
                fp.write(f"{pick.fname}\t{pick.t0}\t{int2s(pick.idx_p)}\t{flt2s(pick.prob_p)}\t{int2s(pick.idx_s)}\t{flt2s(pick.prob_s)}\t{sci2s(amp.amp_p)}\t{sci2s(amp.amp_s)}\n")
            fp.close()

    return 0

