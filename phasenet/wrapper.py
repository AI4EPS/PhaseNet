import os

import numpy as np
import h5py as h5
import glob

from .data_reader import DataReader_pred
from .predict_fn import pred_fn

script_path = os.path.dirname(os.path.realpath(__file__))

def format_data(data, root_PN_inputs='.', filename='data.h5'):
    """Format data for PhasetNet.  

    Save the data array in npz files such that PhaseNet can process it.

    Parameters
    -------------
    data: (n_stations, 3, n_samples) nd.array
        Numpy array with the continuous 3-component seismic data
        on which we want to pick the P- and S-wave arrivals.
    root_PN_inputs: string, default to '.'
        Path to the root folder where formatted data will be stored.
    filename: string, default to 'data.h5'
        Name of the file listing the filenames of all 3-component
        time series to process.
    """
    import h5py as h5
    with h5.File(os.path.join(root_PN_inputs, filename), 'w') as f:
        f.create_group('data')
        for i in range(data.shape[0]):
            # place the component axis at the end
            three_comp_data = np.swapaxes(data[i, ...], 0, 1)
            f['data'].create_dataset(f'sample{i}', data=three_comp_data)

def run_pred(input_length,
             model_path=f'/home/ebeauce/PhaseNet/model/190703-214543',
             data_path='./dataset/waveform_pred/',
             log_dir='./dataset/log/',
             data_file='./dataset/data.h5',
             hdf5_group='data',
             amplitude=False,
             batch_size=1,
             threshold_P=0.6,
             threshold_S=0.6):
    """Run PhaseNet and fetch its raw output: the P and S probabilities.

    Results are stored at the user-defined location `output_filename`.

    Parameters
    ------------
    input_length: int
        Duration, in samples, of the 3-component seismograms.
    model_path: string, default to '/home/ebeauce/PhaseNet/model/190703-214543'
        Path to the trained model. It is of course necessary to change the
        default value to the adequate value on your machine (e.g. where you
        downloaded PhaseNet).
    data_path: string, default to './dataset/waveform_pred/'
        Path to the folder with the 3-component seismograms in npz files.
    log_dir: string, default to './dataset/log/'
    data_list: string, default to './dataset/data_list.csv'
    output_filename: string, default to './prediction.npy'
        Name of the file with PhaseNet's outputs.
    batch_size: int, default to 1
        Number of 3-component seismograms processed by PhaseNet
        at once. This should to take into account the machine's RAM.
    threshold_P: float, default to 0.6
        P-wave identification threshold. When PhaseNet's raw output
        (proba) exceeds `threshold_P`, a detection is triggered.
    threshold_S: float, default to 0.6
        S-wave identification threshold. When PhaseNet's raw output
        (proba) exceeds `threshold_S`, a detection is triggered.
    """
    data_reader = DataReader_pred(
        format='hdf5',
        data_dir=data_path,
        data_list='', # not used with hdf5 format
        hdf5_file=data_file,
        hdf5_group=hdf5_group,
        amplitude=amplitude)
    PhaseNet_proba, PhaseNet_picks = pred_fn(
            data_reader, model_dir=model_path, log_dir=log_dir,
            batch_size=batch_size, input_length=input_length,
            min_p_prob=threshold_P, min_s_prob=threshold_S)
    return PhaseNet_proba, PhaseNet_picks


def load_PN_output(path_to_file, return_picks=False):
    """Read PhaseNet output.

    Parameters
    -------------
    path_to_file: string
        Path to the file storing the output.
    return_picks: boolean, default to False
        If `True`, this function returns not only the P- and S-wave
        arrival probabilities, but also the picks. The picks are obtained
        by postprocessing PhaseNet's raw ouput.

    Returns
    ----------
    PhaseNet_proba: (n_events, n_stations, input_length, 2) nd.array
        Numpy array with P- and S-wave arrival probabilities given as time
        series of same length than the input seismograms.  
        `PhaseNet_proba[..., 0]` is the P-wave probability.  
        `PhaseNet_proba[..., 1]` is the S-wave probability.
    PhaseNet_picks: (n_events, n_stations, 2, 2) nd.array of arrays (!), optional
        `PhaseNet_picks[n, s, 0, 0]` is the array of all P-wave picks, in samples  
        `PhaseNet_picks[n, s, 0, 1]` is the array of all P-wave pick proba
        (it is a measure of how reliable the picks are)  
        `PhaseNet_picks[n, s, 1, 0]` is the array of all S-wave picks, in samples  
        `PhaseNet_picks[n, s, 1, 1]` is the array of all S-wave pick proba
        (it is a measure of how reliable the picks are)  
    """
    PhaseNet_output = dict(np.load(path_to_file, allow_pickle=True))
    PhaseNet_output['filename'] = PhaseNet_output['filename'].astype('U').tolist()
    n_data_points = PhaseNet_output['probabilities'].shape[0]
    PhaseNet_proba = np.zeros_like(PhaseNet_output['probabilities'])
    if return_picks:
        PhaseNet_picks = np.zeros_like(PhaseNet_output['picks'])
    for i in range(n_data_points):
        ordered_idx = PhaseNet_output['filename'].index(f'sample{i}.npz')
        PhaseNet_proba[i, ...] = \
                PhaseNet_output['probabilities'][ordered_idx, ...]
        PhaseNet_picks[i, ...] = \
                PhaseNet_output['picks'][ordered_idx, ...]
    if return_picks:
        return PhaseNet_proba, PhaseNet_picks
    else:
        return PhaseNet_proba

def automatic_picking(data,
                      station_names,
                      PN_base,
                      PN_dataset_name,
                      mini_batch_size=126,
                      threshold_P=0.6,
                      threshold_S=0.6):
    """Wrapper function to call PhaseNet from a python script.  


    Parameters
    -----------
    data: (n_events, n_stations, 3, n_samples) nd.array
        Numpy array with the continuous 3-component seismograms of
        `n_events` earthquakes recorded at a network of `n_stations`
        stations.
    station_names: list or array of strings
        Name of the `n_stations` stations of the array, in the same
        order as given in `data`.
    PN_base: string
        Path to the root folder where PhaseNet formatted data will
        be stored.
    PN_dataset_name: string
        Name of the folder, inside `PN_base`, where the formatted data
        of a given experiment will be stored.
    mini_batch_size: int, default to 126
        Number of 3-component seismograms processed by PhaseNet
        at once. This should to take into account the machine's RAM.
    threshold_P: float, default to 0.6
        P-wave identification threshold. When PhaseNet's raw output
        (proba) exceeds `threshold_P`, a detection is triggered.
    threshold_S: float, default to 0.6
        S-wave identification threshold. When PhaseNet's raw output
        (proba) exceeds `threshold_S`, a detection is triggered.

    Returns
    ---------

    """

    if not os.path.isdir(PN_base):
        print(f'Creating the formatted data root folder at {PN_base}')
        os.mkdir(PN_base)
    # clean up input/output directories if necessary
    root_PN_inputs = os.path.join(PN_base, PN_dataset_name)
    path_waveforms = os.path.join(root_PN_inputs, 'waveform_pred')
    if os.path.isdir(path_waveforms):
        # clean up the directory
        for file_ in glob.glob(os.path.join(path_waveforms, '*')):
            os.remove(file_)
    else:
        try:
            os.mkdir(path_waveforms)
        except:
            print('Have you created the root folder where you want '
                    'PhaseNet inputs to be stored? It should be: {}'.
                    format(root_PN_inputs))
            return

    # assume the data were provided in the shape
    # (n_events x n_stations x 3-comp x time_duration)
    n_events = data.shape[0]
    n_stations = data.shape[1]
    input_length = data.shape[3]
    # for efficiency, we merge the event and the station axes
    batch_size = n_events*n_stations
    print('n events: {:d}, n stations: {:d}, batch size (n events x n stations): {:d}'.
            format(n_events, n_stations, batch_size))
    data = data.reshape(batch_size, 3, input_length)
    # make sure the minibatch size is not larger than the
    # total number of traces
    minibatch_size = min(mini_batch_size, batch_size)

    # generate the input files necessary for PhaseNet
    format_data(data, root_PN_inputs=root_PN_inputs)
    # call PhaseNet
    PhaseNet_proba, PhaseNet_picks = run_pred(
            input_length,
            data_path=os.path.join(PN_base, PN_dataset_name, 'waveform_pred'),
            data_file=os.path.join(PN_base, PN_dataset_name, 'data.h5'),
            log_dir=os.path.join(PN_base, PN_dataset_name, 'log'),
            batch_size=mini_batch_size,
            threshold_P=threshold_P,
            threshold_S=threshold_S)
    # the new PhaseNet_proba is an array of time series with [..., 0] = proba of P arrival
    # and [..., 1] = proba of S arrival (the original [..., 0] was simply 1 - Pp - Ps)
    PhaseNet_proba = PhaseNet_proba.reshape((n_events, n_stations, input_length, 3))[..., 1:]
    PhaseNet_picks = PhaseNet_picks.reshape((n_events, n_stations, 2, 2))
    # return picks in a comprehensive python dictionary
    picks = {}
    picks['P_picks'] = {}
    picks['P_proba'] = {}
    picks['S_picks'] = {}
    picks['S_proba'] = {}
    for s in range(n_stations):
        # (n_events, arrays): array of arrays with all detected P-arrival picks 
        picks['P_picks'][station_names[s]] = PhaseNet_picks[:, s, 0, 0]
        # (n_events, arrays): array of arrays with probabilities of all detected P-arrival picks 
        picks['P_proba'][station_names[s]] = PhaseNet_picks[:, s, 0, 1]
        # (n_events, arrays): array of arrays with all detected S-arrival picks 
        picks['S_picks'][station_names[s]] = PhaseNet_picks[:, s, 1, 0]
        # (n_events, arrays): array of arrays with probabilities of all detected S-arrival picks 
        picks['S_proba'][station_names[s]] = PhaseNet_picks[:, s, 1, 1]
    return PhaseNet_proba, picks


# --------------------------------------------------------------------------------
#     The following functions were tailored for template matching applications
# --------------------------------------------------------------------------------


def get_best_picks(picks, buffer_length=50):
    """Filter picks to keep the best one on each 3-comp seismogram.


    """
    for st in picks['P_picks'].keys():
        for n in range(len(picks['P_picks'][st])):
            pp = picks['P_picks'][st][n]
            ps = picks['S_picks'][st][n]
            # ----------------
            # remove picks form the buffer length
            valid_P_picks = picks['P_picks'][st][n] > int(buffer_length)
            valid_S_picks = picks['S_picks'][st][n] > int(buffer_length)
            picks['P_picks'][st][n] = picks['P_picks'][st][n][valid_P_picks]
            picks['S_picks'][st][n] = picks['S_picks'][st][n][valid_S_picks]
            picks['P_proba'][st][n] = picks['P_proba'][st][n][valid_P_picks]
            picks['S_proba'][st][n] = picks['S_proba'][st][n][valid_S_picks]
            # take only the highest probability trigger
            if len(picks['S_picks'][st][n]) > 0:
                best_S_trigger = picks['S_proba'][st][n].argmax()
                picks['S_picks'][st][n] = picks['S_picks'][st][n][best_S_trigger]
                picks['S_proba'][st][n] = picks['S_proba'][st][n][best_S_trigger]
                # update P picks: keep only those that are before the best S pick
                valid_P_picks = picks['P_picks'][st][n] < picks['S_picks'][st][n]
                picks['P_picks'][st][n] = picks['P_picks'][st][n][valid_P_picks]
                picks['P_proba'][st][n] = picks['P_proba'][st][n][valid_P_picks]
            else:
                # if no valid S pick: fill in with nan
                picks['S_picks'][st][n] = np.nan
                picks['S_proba'][st][n] = np.nan
            if len(picks['P_picks'][st][n]) > 0:
                best_P_trigger = picks['P_proba'][st][n].argmax()
                picks['P_picks'][st][n] = picks['P_picks'][st][n][best_P_trigger]
                picks['P_proba'][st][n] = picks['P_proba'][st][n][best_P_trigger]
            else:
                # if no valid P pick: fill in with nan
                picks['P_picks'][st][n] = np.nan
                picks['P_proba'][st][n] = np.nan
        # convert picks to float to allow NaNs
        picks['P_picks'][st] = np.float32(picks['P_picks'][st])
        picks['S_picks'][st] = np.float32(picks['S_picks'][st])
        picks['P_proba'][st] = np.float32(picks['P_proba'][st])
        picks['S_proba'][st] = np.float32(picks['S_proba'][st])
    return picks

def get_all_picks(picks, buffer_length=50):
    """Combine all picks from multiple events (1 station) in one array.  

    This function makes sense when the (n_events, n_stations, n_components,
    n_samples) `data` array given to `automatic_picking` is an array of
    `n_events` similar earthquakes (i.e. similar locations, and therefore
    similar expected picks).  
    Then, each station has potentially many P-wave and S-wave
    picks with which we can define a mean value and an error (see 
    `fit_probability_density`).

    Parameters
    ---------------
    picks: dictionary
        Picks returned by `automatic_picking`.
    buffer_length: int, default to 50
        Due to some edge effects, PhaseNet tends to trigger false detections
        at the beginning of a 3-comp seismogram. `buffer_length` is the time,
        in samples, to ignore at the beginning.

    Returns
    -----------
    picks: dictionary
        A dictionary with 4 fields: `P_picks`, 'S_picks', 'P_proba',
        'S_proba', and each of these fields is itself a dictionary for one
        entry for each station.  
        Example: picks['P_picks']['station1'] = [124, 123, 126, 250] means that
        4 P-wave picks were identified on station1, with possibly one outlier at
        sample 250.
    """
    for st in picks['P_picks'].keys():
        P_picks = []
        P_proba = []
        S_picks = []
        S_proba = []
        for n in range(len(picks['P_picks'][st])):
            pp = picks['P_picks'][st][n]
            ps = picks['S_picks'][st][n]
            # ----------------
            # remove picks from the buffer length
            valid_P_picks = picks['P_picks'][st][n] > int(buffer_length)
            valid_S_picks = picks['S_picks'][st][n] > int(buffer_length)
            picks['P_picks'][st][n] = picks['P_picks'][st][n][valid_P_picks]
            picks['S_picks'][st][n] = picks['S_picks'][st][n][valid_S_picks]
            picks['P_proba'][st][n] = picks['P_proba'][st][n][valid_P_picks]
            picks['S_proba'][st][n] = picks['S_proba'][st][n][valid_S_picks]
            # take all picks
            P_picks.extend(picks['P_picks'][st][n].tolist())
            P_proba.extend(picks['P_proba'][st][n].tolist())
            S_picks.extend(picks['S_picks'][st][n].tolist())
            S_proba.extend(picks['S_proba'][st][n].tolist())
        picks['P_picks'][st] = np.int32(P_picks)
        picks['S_picks'][st] = np.int32(S_picks)
        picks['P_proba'][st] = np.float32(P_proba)
        picks['S_proba'][st] = np.float32(S_proba)
    return picks

def fit_probability_density(picks, overwrite=False):
    """Estimate pdf of pick distribution.  

    When multiple picks of the same (or similar) arrival time
    are available, their empirical distribution can be used to
    quantify uncertainties on the estimate of this arrival time.
    The pdf is estimated with the kernel density method from scikit-learn.

    Parameters
    -----------
    picks: dictionary
        Picks returned by `automatic_detection` and processed by
        `get_all_picks`.
    overwrite: boolean, default to False
        If True, substitute PhaseNet probas in picks['P/S_proba']['stationXX']
        by the pdf values.

    Returns
    ---------
    picks: dictionary
        Input dictionary updated with the new field 'P/S_kde', which is the
        kernel estimate of the pdf. If `overwrite` is True, 'P/S_proba' is 
        also equal to 'P/S_kde'.
    """
    from sklearn.neighbors import KernelDensity
    # estimate probability density with gaussian kernels
    # of bandwidth 5 (in samples)
    kde = KernelDensity(kernel='gaussian', bandwidth=5)
    # update dictionary
    picks['P_kde'] = {}
    picks['S_kde'] = {}
    for st in picks['P_picks'].keys():
        if len(picks['P_picks'][st]) > 0:
            kde.fit(picks['P_picks'][st].reshape(-1, 1), sample_weight=picks['P_proba'][st])
            log_proba_samples = kde.score_samples(picks['P_picks'][st].reshape(-1, 1))
            picks['P_kde'][st] = np.exp(log_proba_samples).squeeze()
            if overwrite:
                picks['P_proba'][st] = np.array(picks['P_kde'][st], ndmin=1)
        # ---------------------------
        if len(picks['S_picks'][st]) > 0:
            kde.fit(picks['S_picks'][st].reshape(-1, 1), sample_weight=picks['S_proba'][st])
            log_proba_samples = kde.score_samples(picks['S_picks'][st].reshape(-1, 1))
            picks['S_kde'][st] = np.exp(log_proba_samples).squeeze()
            if overwrite:
                picks['S_proba'][st] = np.array(picks['S_kde'][st], ndmin=1)
    return picks


def select_picks_family(picks, n_threshold, err_threshold, central='mode'):
    """Filter picks based on their quality.  

    After processing by `fit_probability_density`, the quality of a
    given P/S composite pick can be evaluated by the number of individual
    picks that went into estimating its pdf, and the level of error indicated
    by the pdf.

    Parameters
    ------------
    picks: dictionary
        Picks returned by `automatic_detection`, processed by
        `get_all_picks` and `fit_probability_density`.
    n_threshold: scalar, int
        Keep composite picks whose pdf was estimated on
        N >= `n_threshold` individual picks.
    err_threshold: scalar, int or float
        Keep composite picks whose pdf indicates an error (see `central`)
        lower than `err_threshold`.
    central: string, default to 'mode'
        The central tendency used in the computation of the error. It should be
        either 'mode' or 'mean'. The error is taken as the RMS deviation about
        the central tendency.

    Returns
    ----------
    selected_picks: dictionary
        The picks filtered according the quality criteria `n_threshold` and
        `err_threshold`.
    """
    n_threshold = max(1, n_threshold)
    picks_p = {}
    picks_s = {}
    for st in picks['P_picks'].keys():
        pp = picks['P_picks'][st]
        ppb = picks['P_proba'][st]
        # remove the invalid picks
        valid = ~np.isnan(pp)
        pp = pp[valid]
        ppb = ppb[valid]
        if len(pp) < n_threshold:
            continue
        if central == 'mode':
            # take the most likely value as the estimate of the pick
            central_tendency = pp[ppb.argmax()]
        elif central == 'mean':
            central_tendency = np.sum(np.float32(pp)*ppb)/np.sum(ppb)
        else:
            print('central should be either mean or mode!')
            return
        # estimate the dispersion around this value
        err = np.sqrt(np.sum(ppb*(pp - central_tendency)**2/ppb.sum()))
        if err > err_threshold:
            continue
        picks_p[st] = np.float32([central_tendency, err])
    for st in picks['S_picks'].keys():
        sp = picks['S_picks'][st]
        spb = picks['S_proba'][st]
        # remove the invalid picks
        valid = ~np.isnan(sp)
        sp = sp[valid]
        spb = spb[valid]
        if len(sp) < n_threshold:
            continue
        if central == 'mode':
            # take the most likely value as the estimate of the pick
            central_tendency = sp[spb.argmax()]
        elif central == 'mean':
            central_tendency = np.sum(np.float32(sp)*spb)/np.sum(spb)
        else:
            print('central should be either mean or mode!')
            return
        # estimate the dispersion around this value
        err = np.sqrt(np.sum(spb*(sp - central_tendency)**2/spb.sum()))
        if err > err_threshold:
            continue
        picks_s[st] = np.float32([central_tendency, err])
    selected_picks = {}
    selected_picks['P_picks'] = picks_p
    selected_picks['S_picks'] = picks_s
    # picks are expressed in samples!
    return selected_picks


def plot_picks(picks, data_stream, figname=None, show=False, figsize=(20, 10)):
    """Plot the picks returned by `automatic_picking`.  

    Parameters
    -------------
    picks: dictionary
        Picks returned by `automatic_detection`, processed by `get_all_picks`,
        `fit_probability_density` and `select_picks_family`.
    data_stream: `obspy.Stream`
        Multi-station 3-component seismograms stored in an `obspy.Stream` object.
    figname: string, default to None
        Name of the `matplotlib.pyplot.Figure` instance.
    show: boolean, default to False
        If True, call `matplotlib.pyplot.show()`.
    figsize: 2-tuple of ints, default to (20, 10)
        Size of the `matplotlib.pyplot.Figure` instance in inches
        (width, height).

    Returns
    ----------
    fig: `matplotlib.pyplot.Figure`
        The `matplotlib.pyplot.Figure` instance created in this function.
    """
    import matplotlib.pyplot as plt
    old_params = plt.rcParams.copy()
    plt.rcParams.update({'ytick.labelsize'  :  10})
    plt.rcParams.update({'legend.fontsize': 7})
    # --------------------------
    stations = list(set(list(picks['P_picks'].keys()) \
                      + list(picks['S_picks'].keys())))
    sr = data_stream[0].stats.sampling_rate
    components = ['N', 'E', 'Z']
    n_components = len(components)
    # --------------------------
    time = np.linspace(
            0., data_stream[0].stats.npts/sr, data_stream[0].stats.npts)
    fig, axes = plt.subplots(
            num=figname, figsize=figsize, nrows=len(stations), ncols=n_components)
    for s in range(len(stations)):
        for c in range(n_components):
            ax = axes[s, c]
            try:
                ax.plot(time, data_stream.select(
                    station=stations[s], component=components[c])[0].data,
                        color='k', lw=0.75, label=f'{stations[s]}.{components[c]}')
            except IndexError:
                # no data
                continue
            ax.legend(loc='upper right', fancybox=True, handlelength=0.2, borderpad=0.1)
            if stations[s] in picks['P_picks'].keys():
                ax.axvline(
                        picks['P_picks'][stations[s]][0], color='C0', lw=1.0)
                xmin = picks['P_picks'][stations[s]][0]\
                        - picks['P_picks'][stations[s]][1]
                xmax = picks['P_picks'][stations[s]][0]\
                        + picks['P_picks'][stations[s]][1]
                ymin, ymax = ax.get_ylim()
                ax.fill([xmin, xmin, xmax, xmax],
                        [ymin, ymax, ymax, ymin],
                        color='C0', alpha=0.5, zorder=-1)
            if stations[s] in picks['S_picks'].keys():
                ax.axvline(
                        picks['S_picks'][stations[s]][0], color='C3', lw=1.0)
                xmin = picks['S_picks'][stations[s]][0]\
                        - picks['S_picks'][stations[s]][1]
                xmax = picks['S_picks'][stations[s]][0]\
                        + picks['S_picks'][stations[s]][1]
                ymin, ymax = ax.get_ylim()
                ax.fill([xmin, xmin, xmax, xmax],
                        [ymin, ymax, ymax, ymin],
                        color='C3', alpha=0.5, zorder=-1)
            ax.set_xlim(time.min(), time.max())
            ax.set_yticks([])
            if s < len(stations)-1:
                plt.setp(ax.get_xticklabels(), visible=False)
            else:
                ax.set_xlabel('Time (s)')
    plt.subplots_adjust(
            top=0.98, bottom=0.06, left=0.02,
            right=0.98, hspace=0.)
    if show:
        plt.show()
    plt.rcParams = old_params
    return fig


def select_picks_single_event(picks, event_id, uncertainty=5):
    picks_p = {}
    picks_s = {}
    for st in picks['P_picks'].keys():
        pp = picks['P_picks'][st][event_id]
        if np.isnan(pp):
            continue
        picks_p[st] = np.int32([pp, uncertainty])
    for st in picks['S_picks'].keys():
        sp = picks['S_picks'][st][event_id]
        if np.isnan(sp):
            continue
        picks_s[st] = np.int32([sp, uncertainty])
    selected_picks = {}
    selected_picks['P_picks'] = picks_p
    selected_picks['S_picks'] = picks_s
    # picks are expressed in samples!
    return selected_picks

def keep_only_PS(picks):
    p_stations = list(picks['P_picks'].keys())
    s_stations = list(picks['S_picks'].keys())
    for st in p_stations:
        # delete P stations that are
        # not in S stations
        if st not in s_stations:
            del picks['P_picks'][st]
    # update station lists
    p_stations = list(picks['P_picks'].keys())
    s_stations = list(picks['S_picks'].keys())
    for st in s_stations:
        # delete S stations that are
        # not in P stations
        if st not in p_stations:
            del picks['S_picks'][st]
    return picks

def convert_picks_to_sec(picks,
                         sampling_rate):
    for st in picks['P_picks'].keys():
        picks['P_picks'][st] = np.float32(picks['P_picks'][st])/sampling_rate
    for st in picks['S_picks'].keys():
        picks['S_picks'][st] = np.float32(picks['S_picks'][st])/sampling_rate
    return picks

def save_picks(picks,
               filename,
               path=''):

    # picks should be expressed in seconds!

    with h5.File(os.path.join(path, filename), 'w') as f:
        f.create_group('P')
        for st in picks['P_picks'].keys():
            f['P'].create_dataset(st, data=np.float32(picks['P_picks'][st]))
        f.create_group('S')
        for st in picks['S_picks'].keys():
            f['S'].create_dataset(st, data=np.float32(picks['S_picks'][st]))
        if 'origin_time' in picks.keys():
            # picks for an actual event are
            # associated to an origin time
            f.create_dataset('origin_time', data=picks['origin_time'])

def read_picks(filename,
               path=''):

    picks = {}
    picks['P_picks'] = {}
    picks['S_picks'] = {}
    with h5.File(os.path.join(path, filename), 'r') as f:
        for key in f['P'].keys():
            picks['P_picks'][key] = f['P'][key][()]
        for key in f['S'].keys():
            picks['S_picks'][key] = f['S'][key][()]
        if 'origin_time' in f.keys():
            picks['origin_time'] = f['origin_time'][()]
    return picks

