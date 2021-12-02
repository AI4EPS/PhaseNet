# Wrapping PhaseNet in python function

See below the content of the README of the original project.

In this fork, I introduce a new module, wrapper.py, to call PhaseNet from within a python script.


Example:

```python

    import os
    from phasenet import wrapper as PN

    # --------------------------
    #   load your data 
    #   data = (n_events, n_stations, 3, n_samples)
    #   data is a nd.array of multi-station 3-component seismograms
    #   of duration n_samples and from n_events different earthquakes
    # --------------------------

    data = some_function()
    station_names = ['name_st1', 'name_st2', ...]

    # -------------------------
    #   define where PhaseNet will write intermediate files
    # -------------------------
    
    # choose the current working directory
    PN_base = os.getcwd()
    # name of the folder where data files will be formatted for PhaseNet
    PN_dataset_name = 'experiment_1'

    # ------------------------
    #  some parameters for PhaseNet
    # ------------------------
    
    # number of 3-component seismograms given at once at PhaseNet
    # most likely, you can keep the value of 128 for your applications
    mini_batch_size = 128
    # detection threshold! here, I impose PhaseNet to predict probabilities
    # of P/S-wave arrivals of more than 0.6 to trigger a phase identification
    threshold_P = 0.6
    threshold_S = 0.6

    # ------------------------
    #    run PhaseNet!
    # ------------------------
    PhaseNet_probas, PhaseNet_picks = PN.automatic_picking(
        data, station_names, PN_base, PN_dataset_name)

    
    # ------------------------------------------------------
    # This is the end of the general use of my wrapper.
    # I use PhaseNet in template matching applications. The
    # following shows how to use extra features of wrapper.py.
    #
    # Let's assume that the n_events multi-station 3-comp seismograms
    # stored in the data array are recordings of similar earthquakes,
    # i.e. of earthquakes with similar expected picks.
    # ------------------------------------------------------

    sampling_rate = 100. # Hz
    # reformat picks and remove potential false picks at the beginning
    buffer_picks = np.int32(3.*sampling_rate)
    all_picks = PN.get_all_picks(PhaseNet_picks, buffer_picks)
    # because each event is supposed to produce the same P/S picks,
    # we can build an empirical distribution of the P/S picks on each station
    all_picks = PN.fit_probability_density(all_picks, overwrite=True)
    # we obtain a "composite" P/S pick on each station, and its uncertainty,
    # by analyzing the central tendency and deviation of the empirical distribution

    # clean up the picks by keeping the best ones
    # keep a composite pick if at least n_threshold picks contributed
    # to the empirical distribution
    n_threshold = 5
    # keep a composite pick if its uncertainty is less than 3 seconds
    err_threshold = 3.0 # in s
    err_threshold = np.int32(err_threshold*sampling_rate) # in samples
    selected_picks = PN.select_picks_family(
            all_picks, n_threshold, err_threshold, central='mean')
    # convert picks to seconds
    selected_picks = PN.convert_picks_to_sec(
            selected_picks, BPMF.cfg.sampling_rate)

    # plot the results
    # first, I assume that you have stored `data` in an obspy.Stream instance,
    # called `data_Stream`
    fig = PN.plot_picks(
            selected_picks, data_Stream, figname='test_PhaseNet', show=True)

```


# PhaseNet: A Deep-Neural-Network-Based Seismic Arrival Time Picking Method

[![](https://github.com/wayneweiqiang/PhaseNet/workflows/documentation/badge.svg)](https://wayneweiqiang.github.io/PhaseNet)


## 1.  Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) and requirements
- Download PhaseNet repository
```bash
git clone https://github.com/wayneweiqiang/PhaseNet.git
cd PhaseNet
```
- Install to default environment
```bash
conda env update -f=env.yml -n base
```
- Install to "phasenet" virtual environment
```bash
conda env create -f env.yml
conda activate phasenet
```

## 2. Pre-trained model
Located in directory: **model/190703-214543**

## 3. Related papers
- Zhu, Weiqiang, and Gregory C. Beroza. "PhaseNet: A Deep-Neural-Network-Based Seismic Arrival Time Picking Method." arXiv preprint arXiv:1803.03211 (2018).
- Liu, Min, et al. "Rapid characterization of the July 2019 Ridgecrest, California, earthquake sequence from raw seismic data using machine‐learning phase picker." Geophysical Research Letters 47.4 (2020): e2019GL086189.
- Park, Yongsoo, et al. "Machine‐learning‐based analysis of the Guy‐Greenbrier, Arkansas earthquakes: A tale of two sequences." Geophysical Research Letters 47.6 (2020): e2020GL087032.
- Chai, Chengping, et al. "Using a deep neural network and transfer learning to bridge scales for seismic phase picking." Geophysical Research Letters 47.16 (2020): e2020GL088651.
- Tan, Yen Joe, et al. "Machine‐Learning‐Based High‐Resolution Earthquake Catalog Reveals How Complex Fault Structures Were Activated during the 2016–2017 Central Italy Sequence." The Seismic Record 1.1 (2021): 11-19.

## 4. Interactive example
See details in the [notebook](https://github.com/wayneweiqiang/PhaseNet/blob/master/docs/example_interactive.ipynb): [example_interactive.ipynb](example_interactive.ipynb)


## 5. Batch prediction
See details in the [notebook](https://github.com/wayneweiqiang/PhaseNet/blob/master/docs/example_batch_prediction.ipynb): [example_batch_prediction.ipynb](example_batch_prediction.ipynb)

## 6. Training

Please let us know of any bugs found in the code. Suggestions and collaborations are welcomed

