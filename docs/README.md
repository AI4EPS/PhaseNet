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
- Install to "phasenet" virtual envirionment
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

## 4. Batch prediction
See examples in the [notebook](https://github.com/wayneweiqiang/PhaseNet/blob/master/docs/example_batch_prediction.ipynb): [example_batch_prediction.ipynb](example_batch_prediction.ipynb)


PhaseNet currently supports four data formats: mseed, sac, hdf5, and numpy. The test data can be downloaded here:
```
wget https://github.com/wayneweiqiang/PhaseNet/releases/download/test_data/test_data.zip
unzip test_data.zip
```

- For mseed format:
```
python phasenet/predict.py --model=model/190703-214543 --data_list=test_data/mseed.csv --data_dir=test_data/mseed --format=mseed --plot_figure
```

- For sac format:
```
python phasenet/predict.py --model=model/190703-214543 --data_list=test_data/sac.csv --data_dir=test_data/sac --format=sac --plot_figure
```

- For numpy format:
```
python phasenet/predict.py --model=model/190703-214543 --data_list=test_data/npz.csv --data_dir=test_data/npz --format=numpy --plot_figure
```

- For hdf5 format:
```
python phasenet/predict.py --model=model/190703-214543 --hdf5_file=test_data/data.h5 --hdf5_group=data --format=hdf5 --plot_figure
```

- For a seismic array (used by [QuakeFlow](https://github.com/wayneweiqiang/QuakeFlow)):
```
python phasenet/predict.py --model=model/190703-214543 --data_list=test_data/mseed_array.csv --data_dir=test_data/mseed_array --stations=test_data/stations.json  --format=mseed_array --amplitude
```
```
python phasenet/predict.py --model=model/190703-214543 --data_list=test_data/mseed2.csv --data_dir=test_data/mseed --stations=test_data/stations.json  --format=mseed_array --amplitude
```

Notes: 
1. Remove the "--plot_figure" argument for large datasets, because plotting can be very slow.

Optional arguments:
```
usage: predict.py [-h] [--batch_size BATCH_SIZE] [--model_dir MODEL_DIR]
                  [--data_dir DATA_DIR] [--data_list DATA_LIST]
                  [--hdf5_file HDF5_FILE] [--hdf5_group HDF5_GROUP]
                  [--result_dir RESULT_DIR] [--result_fname RESULT_FNAME]
                  [--min_p_prob MIN_P_PROB] [--min_s_prob MIN_S_PROB]
                  [--mpd MPD] [--amplitude] [--format FORMAT]
                  [--s3_url S3_URL] [--stations STATIONS] [--plot_figure]
                  [--save_prob]

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        batch size
  --model_dir MODEL_DIR
                        Checkpoint directory (default: None)
  --data_dir DATA_DIR   Input file directory
  --data_list DATA_LIST
                        Input csv file
  --hdf5_file HDF5_FILE
                        Input hdf5 file
  --hdf5_group HDF5_GROUP
                        data group name in hdf5 file
  --result_dir RESULT_DIR
                        Output directory
  --result_fname RESULT_FNAME
                        Output file
  --min_p_prob MIN_P_PROB
                        Probability threshold for P pick
  --min_s_prob MIN_S_PROB
                        Probability threshold for S pick
  --mpd MPD             Minimum peak distance
  --amplitude           if return amplitude value
  --format FORMAT       input format
  --stations STATIONS   seismic station info
  --plot_figure         If plot figure for test
  --save_prob           If save result for test
```

- The output picks are saved to "results/picks.csv" on default

|file_name        |begin_time             |station_id|phase_index|phase_time             |phase_score|phase_amp             |phase_type|
|-----------------|-----------------------|----------|-----------|-----------------------|-----------|----------------------|----------|
|2020-10-01T00:00*|2020-10-01T00:00:00.003|CI.BOM..HH|14734      |2020-10-01T00:02:27.343|0.708      |2.4998866231208325e-14|P         |
|2020-10-01T00:00*|2020-10-01T00:00:00.003|CI.BOM..HH|15487      |2020-10-01T00:02:34.873|0.416      |2.4998866231208325e-14|S         |
|2020-10-01T00:00*|2020-10-01T00:00:00.003|CI.COA..HH|319        |2020-10-01T00:00:03.193|0.762      |3.708662269972206e-14 |P         |

Notes:
1. The *phase_index* means which data point is the pick in the original sequence. So *phase_time* = *begin_time* + *phase_index* / *sampling rate*. The default *sampling_rate* is 100Hz 


## 5. QuakeFlow example
A complete earthquake detection workflow can be found in the [QuakeFlow](https://wayneweiqiang.github.io/QuakeFlow/) project.

## 6. Interactive example
See details in the [notebook](https://github.com/wayneweiqiang/PhaseNet/blob/master/docs/example_interactive.ipynb): [example_interactive.ipynb](example_interactive.ipynb)

## 7. Training
- Download a small sample dataset:
```bash
wget https://github.com/wayneweiqiang/PhaseNet/releases/download/test_data/test_data.zip
unzip test_data.zip
```
- Start training from the pre-trained model
```
python phasenet/train.py  --model_dir=model/190703-214543/ --train_dir=test_data/npz --train_list=test_data/npz.csv  --plot_figure --epochs=10 --batch_size=10
```
- Check results in the **log** folder

