## 1.  Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) and requirements

- Install to default environment
```bash
conda env update -f=env.yml -n base
```
- Install to "phasenet" virtual envirionment
```bash
conda env create -f env.yml
conda activate phasenet
```


## 2.Demo Data

PhaseNet supports three data formats: numpy, hdf5, and mseed

Demo data are stored in the **dataset** folder

## 3.Model
Located in directory: **model/190703-214543**

## 4. Prediction 

For numpy format:
~~~bash
python phasenet/predict.py --model=model/190703-214543 --data_list=phasenet/test_data/selected_phases.csv --data_dir=phasenet/test_data/data --format=numpy
~~~

For hdf5 format:
~~~bash
python phasenet/predict.py --model=model/190703-214543 --hdf5_file=phasenet/test_data/data.h5 --hdf5_group=data --format=hdf5
~~~

For mseed formt:
~~~bash
python phasenet/predict.py --model=model/190703-214543 --data_list=phasenet/test_data/mseed_station.csv --data_dir=phasenet/test_data/waveforms --format=mseed
~~~

For an array of mseed (used by QuakeFlow):
~~~bash
python phasenet/predict.py --model=model/190703-214543 --data_list=phasenet/test_data/mseed.csv --data_dir=phasenet/test_data/waveforms --stations=phasenet/test_data/stations.csv  --format=mseed_array --amplitude
~~~

Notes:
1. The detected P&S phases are stored to file **picks.csv** inside **--output_dir**. The picks.csv has three columns: file name with the beginning sample index (fname_index), P-phase index (itp), P-phase probability (tp_prob), S-phase index (its), S-phase probability (ts_prob). The absolute phase index = fname_index + itp/its.
2. The activation thresholds for P&S phases are set to 0.3 as default. Specify **--tp_prob** and **--ts_prob** to change the two thresholds. 
3. On default, the mseed file is processed twice with 50% overlap to avoid phases being cut in the middle. The second pass are appended to the end of first pass. For example, if the index of the input data is from 0-60000, the index of second pass is from 60000-120000. If the processing window is 3000, the fist 1500 samples of the second pass are the padded zeros.
4. You can customze the preprocssing of mseed file, such as filtering and resampling, inside the function **read_mseed** in data_reader.py.
5. **demo/demo-obspy.ipynb** has a simple example of downloading and preparing mseed data using obspy.

#### b) Data format -- numpy array
Required a csv file and a directory of npz files.

The csv file contains one column: "fname"

The npz file contains one variable: "data"

The shape of "data" variable has a shape of 3000 x 3

~~~bash
source .venv/bin/activate
python run.py --mode=pred --model_dir=model/190703-214543 --data_dir=dataset/waveform_pred --data_list=dataset/waveform.csv --output_dir=output --plot_figure --save_result --batch_size=20
~~~

Notes:
1. For large dataset and GPUs, larger batch size can accelerate the prediction. 
2. Plotting figures and save resutls is very slow. Removing the argument of **--plot_figure, --save_result** can speed the prediction
3. If using input data length other than 3000, specify argument **--input_length=**. 

### 5. Training on new dataset

#### Training data format
Required a csv file and a directory of npz files.

The csv file contains four columns: "fname", "itp", "its", "channels"

The npz file contains four variable: "data", "itp", "its", "channels"

The shape of "data" variables has a shape of 9001 x 3

The variables "itp" and "its" are the data points of first P&S arrivals picked by analysts. 

- Training from scratch:

~~~bash
source .venv/bin/activate
python run.py --mode=train --train_dir=dataset/waveform_train --train_list=dataset/waveform.csv --batch_size=20
~~~

- Training from the pretrain model:

~~~bash
source .venv/bin/activate
python run.py --mode=train --model_dir=model/190703-214543 --train_dir=dataset/waveform_train --train_list=dataset/waveform.csv --batch_size=20
~~~


####  Validation and Testing
~~~bash
source .venv/bin/activate
python run.py --mode=valid --model_dir=model/190703-214543 --data_dir=dataset/waveform_train --data_list=dataset/waveform.csv --plot_figure --save_result --batch_size=20
~~~

Please let us know of any bugs found in the code. 


### Related papers:
- Zhu, W., & Beroza, G. C. (2018). PhaseNet: A Deep-Neural-Network-Based Seismic Arrival Time Picking Method. arXiv preprint arXiv:1803.03211.
- Liu, M., Zhang, M., Zhu, W., Ellsworth, W. L., & Li, H. Rapid Characterization of the July 2019 Ridgecrest, California Earthquake Sequence from Raw Seismic Data using Machine Learning Phase Picker. Geophysical Research Letters, e2019GL086189.
- Park, Y., Mousavi, S. M., Zhu, W., Ellsworth, W. L., & Beroza, G. C. (2020). Machine learning based analysis of the Guy-Greenbrier, Arkansas earthquakes: a tale of two sequences.
- Chai, C., Maceira, M., Santos‐Villalobos, H. J., Venkatakrishnan, S. V., Schoenball, M., Zhu, W., ... & EGS Collab Team. (2020). Using a Deep Neural Network and Transfer Learning to Bridge Scales for Seismic Phase Picking. Geophysical Research Letters, e2020GL088651.
- 

