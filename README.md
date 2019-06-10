### Related paper:
Weiqiang Zhu, Gregory C Beroza; PhaseNet: a deep-neural-network-based seismic arrival-time picking method, Geophysical Journal International, Volume 216, Issue 1, 1 January 2019, Pages 261–273, https://doi.org/10.1093/gji/ggy423

## 1. Install
The code is tested under Python3.6.

```bash
pip install virtualenv
virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2.Demo Data

Located in directory: **Demo**

### 3.Model
Located in directory: **model/190227-104428**

### 4. Prediction
#### Data format
Required a csv file and a directory of npz files.

The csv file contains one column: "fname"

The npz file contains one variable: "data"

The shape of "data" variable has a shape of 3000 x 3

~~~bash
source .venv/bin/activate
python run.py --mode=pred --ckdir=model/190227-104428 --data_dir=../Demo/Waveform_pred --data_list=../Demo/waveform.csv --output_dir=../output --plot_figure --save_result --batch_size=20
~~~

Notes:
1. For large dataset and GPUs, larger batch size can accelerate the prediction. 
2. Plotting figures is slow. Removing the argument of **--plot_figure** can speed the prediction
3. If using input data length other than 3000, specify argument **--input_length=**. But this is not suggested as the model is trained using input length of 3000. Too long input length would degrade the performance.

### 5. Train

#### Data format
Required a csv file and a directory of npz files.

The csv file contains four columns: "fname", "itp", "its", "channels"

The npz file contains four variable: "data", "itp", "its", "channels"

The shape of "data" variables has a shape of 9001 x 3

The variables "itp" and "its" are the data points of first P&S arrivals picked by analysts. 

~~~bash
source .venv/bin/activate
python run.py --mode=train --data_dir=../Demo/Waveform --data_list=../Demo/waveform.csv --batch_size=20
~~~

### 6. Valid and Test
~~~bash
source .venv/bin/activate
python run.py --mode=valid --ckdir=model/190227-104428 --data_dir=../Demo/Waveform --data_list=../Demo/waveform.csv --plot_figure --save_result --batch_size=20
~~~

Please let us know of any bugs found in the code. Suggestions and collaborations are welcomed!