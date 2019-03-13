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

The csv file contains one colomn: "fname"

The npz file contains one variable: "data"

The shape of "data" variable has a shape of 3000 x 3

~~~bash
source .venv/bin/activate
python run.py --mode=pred --ckdir=model/190227-104428 --data_dir=../Demo/Waveform_pred --data_list=../Demo/waveform.csv --output_dir=../output --plot_figure --save_result --batch_size=20
~~~

Notes:
1. If using input data length other than 3000, specify argument **--input_length=**
2. For large dataset and GPUs, larger batch size can speedup the prediction linearly. 
3. Plotting figure is slow. Removing the argument of **--plot_figure** can speedup the prediction

### 5. Train

#### Data format
Required a csv file and a directory of npz files.

The csv file contains four colomns: "fname", "itp", "its", "channels"

The npz file contains four variable: "data", "itp", "its", "channels"

The shape of "data" variables has a shape of 9001 x 3

The variabeles "itp" and "its" are the data pionts of first P&S arrivals picked by analysists. 

~~~bash
source .venv/bin/activate
python run.py --mode=train --data_dir=../Demo/Waveform --data_list=../Demo/waveform.csv --batch_size=20
~~~

### 6. Valid and Test
~~~bash
source .venv/bin/activate
python run.py --mode=valid --ckdir=model/190227-104428 --data_dir=../Demo/Waveform --data_list=../Demo/waveform.csv --plot_figure --save_result --batch_size=20
~~~

Let us know any bugs found in the code. Suggetsions and collobratons are welcomed!