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
~~~bash
source .venv/bin/activate
~~~
~~~bash
python run.py --mode=pred --ckdir=model/190227-104428 --data_dir=../Demo/PhaseNet_test --data_list=../Demo/PhaseNet_test.csv --output_dir=../output --plot_figure --save_result
~~~

If using input data length other than 3000, specify argument --input_length=

### 5. Train

#### Data format
Required a csv file and a npz file.
The csv file contains four colomns: "fname", "itp", "its", "channels"
The npz file contains four variable: "data", "itp", "its", "channels"

~~~bash
python run.py --mode=train --data_dir=../Demo/PhaseNet_train --data_list=../Demo/PhaseNet_train.csv --batch_size=20
~~~