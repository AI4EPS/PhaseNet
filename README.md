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
Located in directory: **log/0118193232**

### 4. Prediction
~~~bash
source .venv/bin/activate
~~~
~~~bash
python run.py --mode=pred --ckdir=log/0118193232 --data_dir=../Demo/PhaseNet_test --data_list=../Demo/PhaseNet_test.csv --output_dir=./output --plot_figure --save_result
~~~
If using input data length other than 3000:
~~~bash
python run.py --mode=pred --ckdir=log/0118193232 --data_dir=../Demo/PhaseNet_test --
data_list=../Demo/PhaseNet_test.csv --input_length=5000 --output_dir=./output --plot_figure --save_result
~~~

### 5. Train
~~~bash
python run.py --mode=train --data_dir=../Demo/PhaseNet_train --data_list=../Demo/PhaseNet_train.csv --batch_size=20
~~~