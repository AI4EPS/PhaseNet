### Related paper:
Weiqiang Zhu, Gregory C Beroza; PhaseNet: a deep-neural-network-based seismic arrival-time picking method, Geophysical Journal International, Volume 216, Issue 1, 1 January 2019, Pages 261–273, https://doi.org/10.1093/gji/ggy423

## Install
```bash
virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Demo

### 1.Data

Located in directory: **Demo**

### 2.Model
Located in directory: **log/0118193232**

### 3. Prediction
~~~bash
python run.py --mode=pred --ckdir=log/0118193232 --data_dir=../Demo/PhaseNet --data_list=../Demo/PhaseNet.csv --output_dir=./output --plot_figure=True --save_result=True
~~~

