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
~~~python
python run.py --mode=pred --ckdir=log/0118193232 --plot_figure=True --save_result=True
~~~

### 4. Output
Located in directory: **log/pred/**

## Hyper paramters

~~~
python run.py --help
~~~
```bash
usage: run.py [-h] [--mode MODE] [--epochs EPOCHS] [--batch_size BATCH_SIZE]
              [--learning_rate LEARNING_RATE] [--decay_step DECAY_STEP]
              [--decay_rate DECAY_RATE] [--momentum MOMENTUM]
              [--filters_root FILTERS_ROOT] [--depth DEPTH]
              [--kernel_size KERNEL_SIZE [KERNEL_SIZE ...]]
              [--pool_size POOL_SIZE [POOL_SIZE ...]] [--drop_rate DROP_RATE]
              [--dilation_rate DILATION_RATE [DILATION_RATE ...]]
              [--loss_type LOSS_TYPE] [--weight_decay WEIGHT_DECAY]
              [--optimizer OPTIMIZER] [--summary SUMMARY]
              [--class_weights CLASS_WEIGHTS [CLASS_WEIGHTS ...]]
              [--logdir LOGDIR] [--ckdir CKDIR] [--plot_number PLOT_NUMBER]
              [--fpred FPRED]
```