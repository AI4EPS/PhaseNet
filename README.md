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

## 4. Interactive example
See details in the [notebook](https://github.com/wayneweiqiang/PhaseNet/blob/master/docs/example_interactive.ipynb): [example_interactive.ipynb](example_interactive.ipynb)


## 5. Batch prediction
See details in the [notebook](https://github.com/wayneweiqiang/PhaseNet/blob/master/docs/example_batch_prediction.ipynb): [example_batch_prediction.ipynb](example_batch_prediction.ipynb)

## 6. Training

Please let us know of any bugs found in the code. Suggestions and collaborations are welcomed

