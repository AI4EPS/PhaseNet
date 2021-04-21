import os
import matplotlib.pyplot as plt
from sds_plugin import show_sds_prediction_results, DataReaderSDS


data_reader = DataReaderSDS(
    data_dir=os.path.join("demo", "sds", "data"),
    data_list=os.path.join("demo", "sds", "fname_sds.csv"),
    queue_size=None,
    coord=None,
    input_length=None)

fig = plt.figure(figsize=(12, 4))
show_sds_prediction_results(
    fig=fig,
    data_reader=data_reader,
    log_dir=os.path.join("demo", "sds", "output"))
