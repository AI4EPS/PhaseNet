import os
import matplotlib.pyplot as plt
from sds_plugin import DataReaderSDS


data_reader = DataReaderSDS(
    data_dir=os.path.join("demo", "sds", "data"),
    data_list=os.path.join("demo", "sds", "fname_sds.csv"),
    queue_size=None,
    coord=None,
    input_length=None)

data_reader.show_sds_prediction_results(
    log_dir=os.path.join("demo", "sds", "output"))
