from sds_plugin import *

class Args(object):
    # generate a fake argument object for testing
    # reproduce the defaults options from run.py
    mode = "pred"
    epochs = 100
    batch_size = 20
    learning_rate = 0.01
    decay_step = -1
    decay_rate = 0.9
    momentum = 0.9
    filters_root = 8
    depth = 5
    kernel_size = [7, 1]
    pool_size = [4, 1]
    drop_rate = 0
    dilation_rate = [1, 1]
    loss_type = "cross_entropy"
    weight_decay = 0
    optimizer = 'adam'
    summary = True
    class_weights = [1, 1, 1]
    log_dir = None
    model_dir = os.path.join("model", "190703-214543")
    num_plots = 10
    tp_prob = 0.3
    ts_prob = 0.3
    input_length = None
    input_mseed = False
    input_sds = True
    data_dir = os.path.join("demo", "sds", "data")             # <====== CHANGE HERE
    data_list = os.path.join("demo", "sds", "fname_sds.csv")   # <====== CHANGE HERE
    train_dir = None
    valid_dir = None
    valid_list = None
    output_dir = os.path.join("demo", "sds", "sds_output")
    plot_figure = True  # will crash if save_result is False
    save_result = True
    fpred = "picks"

args = Args()
assert os.path.isdir(args.model_dir)
assert os.path.isdir(args.data_dir)
assert os.path.isfile(args.data_list)
assert not os.path.isdir(args.output_dir), f"output dir exists already {args.output_dir}, please move it or trash it and rerun"

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
coord = tf.train.Coordinator()

data_reader = DataReaderSDS(
    data_dir=args.data_dir,
    data_list=args.data_list,
    queue_size=args.batch_size * 10,
    coord=coord,
    input_length=args.input_length)

pred_fn_sds(args, data_reader, log_dir=args.output_dir)

if args.plot_figure:
    data_reader.show_sds_prediction_results(
        output_dir=args.output_dir)
