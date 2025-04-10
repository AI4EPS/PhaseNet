import os

cmd = f"python ../../phasenet/predict.py --model=../../model/190703-214543 --data_dir=./ --data_list=data.lst --format=das --amplitude --sampling_rate=1333.3333333333333 --result_dir=results --batch_size=1 --subdir=0"

os.system(cmd)
