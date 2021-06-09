import numpy as np
import sys, glob, os, time
from obspy.core import read as ocread
from obspy.core.stream import Stream

input_data = 'demo/sds/data/2000/*/AAAA/*Z.D/*'


os.system('''
trash tmp_output tmp_input tmp_fname.csv log
mkdir tmp_output tmp_input
''')


with open('tmp_fname.csv', 'w') as fidr:
    fidr.write('fname,E,N,Z\n')

    for zfname in glob.iglob(input_data):

        nfname = zfname.replace('HZ.', "HN.")
        efname = zfname.replace('HZ.', "HE.")
        output_file = "tmp_input/" + zfname.split('/')[-1].replace('HZ.', 'H.')

        assert os.path.isfile(nfname), nfname
        assert os.path.isfile(efname), efname

        print(zfname)
        print(nfname)
        print(efname)
        print(output_file)

        stz = ocread(zfname, format="MSEED")
        stn = ocread(nfname, format="MSEED")
        ste = ocread(efname, format="MSEED")

        if len(stz) > 1:
            trz = stz.merge(fill_value=0)[0]
        else:
            trz = stz[0]

        if len(stn) > 1:
            trn = stn.merge(fill_value=0)[0]
        else:
            trn = stn[0]

        if len(ste) > 1:
            tre = ste.merge(fill_value=0)[0]
        else:
            tre = ste[0]

        st = Stream(traces=[tre, trn, trz])
        for tr in st:
            tr.data = np.asarray(tr.data, np.dtype('int32'))
            tr.stats.mseed = {}

        for tr in st:
            print(tr)
        st.write(output_file, format="MSEED")
        fidr.write(f'{output_file.split("/")[-1]},{tre.stats.channel},{trn.stats.channel},{trz.stats.channel}\n')


script = """
/home/lehujeur/anaconda3/envs/py38-phasenet/bin/python run.py \\
    --mode=pred  \\
    --model_dir=model/190703-214543  \\
    --data_dir=tmp_input  \\
    --data_list=tmp_fname.csv  \\
    --output_dir=tmp_output  \\
    --batch_size=20  \\
    --input_mseed  && touch tmp_output/DONE 
"""

os.system(script)

while not os.path.isfile('tmp_output/DONE'):
    time.sleep(1.0)


import numpy as np
import sys, glob, os, time
from obspy.core import read as ocread
from obspy.core.stream import Stream

known_files = {}
with open('tmp_output/picks.csv', 'r') as fidr, open('tmp_output/picks_converted.csv', 'w') as fidw:
    fidr.readline()  # header
    fidw.write('seedid,phasename,time,probability\n')

    for line in fidr:
        line = line.strip('\n')
        fname, itp, tp_prob, its, ts_prob = line.split(',')
        if itp == "[]" and its == '[]':
            continue

        fname, first_samp = fname.split('_')
        first_samp = int(first_samp)

        itp = eval(",".join(itp.split()))
        tp_prob = eval(",".join(tp_prob.split()))
        its = eval(",".join(its.split()))
        ts_prob = eval(','.join(ts_prob.split()))

        fname = "tmp_input/" + fname
        assert os.path.isfile(fname), fname
        # print(fname, itp, tp_prob, its, ts_prob)

        try:
            trz = known_files[fname]
        except KeyError:
            trz = known_files[fname] = ocread(fname, format="MSEED", headonly=True)[0]
        seedid = f"{trz.stats.network}.{trz.stats.station}.{trz.stats.location}.{trz.stats.channel[:2]}.{trz.stats.mseed['dataquality']}"
        for i, p in zip(itp, tp_prob):
            true_time = trz.stats.starttime + trz.stats.delta * (first_samp + i)
            fidw.write(f"{seedid},P,{str(true_time)},{p}\n")

        for i, p in zip(its, ts_prob):
            true_time = trz.stats.starttime + trz.stats.delta * (first_samp + i)
            fidw.write(f"{seedid},S,{str(true_time)},{p}\n")
