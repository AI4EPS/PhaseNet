import numpy as np
import sys, glob, os, time
from obspy.core import read as ocread
from obspy.core.stream import Stream

"""
WARNING ABSOLUTE TIMES WILL BE WRONG FOR THE SECOND PART OF THE FILE
"""

input_data = 'demo/sds/data/2000/*/AAAA/*Z.D/*'
output_dir = "demo/sds/default_output"
temp_csv_file = os.path.join(output_dir, "default_fname.csv")
preformatted_data_dir = os.path.join(output_dir, "preformatted_data")
input_length = 3000

if os.path.isdir(output_dir):
    raise Exception(output_dir, 'exists already')
os.system('mkdir ' + output_dir)
os.system('mkdir ' + preformatted_data_dir)

with open(temp_csv_file, 'w') as fid:
    fid.write('fname,E,N,Z\n')

    for zfname in glob.iglob(input_data):

        nfname = zfname.replace('HZ.', "HN.")
        efname = zfname.replace('HZ.', "HE.")
        output_file = os.path.join(
            preformatted_data_dir,
            zfname.split('/')[-1].replace('HZ.', 'H.'))

        assert os.path.isfile(nfname), nfname
        assert os.path.isfile(efname), efname
        assert efname != zfname, nfname
        assert nfname != zfname, efname

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

            tr.trim(starttime=tr.stats.starttime + 1000.,
                    endtime=tr.stats.starttime + 5000.)

            if tr.stats.npts % input_length:
                tr.data = np.hstack((tr.data, np.zeros(input_length - tr.stats.npts % input_length)))

        for tr in st:
            print(tr)
        st.write(output_file, format="MSEED")
        fid.write(f'{output_file.split("/")[-1]},{tre.stats.channel},{trn.stats.channel},{trz.stats.channel}\n')


script = f"""
/home/lehujeur/anaconda3/envs/py38-phasenet/bin/python run.py \\
    --mode=pred  \\
    --model_dir=model/190703-214543  \\
    --data_dir={preformatted_data_dir}  \\
    --data_list={temp_csv_file}  \\
    --output_dir={output_dir}  \\
    --batch_size=20  \\
    --input_length {input_length} \\
    --input_mseed  && \\
    touch {output_dir}/DONE
    
    # --plot_figure 
     
"""

os.system(script)

while not os.path.isfile(f'{output_dir}/DONE'):
    time.sleep(1.0)


known_files = {}
with open(output_dir + '/picks.csv', 'r') as fidr, \
     open(output_dir + '/picks_converted.csv', 'w') as fidw:

    fidr.readline()  # header
    fidw.write('seedid,phasename,time,probability\n')

    for line in fidr:
        line = line.strip('\n')
        fname, itp, tp_prob, its, ts_prob = line.split(',')
        if itp == "[]" and its == '[]':
            continue

        fname, batch_index = fname.split('_')
        batch_index = int(batch_index)

        itp = eval(",".join(itp.split()))
        tp_prob = eval(",".join(tp_prob.split()))
        its = eval(",".join(its.split()))
        ts_prob = eval(','.join(ts_prob.split()))

        fname = os.path.join(preformatted_data_dir, fname)
        assert os.path.isfile(fname), fname
        # print(fname, itp, tp_prob, its, ts_prob)

        try:
            trz = known_files[fname]
        except KeyError:
            trz = known_files[fname] = ocread(fname, format="MSEED", headonly=True)[0]

        if batch_index < trz.stats.npts:
            first_samp = batch_index
        else:
            # this is wrong, just use
            # absolute times will likely be wrong for the second half of the file
            first_samp = batch_index - trz.stats.npts - input_length

        seedid = f"{trz.stats.network}.{trz.stats.station}.{trz.stats.location}.{trz.stats.channel[:2]}.{trz.stats.mseed['dataquality']}"
        for i, p in zip(itp, tp_prob):
            true_time = trz.stats.starttime + trz.stats.delta * (first_samp + i)
            fidw.write(f"{seedid},P,{str(true_time)},{p:.6f}\n")

        for i, p in zip(its, ts_prob):
            true_time = trz.stats.starttime + trz.stats.delta * (first_samp + i)
            fidw.write(f"{seedid},S,{str(true_time)},{p:.6f}\n")
