import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_residual(diff_p, diff_s, diff_ps, tol, dt):
    box = dict(boxstyle='round', facecolor='white', alpha=1)
    text_loc = [0.07, 0.95]
    plt.figure(figsize=(8,3))
    plt.subplot(1,3,1)
    plt.hist(diff_p, range=(-tol, tol), bins=int(2*tol/dt)+1, facecolor='b', edgecolor='black', linewidth=1)
    plt.ylabel("Number of picks")
    plt.xlabel("Residual (s)")
    plt.text(text_loc[0], text_loc[1], "(i)", horizontalalignment='left', verticalalignment='top',
            transform=plt.gca().transAxes, fontsize="small", fontweight="normal", bbox=box)
    plt.title("P-phase")
    plt.subplot(1,3,2)
    plt.hist(diff_s, range=(-tol, tol), bins=int(2*tol/dt)+1, facecolor='b', edgecolor='black', linewidth=1)
    plt.xlabel("Residual (s)")
    plt.text(text_loc[0], text_loc[1], "(ii)", horizontalalignment='left', verticalalignment='top',
            transform=plt.gca().transAxes, fontsize="small", fontweight="normal", bbox=box)
    plt.title("S-phase")
    plt.subplot(1,3,3)
    plt.hist(diff_ps, range=(-tol, tol), bins=int(2*tol/dt)+1, facecolor='b', edgecolor='black', linewidth=1)
    plt.xlabel("Residual (s)")
    plt.text(text_loc[0], text_loc[1], "(iii)", horizontalalignment='left', verticalalignment='top',
            transform=plt.gca().transAxes, fontsize="small", fontweight="normal", bbox=box)
    plt.title("PS-phase")
    plt.tight_layout()
    plt.savefig("residuals.png", dpi=300)
    plt.savefig("residuals.pdf")


# def plot_waveform(config, data, pred, label=None, 
#                   itp=None, its=None, itps=None,
#                   itp_pred=None, its_pred=None, itps_pred=None,
#                   fname=None, figure_dir="./", epoch=0, max_fig=10):

#     dt = config.dt if hasattr(config, "dt") else 1.0
#     t = np.arange(0, pred.shape[1]) * dt
#     box = dict(boxstyle='round', facecolor='white', alpha=1)
#     text_loc = [0.05, 0.77]
#     if fname is None:
#         fname = [f"{epoch:03d}_{i:02d}" for i in range(len(data))]
#     else:
#         fname = [fname[i].decode().rstrip(".npz") for i in range(len(fname))]
        
#     for i in range(min(len(data), max_fig)):
#         plt.figure(i)
        
#         plt.subplot(411)
#         plt.plot(t, data[i, :, 0, 0], 'k', label='E', linewidth=0.5)
#         plt.autoscale(enable=True, axis='x', tight=True)
#         tmp_min = np.min(data[i, :, 0, 0])
#         tmp_max = np.max(data[i, :, 0, 0])
#         if (itp is not None) and (its is not None):
#             for j in range(len(itp[i])):
#                 lb = "P" if j==0 else ""
#                 plt.plot([itp[i][j]*dt, itp[i][j]*dt], [tmp_min, tmp_max], 'C0', label=lb, linewidth=0.5)
#             for j in range(len(its[i])):
#                 lb = "S" if j==0 else ""
#                 plt.plot([its[i][j]*dt, its[i][j]*dt], [tmp_min, tmp_max], 'C1', label=lb, linewidth=0.5)
#         if (itps is not None):
#             for j in range(len(itps[i])):
#                 lb = "PS" if j==0 else ""
#                 plt.plot([itps[i][j]*dt, its[i][j]*dt], [tmp_min, tmp_max], 'C2', label=lb, linewidth=0.5)
#         plt.ylabel('Amplitude')
#         plt.legend(loc='upper right', fontsize='small')
#         plt.gca().set_xticklabels([])
#         plt.text(text_loc[0], text_loc[1], '(i)', horizontalalignment='center',
#                  transform=plt.gca().transAxes, fontsize="small", fontweight="normal", bbox=box)
        
#         plt.subplot(412)
#         plt.plot(t, data[i, :, 0, 1], 'k', label='N', linewidth=0.5)
#         plt.autoscale(enable=True, axis='x', tight=True)
#         tmp_min = np.min(data[i, :, 0, 1])
#         tmp_max = np.max(data[i, :, 0, 1])
#         if (itp is not None) and (its is not None):
#             for j in range(len(itp[i])):
#                 lb = "P" if j==0 else ""
#                 plt.plot([itp[i][j]*dt, itp[i][j]*dt], [tmp_min, tmp_max], 'C0', label=lb, linewidth=0.5)
#             for j in range(len(its[i])):
#                 lb = "S" if j==0 else ""
#                 plt.plot([its[i][j]*dt, its[i][j]*dt], [tmp_min, tmp_max], 'C1', label=lb, linewidth=0.5)
#         if (itps is not None):
#             for j in range(len(itps[i])):
#                 lb = "PS" if j==0 else ""
#                 plt.plot([itps[i][j]*dt, itps[i][j]*dt], [tmp_min, tmp_max], 'C2', label=lb, linewidth=0.5)
#         plt.ylabel('Amplitude')
#         plt.legend(loc='upper right', fontsize='small')
#         plt.gca().set_xticklabels([])
#         plt.text(text_loc[0], text_loc[1], '(ii)', horizontalalignment='center',
#                 transform=plt.gca().transAxes, fontsize="small", fontweight="normal", bbox=box)
        
#         plt.subplot(413)
#         plt.plot(t, data[i, :, 0, 2], 'k', label='Z', linewidth=0.5)
#         plt.autoscale(enable=True, axis='x', tight=True)
#         tmp_min = np.min(data[i, :, 0, 2])
#         tmp_max = np.max(data[i, :, 0, 2])
#         if (itp is not None) and (its is not None):
#             for j in range(len(itp[i])):
#                 lb = "P" if j==0 else ""
#                 plt.plot([itp[i][j]*dt, itp[i][j]*dt], [tmp_min, tmp_max], 'C0', label=lb, linewidth=0.5)
#             for j in range(len(its[i])):
#                 lb = "S" if j==0 else ""
#                 plt.plot([its[i][j]*dt, its[i][j]*dt], [tmp_min, tmp_max], 'C1', label=lb, linewidth=0.5)
#         if (itps is not None):
#             for j in range(len(itps[i])):
#                 lb = "PS" if j==0 else ""
#                 plt.plot([itps[i][j]*dt, itps[i][j]*dt], [tmp_min, tmp_max], 'C2', label=lb, linewidth=0.5)
#         plt.ylabel('Amplitude')
#         plt.legend(loc='upper right', fontsize='small')
#         plt.gca().set_xticklabels([])
#         plt.text(text_loc[0], text_loc[1], '(iii)', horizontalalignment='center',
#                 transform=plt.gca().transAxes, fontsize="small", fontweight="normal", bbox=box)
        
#         plt.subplot(414)
#         if label is not None:
#             plt.plot(t, label[i, :, 0, 1], 'C0', label='P', linewidth=1)
#             plt.plot(t, label[i, :, 0, 2], 'C1', label='S', linewidth=1)
#             if label.shape[-1] == 4:
#                 plt.plot(t, label[i, :, 0, 3], 'C2', label='PS', linewidth=1)
#         plt.plot(t, pred[i, :, 0, 1], '--C0', label='$\hat{P}$', linewidth=1)
#         plt.plot(t, pred[i, :, 0, 2], '--C1', label='$\hat{S}$', linewidth=1)
#         if pred.shape[-1] == 4:
#             plt.plot(t, pred[i, :, 0, 3], '--C2', label='$\hat{PS}$', linewidth=1)
#         plt.autoscale(enable=True, axis='x', tight=True)
#         if (itp_pred is not None) and (its_pred is not None) :
#             for j in range(len(itp_pred)):
#                 plt.plot([itp_pred[j]*dt, itp_pred[j]*dt], [-0.1, 1.1], '--C0', linewidth=1)
#             for j in range(len(its_pred)):
#                 plt.plot([its_pred[j]*dt, its_pred[j]*dt], [-0.1, 1.1], '--C1', linewidth=1)
#         if (itps_pred is not None):
#             for j in range(len(itps_pred)):
#                 plt.plot([itps_pred[j]*dt, itps_pred[j]*dt], [-0.1, 1.1], '--C2', linewidth=1)
#         plt.ylim([-0.05, 1.05])
#         plt.text(text_loc[0], text_loc[1], '(iv)', horizontalalignment='center',
#                  transform=plt.gca().transAxes, fontsize="small", fontweight="normal", bbox=box)
#         plt.legend(loc='upper right', fontsize='small', ncol=2)
#         plt.xlabel('Time (s)')
#         plt.ylabel('Probability')
#         plt.tight_layout()
#         plt.gcf().align_labels()

#         try:
#             plt.savefig(os.path.join(figure_dir, fname[i]+'.png'), bbox_inches='tight')
#         except FileNotFoundError:
#             os.makedirs(os.path.dirname(os.path.join(figure_dir, fname[i])), exist_ok=True)
#             plt.savefig(os.path.join(figure_dir, fname[i]+'.png'), bbox_inches='tight')

#         plt.close(i)
#     return 0


def plot_waveform(data, pred, fname, label=None, 
                  itp=None, its=None, itps=None,
                  itp_pred=None, its_pred=None, itps_pred=None,
                  figure_dir="./", dt=0.01):

    t = np.arange(0, pred.shape[0]) * dt
    box = dict(boxstyle='round', facecolor='white', alpha=1)
    text_loc = [0.05, 0.77]

    plt.figure()
    
    plt.subplot(411)
    plt.plot(t, data[:, 0, 0], 'k', label='E', linewidth=0.5)
    plt.autoscale(enable=True, axis='x', tight=True)
    tmp_min = np.min(data[:, 0, 0])
    tmp_max = np.max(data[:, 0, 0])
    if (itp is not None) and (its is not None):
        for j in range(len(itp)):
            lb = "P" if j==0 else ""
            plt.plot([itp[j]*dt, itp[j]*dt], [tmp_min, tmp_max], 'C0', label=lb, linewidth=0.5)
        for j in range(len(its[i])):
            lb = "S" if j==0 else ""
            plt.plot([its[j]*dt, its[j]*dt], [tmp_min, tmp_max], 'C1', label=lb, linewidth=0.5)
    if (itps is not None):
        for j in range(len(itps)):
            lb = "PS" if j==0 else ""
            plt.plot([itps[j]*dt, its[j]*dt], [tmp_min, tmp_max], 'C2', label=lb, linewidth=0.5)
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right', fontsize='small')
    plt.gca().set_xticklabels([])
    plt.text(text_loc[0], text_loc[1], '(i)', horizontalalignment='center',
                transform=plt.gca().transAxes, fontsize="small", fontweight="normal", bbox=box)
    
    plt.subplot(412)
    plt.plot(t, data[:, 0, 1], 'k', label='N', linewidth=0.5)
    plt.autoscale(enable=True, axis='x', tight=True)
    tmp_min = np.min(data[:, 0, 1])
    tmp_max = np.max(data[:, 0, 1])
    if (itp is not None) and (its is not None):
        for j in range(len(itp)):
            lb = "P" if j==0 else ""
            plt.plot([itp[j]*dt, itp[j]*dt], [tmp_min, tmp_max], 'C0', label=lb, linewidth=0.5)
        for j in range(len(its)):
            lb = "S" if j==0 else ""
            plt.plot([its[j]*dt, its[j]*dt], [tmp_min, tmp_max], 'C1', label=lb, linewidth=0.5)
    if (itps is not None):
        for j in range(len(itps)):
            lb = "PS" if j==0 else ""
            plt.plot([itps[j]*dt, itps[j]*dt], [tmp_min, tmp_max], 'C2', label=lb, linewidth=0.5)
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right', fontsize='small')
    plt.gca().set_xticklabels([])
    plt.text(text_loc[0], text_loc[1], '(ii)', horizontalalignment='center',
            transform=plt.gca().transAxes, fontsize="small", fontweight="normal", bbox=box)
    
    plt.subplot(413)
    plt.plot(t, data[:, 0, 2], 'k', label='Z', linewidth=0.5)
    plt.autoscale(enable=True, axis='x', tight=True)
    tmp_min = np.min(data[:, 0, 2])
    tmp_max = np.max(data[:, 0, 2])
    if (itp is not None) and (its is not None):
        for j in range(len(itp)):
            lb = "P" if j==0 else ""
            plt.plot([itp[j]*dt, itp[j]*dt], [tmp_min, tmp_max], 'C0', label=lb, linewidth=0.5)
        for j in range(len(its)):
            lb = "S" if j==0 else ""
            plt.plot([its[j]*dt, its[j]*dt], [tmp_min, tmp_max], 'C1', label=lb, linewidth=0.5)
    if (itps is not None):
        for j in range(len(itps)):
            lb = "PS" if j==0 else ""
            plt.plot([itps[j]*dt, itps[j]*dt], [tmp_min, tmp_max], 'C2', label=lb, linewidth=0.5)
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right', fontsize='small')
    plt.gca().set_xticklabels([])
    plt.text(text_loc[0], text_loc[1], '(iii)', horizontalalignment='center',
            transform=plt.gca().transAxes, fontsize="small", fontweight="normal", bbox=box)
    
    plt.subplot(414)
    if label is not None:
        plt.plot(t, label[:, 0, 1], 'C0', label='P', linewidth=1)
        plt.plot(t, label[:, 0, 2], 'C1', label='S', linewidth=1)
        if label.shape[-1] == 4:
            plt.plot(t, label[:, 0, 3], 'C2', label='PS', linewidth=1)
    plt.plot(t, pred[:, 0, 1], '--C0', label='$\hat{P}$', linewidth=1)
    plt.plot(t, pred[:, 0, 2], '--C1', label='$\hat{S}$', linewidth=1)
    if pred.shape[-1] == 4:
        plt.plot(t, pred[:, 0, 3], '--C2', label='$\hat{PS}$', linewidth=1)
    plt.autoscale(enable=True, axis='x', tight=True)
    if (itp_pred is not None) and (its_pred is not None) :
        for j in range(len(itp_pred)):
            plt.plot([itp_pred[j]*dt, itp_pred[j]*dt], [-0.1, 1.1], '--C0', linewidth=1)
        for j in range(len(its_pred)):
            plt.plot([its_pred[j]*dt, its_pred[j]*dt], [-0.1, 1.1], '--C1', linewidth=1)
    if (itps_pred is not None):
        for j in range(len(itps_pred)):
            plt.plot([itps_pred[j]*dt, itps_pred[j]*dt], [-0.1, 1.1], '--C2', linewidth=1)
    plt.ylim([-0.05, 1.05])
    plt.text(text_loc[0], text_loc[1], '(iv)', horizontalalignment='center',
                transform=plt.gca().transAxes, fontsize="small", fontweight="normal", bbox=box)
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Probability')
    plt.tight_layout()
    plt.gcf().align_labels()

    try:
        plt.savefig(os.path.join(figure_dir, fname+'.png'), bbox_inches='tight')
    except FileNotFoundError:
        os.makedirs(os.path.dirname(os.path.join(figure_dir, fname)), exist_ok=True)
        plt.savefig(os.path.join(figure_dir, fname+'.png'), bbox_inches='tight')

    plt.close()
    return 0


def plot_array(config, data, pred, label=None,
               itp=None, its=None, itps=None,
               itp_pred=None, its_pred=None, itps_pred=None,
               fname=None, figure_dir="./", epoch=0):

    dt = config.dt if hasattr(config, "dt") else 1.0
    t = np.arange(0, pred.shape[1]) * dt
    box = dict(boxstyle='round', facecolor='white', alpha=1)
    text_loc = [0.05, 0.95]
    if fname is None:
        fname = [f"{epoch:03d}_{i:03d}" for i in range(len(data))]
    else:
        fname = [fname[i].decode().rstrip(".npz") for i in range(len(fname))]
        
    for i in range(len(data)):
        plt.figure(i, figsize=(10, 5))
        plt.clf()

        plt.subplot(121)
        for j in range(data.shape[-2]):
            plt.plot(t, data[i, :, j, 0]/10 + j, 'k', label='E', linewidth=0.5)
        plt.autoscale(enable=True, axis='x', tight=True)
        tmp_min = np.min(data[i, :, 0, 0])
        tmp_max = np.max(data[i, :, 0, 0])
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        # plt.legend(loc='upper right', fontsize='small')
        # plt.gca().set_xticklabels([])
        plt.text(text_loc[0], text_loc[1], '(i)', horizontalalignment='center', verticalalignment="top", 
                 transform=plt.gca().transAxes, fontsize="large", fontweight="normal", bbox=box)
        
        plt.subplot(122)
        for j in range(pred.shape[-2]):
            if label is not None:
                plt.plot(t, label[i, :, j, 1]+j, 'C2', label='P', linewidth=0.5)
                plt.plot(t, label[i, :, j, 2]+j, 'C3', label='S', linewidth=0.5)
                # plt.plot(t, label[i, :, j, 0]+j, 'C4', label='N', linewidth=0.5)
            plt.plot(t, pred[i, :, j, 1]+j, 'C0', label='$\hat{P}$', linewidth=1)
            plt.plot(t, pred[i, :, j, 2]+j, 'C1', label='$\hat{S}$', linewidth=1)
            plt.autoscale(enable=True, axis='x', tight=True)
        if (itp_pred is not None) and (its_pred is not None) and (itps_pred is not None):
            for j in range(len(itp_pred)):
                plt.plot([itp_pred[j]*dt, itp_pred[j]*dt], [-0.1, 1.1], '--C0', linewidth=1)
            for j in range(len(its_pred)):
                plt.plot([its_pred[j]*dt, its_pred[j]*dt], [-0.1, 1.1], '--C1', linewidth=1)
            for j in range(len(itps_pred)):
                plt.plot([itps_pred[j]*dt, itps_pred[j]*dt], [-0.1, 1.1], '--C2', linewidth=1)
        # plt.ylim([-0.05, 1.05])
        plt.text(text_loc[0], text_loc[1], '(ii)', horizontalalignment='center', verticalalignment="top", 
                 transform=plt.gca().transAxes, fontsize="large", fontweight="normal", bbox=box)
        # plt.legend(loc='upper right', fontsize='small', ncol=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Probability')
        plt.tight_layout()
        plt.gcf().align_labels()

        try:
            plt.savefig(os.path.join(figure_dir, fname[i]+'.png'), bbox_inches='tight')
        except FileNotFoundError:
            os.makedirs(os.path.dirname(os.path.join(figure_dir, fname[i])), exist_ok=True)
            plt.savefig(os.path.join(figure_dir, fname[i]+'.png'), bbox_inches='tight')

        plt.close(i)
    return 0


def plot_spectrogram(config, data, pred, label=None, 
                     itp=None, its=None, itps=None,
                     itp_pred=None, its_pred=None, itps_pred=None,
                     time=None, freq=None,
                     fname=None, figure_dir="./", epoch=0):

    # dt = config.dt
    # df = config.df
    # t = np.arange(0, data.shape[1]) * dt
    # f = np.arange(0, data.shape[2]) * df
    t, f = time, freq
    dt = t[1] - t[0]
    box = dict(boxstyle='round', facecolor='white', alpha=1)
    text_loc = [0.05, 0.75]
    if fname is None:
        fname = [f"{i:03d}" for i in range(len(data))]
    elif type(fname[0]) is bytes:
        fname = [f.decode() for f in fname]
        
    numbers = ["(i)", "(ii)", "(iii)", "(iv)"]
    for i in range(len(data)):
        fig = plt.figure(i)
        # gs = fig.add_gridspec(4, 1)
        
        for j in range(3):
            # fig.add_subplot(gs[j, 0])
            plt.subplot(4,1,j+1)
            plt.pcolormesh(t, f, np.abs(data[i, :, :, j]+1j*data[i, :, :, j+3]).T, vmax=2*np.std(data[i, :, :, j]+1j*data[i, :, :, j+3]), cmap="jet", shading='auto')
            plt.autoscale(enable=True, axis='x', tight=True)
            plt.gca().set_xticklabels([])
            if j == 1:
                plt.ylabel('Frequency (Hz)')
            plt.text(text_loc[0], text_loc[1], numbers[j], horizontalalignment='center',
                    transform=plt.gca().transAxes, fontsize="small", fontweight="normal", bbox=box)

        # fig.add_subplot(gs[-1, 0])
        plt.subplot(4,1,4)
        if label is not None:
            plt.plot(t, label[i, :, 0, 1], '--C0', linewidth=1)
            plt.plot(t, label[i, :, 0, 2], '--C3', linewidth=1)
            plt.plot(t, label[i, :, 0, 3], '--C1', linewidth=1)
        plt.plot(t, pred[i, :, 0, 1], 'C0', label='P', linewidth=1)
        plt.plot(t, pred[i, :, 0, 2], 'C3', label='S', linewidth=1)
        plt.plot(t, pred[i, :, 0, 3], 'C1', label='PS', linewidth=1)
        plt.plot(t, t*0, 'k', linewidth=1)
        plt.autoscale(enable=True, axis='x', tight=True)
        if (itp_pred is not None) and (its_pred is not None) and (itps_pred is not None):
            for j in range(len(itp_pred)):
                plt.plot([itp_pred[j]*dt, itp_pred[j]*dt], [-0.1, 1.1], ':C3', linewidth=1)
            for j in range(len(its_pred)):
                plt.plot([its_pred[j]*dt, its_pred[j]*dt], [-0.1, 1.1], '-.C6', linewidth=1)
            for j in range(len(itps_pred)):
                plt.plot([itps_pred[j]*dt, itps_pred[j]*dt], [-0.1, 1.1], '--C8', linewidth=1)
        plt.ylim([-0.05, 1.05])
        plt.text(text_loc[0], text_loc[1], numbers[-1], horizontalalignment='center',
                 transform=plt.gca().transAxes, fontsize="small", fontweight="normal", bbox=box)
        plt.legend(loc='upper right', fontsize='small', ncol=1)
        plt.xlabel('Time (s)')
        plt.ylabel('Probability')
        # plt.tight_layout()
        plt.gcf().align_labels()

        try:
            plt.savefig(os.path.join(figure_dir, f'{epoch:02d}_'+fname[i]+'.png'), bbox_inches='tight')
        except FileNotFoundError:
            os.makedirs(os.path.dirname(os.path.join(figure_dir, fname[i])), exist_ok=True)
            plt.savefig(os.path.join(figure_dir, f'{epoch:02d}_'+fname[i]+'.png'), bbox_inches='tight')

        plt.close(i)
    return 0


def plot_spectrogram_waveform(config, spectrogram, waveform, pred, label=None, 
                              itp=None, its=None, itps=None, picks=None,
                              time=None, freq=None,
                              fname=None, figure_dir="./", epoch=0):

    # dt = config.dt
    # df = config.df
    # t = np.arange(0, spectrogram.shape[1]) * dt
    # f = np.arange(0, spectrogram.shape[2]) * df
    t, f = time, freq
    dt = t[1] - t[0]
    box = dict(boxstyle='round', facecolor='white', alpha=1)
    text_loc = [0.02, 0.90]
    if fname is None:
        fname = [f"{i:03d}" for i in range(len(spectrogram))]
    elif type(fname[0]) is bytes:
        fname = [f.decode() for f in fname]
        
    numbers = ["(i)", "(ii)", "(iii)", "(iv)", "(v)", "(vi)", "(vii)"]
    for i in range(len(spectrogram)):
        fig = plt.figure(i, figsize=(6.4, 10))
        # gs = fig.add_gridspec(4, 1)
        
        for j in range(3):
            # fig.add_subplot(gs[j, 0])
            plt.subplot(7,1,j*2+1)
            plt.plot(waveform[i,:,j], 'k', linewidth=0.5)
            plt.autoscale(enable=True, axis='x', tight=True)
            plt.gca().set_xticklabels([])
            plt.ylabel('')
            plt.text(text_loc[0], text_loc[1], numbers[j*2], horizontalalignment='left', verticalalignment='top',
                    transform=plt.gca().transAxes, fontsize="small", fontweight="normal", bbox=box)

        for j in range(3):
            # fig.add_subplot(gs[j, 0])
            plt.subplot(7,1,j*2+2)
            plt.pcolormesh(t, f, np.abs(spectrogram[i, :, :, j]+1j*spectrogram[i, :, :, j+3]).T, vmax=2*np.std(spectrogram[i, :, :, j]+1j*spectrogram[i, :, :, j+3]), cmap="jet", shading='auto')
            plt.autoscale(enable=True, axis='x', tight=True)
            plt.gca().set_xticklabels([])
            if j == 1:
                plt.ylabel('Frequency (Hz) or Amplitude')
            plt.text(text_loc[0], text_loc[1], numbers[j*2+1], horizontalalignment='left', verticalalignment='top',
                    transform=plt.gca().transAxes, fontsize="small", fontweight="normal", bbox=box)

        # fig.add_subplot(gs[-1, 0])
        plt.subplot(7,1,7)
        if label is not None:
            plt.plot(t, label[i, :, 0, 1], '--C0', linewidth=1)
            plt.plot(t, label[i, :, 0, 2], '--C3', linewidth=1)
            plt.plot(t, label[i, :, 0, 3], '--C1', linewidth=1)
        plt.plot(t, pred[i, :, 0, 1], 'C0', label='P', linewidth=1)
        plt.plot(t, pred[i, :, 0, 2], 'C3', label='S', linewidth=1)
        plt.plot(t, pred[i, :, 0, 3], 'C1', label='PS', linewidth=1)
        plt.plot(t, t*0, 'k', linewidth=1)
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.ylim([-0.05, 1.05])
        plt.text(text_loc[0], text_loc[1], numbers[-1], horizontalalignment='left', verticalalignment='top',
                 transform=plt.gca().transAxes, fontsize="small", fontweight="normal", bbox=box)
        plt.legend(loc='upper right', fontsize='small', ncol=1)
        plt.xlabel('Time (s)')
        plt.ylabel('Probability')
        # plt.tight_layout()
        plt.gcf().align_labels()

        try:
            plt.savefig(os.path.join(figure_dir, f'{epoch:02d}_'+fname[i]+'.png'), bbox_inches='tight')
        except FileNotFoundError:
            os.makedirs(os.path.dirname(os.path.join(figure_dir, fname[i])), exist_ok=True)
            plt.savefig(os.path.join(figure_dir, f'{epoch:02d}_'+fname[i]+'.png'), bbox_inches='tight')

        plt.close(i)
    return 0