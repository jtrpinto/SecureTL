'''
Secure Triplet Loss Project Repository (https://github.com/jtrpinto/SecureTL)

File: aux_functions.py
- Miscellaneous functions used to load and process face and ECG data.

"Secure Triplet Loss: Achieving Cancelability and Non-Linkability in End-to-End Deep Biometrics"
João Ribeiro Pinto, Miguel V. Correia, and Jaime S. Cardoso
IEEE Transactions on Biometrics, Behavior, and Identity Science

joao.t.pinto@inesctec.pt  |  https://jtrpinto.github.io
'''

import matplotlib
matplotlib.use('agg')   # Do not try to show any figures

import os
import cv2
import warnings
import numpy as np
from PIL import Image
from copy import deepcopy
from matplotlib import pyplot as pl
from sklearn.preprocessing import normalize


# AUXILIARY FUNCTIONS FOR THE FACE DATA

def process_image(origin, destination, id_name):
    # Used to open images and store them, prepared to be used,
    # in the destination directory.
    img_name = os.path.basename(origin)
    out_path = os.path.join(destination, id_name + '_' + img_name)
    if os.path.exists(out_path):
        return out_path
    else:
        img = Image.open(origin)
        shp = img.size
        img = np.array(img)[int(0.15*shp[0]):int(0.85*shp[0]), int(0.15*shp[1]):int(0.85*shp[1])]
        img = cv2.resize(img, (160, 160), interpolation=cv2.INTER_AREA)
        img = Image.fromarray(img)
        img.save(out_path)
        return out_path

def generate_id_triplets(identity, db_dir, save_dir, all_identities, n_triplets=100):
    # Generates n triplets for a given identity.
    triplets = list()
    id_dir = os.path.join(db_dir, identity)
    
    for nn in range(n_triplets):
        videos = os.listdir(id_dir)
        
        # If there is more than one folder for this ID, take the anchor and
        # positive images from two different folders:
        if len(videos) > 1:  
            a_p_vids = np.random.choice(videos, size=2, replace=False)
            a_vid_dir = os.path.join(id_dir, a_p_vids[0])
            anchor = os.path.join(a_vid_dir, np.random.choice(os.listdir(a_vid_dir)))
            p_vid_dir = os.path.join(id_dir, a_p_vids[1])
            positive = os.path.join(p_vid_dir, np.random.choice(os.listdir(p_vid_dir)))
        # Otherwise, choose both images randomly within the same folder:
        else:
            video_dir = os.path.join(id_dir, videos[0])
            frames = os.listdir(video_dir)
            a_p = np.random.choice(frames, size=2, replace=False)
            anchor = os.path.join(video_dir, a_p[0])
            positive = os.path.join(video_dir, a_p[1])
        
        # Choosing randomly a negative image, from a different identity:
        neg_id = np.random.choice(all_identities[np.argwhere(all_identities != identity)[:,0]])
        neg_id_dir = os.path.join(db_dir, neg_id)
        neg_vids = os.listdir(neg_id_dir)
        neg_v = np.random.choice(neg_vids)
        neg_v_dir = os.path.join(neg_id_dir, neg_v)
        neg_frames = os.listdir(neg_v_dir)
        negative = os.path.join(neg_v_dir, np.random.choice(neg_frames))

        # Take the images and process them to be used later:
        anchor_path = process_image(anchor, save_dir, identity)
        positive_path = process_image(positive, save_dir, identity)
        negative_path = process_image(negative, save_dir, neg_id)

        # Generate two keys:
        keys = np.random.randint(0, high=2, size=(2, 100))
        key1, key2 = normalize(keys, norm='l2', axis=1)

        # Save the triplets info:
        triplets.append([anchor_path, positive_path, negative_path, key1, key2])
    return triplets



# AUXILIARY FUNCIONS FOR THE ECG DATA

def load_uoftdb_recording(path, subject, session, filename):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        signal = np.loadtxt(path + 'uoftdb_' + subject + '_' + session + '_' + filename + '.txt')
    return {'signal': signal, 'fs': 200.0}


def getSegments(path, subjects, length, step):
    uoftdb_ses = ['1', '2', '3', '4', '5', '6']
    uoftdb_fil = ['1', '2', '3', '4', '5']

    segments = list()
    y = list()
    t = list()

    for sub in subjects:
        file_matrix = np.zeros((6, 5))
        for ses in uoftdb_ses:
            for fil in uoftdb_fil:
                data = load_uoftdb_recording(path, str(sub), ses, fil)
                file_matrix[int(ses)-1, int(fil)-1] = len(data['signal'])
                for kk in np.arange(length, len(data['signal']), step):
                    segments.append(data['signal'][kk-length:kk])
                    y.append(sub)
    return {'segments': segments, 'labels': y, 'times': t}


def extract_data(path, subjects, fs=200.0):
    no_templ = 26   # To reserve first 30 seconds for anchors

    data = getSegments(path, subjects, int(fs * 5), int(fs * 1))

    data_train_x = list()
    data_train_y = list()
    train_subs = {}

    for kk in range(len(data['labels'])):
        if str(data['labels'][kk]) not in train_subs:
            train_subs[str(data['labels'][kk])] = 0
        if train_subs[str(data['labels'][kk])] < no_templ:
            data_train_x.append(data['segments'][kk])
            data_train_y.append(data['labels'][kk])
            train_subs[str(data['labels'][kk])] += 1

    data = getSegments(path, subjects, int(fs * 5), int(fs * 1))

    data_test_x = list()
    data_test_y = list()
    test_subs = {}

    for kk in range(len(data['labels'])):
        if str(data['labels'][kk]) not in test_subs:
            test_subs[str(data['labels'][kk])] = 0
        if test_subs[str(data['labels'][kk])] >= no_templ + 4:
            data_test_x.append(data['segments'][kk])
            data_test_y.append(data['labels'][kk])
        test_subs[str(data['labels'][kk])] += 1

    train_x = np.array(data_train_x)
    test_x = np.array(data_test_x)

    return {'X_anchors': train_x, 'y_anchors': data_train_y, 'X_remaining': test_x, 'y_remaining': data_test_y}


def generate_keys(n_keys, key_length):
    keys = np.random.randint(0, high=2, size=(n_keys, key_length))
    keys = normalize(keys, norm='l2', axis=1)
    return keys


def generate_triplets(X_anchors, y_anchors, X_remaining, y_remaining, N=1000):
    xA, xP, xN, yD = random_triplets(X_anchors, y_anchors, X_remaining, y_remaining, size=N)
    k1 = generate_keys(N, 100)
    k2 = generate_keys(N, 100)
    return [xA, xP, xN, yD, k1, k2]


def normalise_templates(templates):
    out = deepcopy(templates)
    for ii in range(len(out)):
        numer = np.subtract(out[ii], np.mean(out[ii]))
        denum = np.std(out[ii])
        out[ii] = np.divide(numer, denum)
    return out


def prepare_for_dnn(X, y, z_score_normalise=True):
    if z_score_normalise:
        X = normalise_templates(X)
    X_cnn = X.reshape(X.shape + (1,))
    y_cnn = np.asarray(y, dtype='float')
    return X_cnn, y_cnn



# AUXILIARY FUNCTIONS FOR PLOTTING RESULTS

def cancelability_fmr_at_eer(canc_fmr, thresholds, eer_thr):
    aux = np.abs(thresholds - eer_thr)
    idx = np.argmin(aux)
    if thresholds[idx] == eer_thr:
        return canc_fmr[idx]
    elif thresholds[idx] > eer_thr:
        d_before = np.abs(thresholds[idx-1] - eer_thr)
        d_after = np.abs(thresholds[idx] - eer_thr)
        d_total = d_before + d_after
        return d_before*canc_fmr[idx]/d_total + d_after*canc_fmr[idx-1]/d_total
    elif thresholds[idx] < eer_thr:
        d_before = np.abs(thresholds[idx] - eer_thr)
        d_after = np.abs(thresholds[idx+1] - eer_thr)
        d_total = d_before + d_after
        return d_after*canc_fmr[idx]/d_total + d_before*canc_fmr[idx+1]/d_total


def fnmr_at_fmr(fnmr, fmr, reference_fmr=0.01):
    aux = np.abs(fmr - reference_fmr)
    idx = np.argmin(aux)
    if fmr[idx] == reference_fmr:
        return fnmr[idx]
    elif fmr[idx] > reference_fmr:
        d_before = np.abs(fmr[idx-1] - reference_fmr)
        d_after = np.abs(fmr[idx] - reference_fmr)
        d_total = d_before + d_after
        return d_before*fnmr[idx]/d_total + d_after*fnmr[idx-1]/d_total
    elif fmr[idx] < reference_fmr:
        d_before = np.abs(fmr[idx] - reference_fmr)
        d_after = np.abs(fmr[idx+1] - reference_fmr)
        d_total = d_before + d_after
        return d_after*fnmr[idx]/d_total + d_before*fnmr[idx+1]/d_total


def plot_perf_curves(results, title=None, figsize=[6.4, 4.0], savefile=None):    
    pl.figure(figsize=figsize)
    pl.plot(np.linspace(0, 1, len(results['roc'][1])), results['roc'][1], 'r-', label=r'$FNMR$')
    pl.plot(np.linspace(0, 1, len(results['roc'][0])), results['roc'][0], 'b-', label=r'$FMR$')
    pl.scatter(results['eer'][0], results['eer'][1], color='black', marker='o', label=r'$EER$')
    pl.legend()
    pl.xlabel(r'Threshold ($t$)')
    pl.ylabel(r'Error Rate')
    pl.xlim([-0.02, 1.02])
    pl.ylim([-0.02, 1.02])
    if title is not None:
        pl.title(title)
        print(title)
        print('EER', results['eer'][1])
        print('FNMR@0.1%FMR', fnmr_at_fmr(results['roc'][1], results['roc'][0], reference_fmr=0.001))
        print('FNMR@1%FMR', fnmr_at_fmr(results['roc'][1], results['roc'][0], reference_fmr=0.01))
        print()
    pl.tight_layout()
    pl.grid(which='both')
    if savefile is not None:
        pl.savefig(savefile)
    else:
        pl.show()
    
    
def plot_perf_vs_canc_curves(results, smin=0, smax=1, title=None, figsize=[6.4, 4.0], savefile=None):
    eer = results[0]
    canc = results[1]
    fmrc_eer = cancelability_fmr_at_eer(canc['roc'][0], canc['roc'][2], eer['eer'][0])
    pl.figure(figsize=figsize)
    pl.plot(np.linspace(smin, smax, len(eer['roc'][1])), eer['roc'][1], 'r-', label=r'$FNMR$')
    pl.plot(np.linspace(smin, smax, len(eer['roc'][0])), eer['roc'][0], 'b-', label=r'$FMR_V$')
    pl.plot(np.linspace(smin, smax, len(canc['roc'][0])), canc['roc'][0], 'g-', label=r'$FMR_C$')
    pl.scatter(eer['eer'][0], eer['eer'][1], color='black', marker='o', label=r'$EER$')
    pl.scatter(eer['eer'][0], fmrc_eer, color='green', marker='o', label=r'$FMR_C@EER$')
    pl.legend()
    pl.xlabel(r'Threshold ($t$)')
    pl.ylabel(r'Error Rate')
    pl.xlim([smin - 0.02, smax + 0.02])
    pl.ylim([smin - 0.02, smax + 0.02])
    if title is not None:
        pl.title(title)
        print(title)
        print('EER', eer['eer'][1])
        print('FNMR@0.1%FMR', fnmr_at_fmr(eer['roc'][1], eer['roc'][0], reference_fmr=0.001))
        print('FNMR@1%FMR', fnmr_at_fmr(eer['roc'][1], eer['roc'][0], reference_fmr=0.01))
        print('FMR_C@EER', fmrc_eer)
    pl.tight_layout()
    pl.grid(which='both')
    if savefile is not None:
        pl.savefig(savefile)
    else:
        pl.show()


def plot_dsys(results, smin=0, smax=1, title=None, figsize=[6.4, 4.0], savefile=None):
    d_s = results[2][1]
    p_sHm = results[2][2]
    p_sHnm = results[2][3]
    N = len(d_s)
    fig = pl.figure(figsize=figsize)
    if title is not None:
        pl.title(title)
        print('d_sys', results[2][0])
        print()
    ax = fig.add_subplot()
    ax.set_xlabel(r'Distance score ($d$)')
    ax.set_xlim([smin-0.02, smax+0.02])
    ax.set_ylabel(r'Probability density ($p$)')
    max_y = max(np.max(p_sHm), np.max(p_sHnm))
    ax.set_ylim([-0.02*max_y, 1.02*max_y])
    c1 = ax.plot(np.linspace(smin, smax, num=N), p_sHm, 'g', label=r'$p(d|H_{m})$')
    c2 = ax.plot(np.linspace(smin, smax, num=N), p_sHnm, 'r', label=r'$p(d|H_{nm})$')
    ax.tick_params(axis='y')
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel(r'$D_{\leftrightarrow}(d)$')
    ax2.set_ylim([-0.02, 1.02])
    c3 = ax2.plot(np.linspace(smin, smax, num=N), d_s, 'b', label=r'$D_\leftrightarrow(d)$')
    ax2.tick_params(axis='y')
    lns = c1 + c2 + c3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)
    fig.tight_layout()
    if savefile is not None:
        pl.savefig(savefile)
    else:
        pl.show()


def plot_roc(rocs, names, title=None, figsize=[6.4, 4.0], savefile=None):
    fig = pl.figure(figsize=figsize)
    ax = fig.add_subplot()
    axins = ax.inset_axes([0.4, 0.35, 0.55, 0.55])
    x1, x2, y1, y2 = 0.05, 0.25, 0.75, 0.95
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    for ii in range(len(rocs)):
        ax.plot(rocs[ii][0], 1 - rocs[ii][1], label=names[ii])
        axins.plot(rocs[ii][0], 1 - rocs[ii][1], label=names[ii])
    pl.legend()
    ax.set_xlabel(r'$FMR$')
    ax.set_ylabel(r'$1-FNMR$')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    axins.set_xticks([0.05, 0.10, 0.15, 0.20, 0.25])
    axins.set_yticks([0.75, 0.80, 0.85, 0.90, 0.95])
    ax.indicate_inset_zoom(axins)
    if title is not None:
        pl.title(title)
    fig.tight_layout()    
    if savefile is not None:
        pl.savefig(savefile)
    else:
        pl.show()

def plot_det(rocs, names, title=None, figsize=[6.4, 4.0], savefile=None):
    fig = pl.figure(figsize=figsize)
    ax = fig.add_subplot()
    for ii in range(len(rocs)):
        ax.loglog(rocs[ii][0], rocs[ii][1], label=names[ii])
        #axins.loglog(rocs[ii][0], rocs[ii][1], label=names[ii])
    pl.legend(loc='lower left')
    pl.grid(which='both', alpha=0.3)
    ax.set_xlabel(r'$FMR$')
    ax.set_ylabel(r'$FNMR$')
    ax.set_xlim([0.001, 1.0])
    ax.set_ylim([0.001, 1.0])
    if title is not None:
        pl.title(title)
    fig.tight_layout()    
    if savefile is not None:
        pl.savefig(savefile)
    else:
        pl.show()