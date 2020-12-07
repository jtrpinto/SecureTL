'''
Secure Triplet Loss Project Repository (https://github.com/jtrpinto/SecureTL)

File: aux_functions.py
- Miscellaneous functions used to load and process face and ECG data.

"Secure Triplet Loss: Achieving Cancelability and Non-Linkability in End-to-End Deep Biometrics"
João Ribeiro Pinto, Miguel V. Correia, and Jaime S. Cardoso
IEEE Transactions on Biometrics, Behavior, and Identity Science

joao.t.pinto@inesctec.pt  |  https://jtrpinto.github.io
'''

import os
import cv2
import warnings
import numpy as np
from PIL import Image
from copy import deepcopy
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