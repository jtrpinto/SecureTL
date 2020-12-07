'''
Secure Triplet Loss Project Repository (https://github.com/jtrpinto/SecureTL)

File: ecg_prepare_uoftdb.py
- Takes the University of Toronto ECG Database (UofTDB) and prepares everything to
  be used in the Secure Triplet Loss training and experiments.
  
  Attention: For best results, the original .mat file that carries all UofTDB data has
  been divided into multiple .txt files, with names 'uoftdb_SUB_SES_FILE.txt', where
  SUB is the subject ID, SES is the session number, and FILE is the posture number
  from the respective session.

"Secure Triplet Loss: Achieving Cancelability and Non-Linkability in End-to-End Deep Biometrics"
João Ribeiro Pinto, Miguel V. Correia, and Jaime S. Cardoso
IEEE Transactions on Biometrics, Behavior, and Identity Science

joao.t.pinto@inesctec.pt  |  https://jtrpinto.github.io
'''

import pickle
import numpy as np
import aux_functions as af


N_TRAIN = 100000   # Number of triplets for the train set
N_TEST = 10000     # Number of triplets for the test set

UOFTDB_PATH = 'UofTDB/txt'  # Path to UofTDB path with recordings divided by .txt files

SAVE_TRAIN = 'ecg_train_data.pickle'
SAVE_TEST = 'ecg_test_data.pickle'

fs = 200.0  # Sampling frequency of data

# Dividing subjects for training and for testing
train_data = af.extract_data(UOFTDB_PATH, range(921, 1021), fs=200.0)
test_data = af.extract_data(UOFTDB_PATH, range(921), fs=200.0)

# Preparing data for a deep neural network
X_train_a, y_train_a = af.prepare_for_dnn(train_data['X_anchors'], train_data['y_anchors'])
X_train_r, y_train_r = af.prepare_for_dnn(train_data['X_remaining'], train_data['y_remaining'])

X_test_a, y_test_a = af.prepare_for_dnn(test_data['X_anchors'], test_data['y_anchors'])
X_test_r, y_test_r = af.prepare_for_dnn(test_data['X_remaining'], test_data['y_remaining'])

train_triplets = af.generate_triplets(X_train_a, y_train_a, X_train_r, y_train_r, N=N_TRAIN)
test_triplets = af.generate_triplets(X_test_a, y_test_a, X_test_r, y_test_r, N=N_TEST)

with open(SAVE_TRAIN, 'wb') as handle:
    pickle.dump(train_triplets, handle)

with open(SAVE_TEST, 'wb') as handle:
    pickle.dump(test_triplets, handle)