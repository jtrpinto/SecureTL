'''
Secure Triplet Loss Project Repository (https://github.com/jtrpinto/SecureTL)

File: face_test_secure_model.py
- Uses the face test data to evaluate models trained with any of the Secure
  Triplet Loss formulations, with or without linkability.

  REQUIRES:
  - facenet_pytorch package by Tim Esler
    (https://github.com/timesler/facenet-pytorch)

"Secure Triplet Loss: Achieving Cancelability and Non-Linkability in End-to-End Deep Biometrics"
João Ribeiro Pinto, Miguel V. Correia, and Jaime S. Cardoso
IEEE Transactions on Biometrics, Behavior, and Identity Science

joao.t.pinto@inesctec.pt  |  https://jtrpinto.github.io
'''

import os
import torch
import numpy as np
import pickle as pk
from models import SecureModel, SecureFaceNetwork
from losses import SecureTripletLoss
from dataset import SecureFaceDataset
from trainer import train_secure_triplet_model
from torch.utils.data import DataLoader
from eval import evaluate_secure_model
from facenet_pytorch import InceptionResnetV1


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

MODEL = 'model_name'
TEST_SET = 'face_test_data.npy'

BATCH_SIZE = 32 

print('Testing model: ' + MODEL)

# Preparing the dataset
testset = SecureFaceDataset(TEST_SET)
test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Creating the model
pretrained = InceptionResnetV1(pretrained='vggface2')
network = SecureFaceNetwork(pretrained).to(DEVICE)
model = SecureModel(network)

# Locading saved weights
model.load_state_dict(torch.load(MODEL + '.pth', map_location=DEVICE))
model = model.to(DEVICE)

# Evaluating the model on test data
output = evaluate_secure_model(model, test_loader, BATCH_SIZE, DEVICE, debug=True, N=10000, output_shape=100)

# Saving the results to a pickle on the 'results' folder
with open(os.path.basename(MODEL) + '_results.pk', 'wb') as hf:
    pk.dump(output, hf)

# Printing the main results
print('EER {:.4f} at threshold {:.4f} :: Canc_EER {:.4f} at threshold {:.4f} :: d_sys {:.4f}'.format(output[0]['eer'][1], output[0]['eer'][0], output[1]['eer'][1], output[1]['eer'][0], output[2][0]))
print('Privacy Leakage Rate {:.4f}, Secrecy Leakage {:.4f}'.format(output[3]['plr'], output[3]['sl']))


