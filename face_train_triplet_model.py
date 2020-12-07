'''
Secure Triplet Loss Project Repository (https://github.com/jtrpinto/SecureTL)

File: face_train_triplet_model.py
- Used to train a model with face data, with the original triplet loss.

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
from models import TripletModel, TripletFaceNetwork
from losses import TripletLoss
from dataset import TripletFaceDataset
from trainer import train_triplet_model
from torch.utils.data import DataLoader
from facenet_pytorch import InceptionResnetV1


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

SAVE_MODEL = 'model_name'
TRAIN_SET = 'face_train_data.npy'

LEARN_RATE = 1e-4     # learning rate
REG = 0.001           # L2 regularization hyperparameter

N_EPOCHS = 500
BATCH_SIZE = 32
VALID_SPLIT = .2 
PATIENCE = 50

print('Training model: ' + SAVE_MODEL)

# Preparing and dividing the dataset
trainset = TripletFaceDataset(TRAIN_SET)

dataset_size = len(trainset)  # number of samples in training + validation sets
indices = list(range(dataset_size))
split = int(np.floor(VALID_SPLIT * dataset_size))  # number of samples in validation set
np.random.seed(42)
np.random.shuffle(indices)
train_indices, valid_indices = indices[split:], indices[:split]

train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)

train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, sampler=train_sampler)
valid_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, sampler=valid_sampler)

# Creating the network and the model
pretrained = InceptionResnetV1(pretrained='vggface2')
network = TripletFaceNetwork(pretrained, dropout_prob=0.6).to(DEVICE)
network.freeze_parameters()  # Freezing all parameters except the fully-connected layer(s)
model = TripletModel(network)
loss = TripletLoss(margin=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE, weight_decay=REG)

# Training the model
train_hist, valid_hist = train_triplet_model(model, loss, optimizer, train_loader, N_EPOCHS, BATCH_SIZE, DEVICE, patience=PATIENCE, valid_loader=valid_loader, filename=SAVE_MODEL)


