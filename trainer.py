'''
Secure Triplet Loss Project Repository (https://github.com/jtrpinto/SecureTL)

File: trainer.py
- Defines the training routines for the original triplet loss and for the proposed
  Secure Triplet Loss.

"Secure Triplet Loss: Achieving Cancelability and Non-Linkability in End-to-End Deep Biometrics"
João Ribeiro Pinto, Miguel V. Correia, and Jaime S. Cardoso
IEEE Transactions on Biometrics, Behavior, and Identity Science

joao.t.pinto@inesctec.pt  |  https://jtrpinto.github.io
'''

import torch
import numpy as np
import sys
import pickle
from eval import compute_distances_triplet, compute_distances_secure, triplet_metrics, secure_metrics


def train_triplet_model(model, loss_fn, optimizer, train_loader, n_epochs, batch_size, device, patience=1, valid_loader=None, filename=None):
    train_hist = []
    train_eer = []
    valid_hist = []
    valid_eer = []

    # For early stopping:
    plateau = 0  
    best_valid_loss = None

    for epoch in range(n_epochs):
        print('Epoch {}'.format(epoch + 1))
    
        # training loop
        model.train()  # set model to training mode (affects dropout and batch norm.)
        for i, (xA, xP, xN) in enumerate(train_loader):
            # copy the mini-batch to GPU
            xA = xA.to(device, dtype=torch.float)
            xP = xP.to(device, dtype=torch.float)
            xN = xN.to(device, dtype=torch.float)
            
            yA, yP, yN = model(xA, xP, xN)    # forward pass
            loss = loss_fn(yA, yP, yN)  # compute the loss
            optimizer.zero_grad()     # set all gradients to zero (otherwise they are accumulated)
            loss.backward()           # backward pass (i.e. compute gradients)
            optimizer.step()          # update the parameters
        
            # display the mini-batch loss
            sys.stdout.write("\r" + '........mini-batch no. {} loss: {:.4f}'.format(i+1, loss.item()))
            sys.stdout.flush()
        
            if torch.isnan(loss):
                print('NaN loss. Terminating train.')
                return [], []

        # compute the training and validation losses to monitor the training progress (optional)
        print()
        with torch.no_grad():  # now we are doing inference only, so we do not need gradients
            model.eval()       # set model to inference mode (affects dropout and batch norm.)
        
            train_loss = 0.
            distances = np.zeros((2, 0))
            for i, (xA, xP, xN) in enumerate(train_loader):
                # copy the mini-batch to GPU
                xA = xA.to(device, dtype=torch.float)
                xP = xP.to(device, dtype=torch.float)
                xN = xN.to(device, dtype=torch.float)
            
                yA, yP, yN = model(xA, xP, xN)      # forward pass
                train_loss += loss_fn(yA, yP, yN)  # accumulate the loss of the mini-batch
                distances = np.concatenate((distances, compute_distances_triplet(yA, yP, yN)), axis=1)
            train_loss /= i + 1
            train_hist.append(train_loss.item())
            _, t_eer = triplet_metrics(distances)
            train_eer.append(t_eer)
            print('....train loss: {:.4f} :: EER {:.4f}'.format(train_loss.item(), t_eer))
        
            if valid_loader is None:
                print()
                continue
        
            valid_loss = 0.
            distances = np.zeros((2, 0))
            for i, (xA, xP, xN) in enumerate(valid_loader):
                # copy the mini-batch to GPU
                xA = xA.to(device, dtype=torch.float)
                xP = xP.to(device, dtype=torch.float)
                xN = xN.to(device, dtype=torch.float)
            
                yA, yP, yN = model(xA, xP, xN)   # forward pass
                valid_loss += loss_fn(yA, yP, yN)  # accumulate the loss of the mini-batch
                distances = np.concatenate((distances, compute_distances_triplet(yA, yP, yN)), axis=1)
            valid_loss /= i + 1
            valid_hist.append(valid_loss.item())
            _, v_eer = triplet_metrics(distances)
            valid_eer.append(v_eer)
            print('....valid loss: {:.4f} :: EER {:.4f}'.format(valid_loss.item(), v_eer))

        # ATTENTION: CHANGES valid_loss FOR v_eer
        if best_valid_loss is None:
            best_valid_loss = v_eer
            torch.save(model.state_dict(), filename + '.pth')
            with open(filename + '_trainhist.pk', 'wb') as hf:
                pickle.dump({'loss': train_hist, 'eer': train_eer}, hf)
            with open(filename + '_validhist.pk', 'wb') as hf:
                pickle.dump({'loss': valid_hist, 'eer': valid_eer}, hf)
            print('....Saving...')
        elif v_eer < best_valid_loss:
            best_valid_loss = v_eer
            torch.save(model.state_dict(), filename + '.pth')
            with open(filename + '_trainhist.pk', 'wb') as hf:
                pickle.dump({'loss': train_hist, 'eer': train_eer}, hf)
            with open(filename + '_validhist.pk', 'wb') as hf:
                pickle.dump({'loss': valid_hist, 'eer': valid_eer}, hf)
            plateau = 0
            print('....Saving...')
        else:
            plateau += 1
            if plateau >= patience:
                print('....Early stopping the train.')
                return train_hist, valid_hist

    return train_hist, valid_hist


def train_secure_triplet_model(model, loss_fn, optimizer, train_loader, n_epochs, batch_size, device, patience=1, valid_loader=None, filename=None):
    train_hist = []
    train_eer = []
    train_canc = []
    train_dsys = []
    valid_hist = []
    valid_eer = []
    valid_canc = []
    valid_dsys = []

    # For early stopping:
    plateau = 0  
    best_valid_loss = None

    for epoch in range(n_epochs):
        print('Epoch {}'.format(epoch + 1))
    
        # training loop
        model.train()  # set model to training mode (affects dropout and batch norm.)
        for i, (xA, xP, xN, k1, k2) in enumerate(train_loader):
            # copy the mini-batch to GPU
            xA = xA.to(device, dtype=torch.float)
            xP = xP.to(device, dtype=torch.float)
            xN = xN.to(device, dtype=torch.float)
            k1 = k1.to(device, dtype=torch.float)
            k2 = k2.to(device, dtype=torch.float)
            
            yA, yP1, yP2, yN1, yN2 = model(xA, xP, xN, k1, k2)    # forward pass
            loss = loss_fn(yA, yP1, yP2, yN1, yN2)  # compute the loss
            optimizer.zero_grad()     # set all gradients to zero (otherwise they are accumulated)
            loss.backward()           # backward pass (i.e. compute gradients)
            optimizer.step()          # update the parameters
        
            # display the mini-batch loss
            sys.stdout.write("\r" + '........mini-batch no. {} loss: {:.4f}'.format(i + 1, loss.item()))
            sys.stdout.flush()

            if torch.isnan(loss):
                print('NaN loss. Terminating train.')
                return [], []
        
        # compute the training and validation losses to monitor the training progress (optional)
        print()
        with torch.no_grad():  # now we are doing inference only, so we do not need gradients
            model.eval()       # set model to inference mode (affects dropout and batch norm.)
        
            train_loss = 0.
            distances = np.zeros((4, 0))
            for i, (xA, xP, xN, k1, k2) in enumerate(train_loader):
                # copy the mini-batch to GPU
                xA = xA.to(device, dtype=torch.float)
                xP = xP.to(device, dtype=torch.float)
                xN = xN.to(device, dtype=torch.float)
                k1 = k1.to(device, dtype=torch.float)
                k2 = k2.to(device, dtype=torch.float)
            
                yA, yP1, yP2, yN1, yN2 = model(xA, xP, xN, k1, k2)    # forward pass
                train_loss += loss_fn(yA, yP1, yP2, yN1, yN2)  # accumulate the loss of the mini-batch
                distances = np.concatenate((distances, compute_distances_secure(yA, yP1, yP2, yN1, yN2)), axis=1)
            train_loss /= i + 1
            train_hist.append(train_loss.item())
            _, t_eer, _, t_canc, t_dsys = secure_metrics(distances)
            train_eer.append(t_eer)
            train_canc.append(t_canc)
            train_dsys.append(t_dsys)
            print('....train loss: {:.4f} :: EER {:.4f} :: Canc_EER {:.4f} :: D_sys {:.4f}'.format(train_loss.item(), t_eer, t_canc, t_dsys))
        
            if valid_loader is None:
                print()
                continue
        
            valid_loss = 0.
            distances = np.zeros((4, 0))
            for i, (xA, xP, xN, k1, k2) in enumerate(valid_loader):
                # copy the mini-batch to GPU
                xA = xA.to(device, dtype=torch.float)
                xP = xP.to(device, dtype=torch.float)
                xN = xN.to(device, dtype=torch.float)
                k1 = k1.to(device, dtype=torch.float)
                k2 = k2.to(device, dtype=torch.float)
            
                yA, yP1, yP2, yN1, yN2 = model(xA, xP, xN, k1, k2)    # forward pass
                valid_loss += loss_fn(yA, yP1, yP2, yN1, yN2)  # accumulate the loss of the mini-batch
                distances = np.concatenate((distances, compute_distances_secure(yA, yP1, yP2, yN1, yN2)), axis=1)
            valid_loss /= i + 1
            valid_hist.append(valid_loss.item())
            _, v_eer, _, v_canc, v_dsys = secure_metrics(distances)
            valid_eer.append(v_eer)
            valid_canc.append(v_canc)
            valid_dsys.append(v_dsys)
            print('....valid loss: {:.4f} :: EER {:.4f} :: Canc_EER {:.4f} :: D_sys {:.4f}'.format(valid_loss.item(), v_eer, v_canc, v_dsys))
            
        # Saving best model and early stopping:
        # ATTENTION: CHANGES valid_loss FOR v_eer    
        if best_valid_loss is None:
            best_valid_loss = v_eer
            torch.save(model.state_dict(), filename + '.pth')
            with open(filename + '_trainhist.pk', 'wb') as hf:
                pickle.dump({'loss': train_hist, 'eer': train_eer, 'canc': train_canc, 'dsys': train_dsys}, hf)
            with open(filename + '_validhist.pk', 'wb') as hf:
                pickle.dump({'loss': valid_hist, 'eer': valid_eer, 'canc': valid_canc, 'dsys': valid_dsys}, hf)
            print('....Saving...')
        elif v_eer < best_valid_loss:
            best_valid_loss = v_eer
            torch.save(model.state_dict(), filename + '.pth')
            with open(filename + '_trainhist.pk', 'wb') as hf:
                pickle.dump({'loss': train_hist, 'eer': train_eer, 'canc': train_canc, 'dsys': train_dsys}, hf)
            with open(filename + '_validhist.pk', 'wb') as hf:
                pickle.dump({'loss': valid_hist, 'eer': valid_eer, 'canc': valid_canc, 'dsys': valid_dsys}, hf)
            plateau = 0
            print('....Saving...')
        else:
            plateau += 1
            if plateau >= patience:
                print('....Early stopping the train.')
                return train_hist, valid_hist
            
    return train_hist, valid_hist
