'''
Secure Triplet Loss Project Repository (https://github.com/jtrpinto/SecureTL)

File: eval.py
- Defines various functions to evaluate verification performance, cancellability,
  non-linkability, and non-invertibility of the models.

  REQUIRES:
  - entropy_estimators package by Paul Brodersen
    (https://github.com/paulbrodersen/entropy_estimators)

"Secure Triplet Loss: Achieving Cancelability and Non-Linkability in End-to-End Deep Biometrics"
João Ribeiro Pinto, Miguel V. Correia, and Jaime S. Cardoso
IEEE Transactions on Biometrics, Behavior, and Identity Science

joao.t.pinto@inesctec.pt  |  https://jtrpinto.github.io
'''

import torch
import numpy as np
import pickle as pk
import scipy.stats as stats
from entropy_estimators import continuous


def get_triplet_outputs(model, data_loader, batch_size, device, output_shape):
    # Gets triplet embeddings from a dataset (given by the data_loader), using
    # a TripletModel, to be used for evaluation.
    # Note: input_A is returned as it is needed for non-invertibility evaluation.
    input_A = list()
    output_A = np.zeros((0, output_shape))
    output_P = np.zeros((0, output_shape))
    output_N = np.zeros((0, output_shape))
    with torch.no_grad():  # we do not need gradients
        model.eval()  # set the model to inference mode
        for i, (xA, xP, xN) in enumerate(data_loader):
            # copy the mini-batch to GPU
            xA = xA.to(device, dtype=torch.float)
            xP = xP.to(device, dtype=torch.float)
            xN = xN.to(device, dtype=torch.float)
            yA, yP, yN = model(xA, xP, xN)
            output_A = np.concatenate((output_A, yA.cpu().numpy()))
            output_P = np.concatenate((output_P, yP.cpu().numpy()))
            output_N = np.concatenate((output_N, yN.cpu().numpy()))
            input_A.append(xA.cpu().numpy())
    input_A = np.concatenate(input_A)
    return output_A, output_P, output_N, input_A


def get_secure_outputs(model, data_loader, batch_size, device, output_shape):
    # Gets secure embeddings from a dataset (given by the data_loader), using
    # a SecureModel, to be used for evaluation.
    # Note: input_A and input_k1 are returned as they are needed for
    # non-invertibility evaluation.
    input_A = list()
    input_k1 = list()
    output_A = np.zeros((0, output_shape))
    output_P1 = np.zeros((0, output_shape))
    output_P2 = np.zeros((0, output_shape))
    output_N1 = np.zeros((0, output_shape))
    output_N2 = np.zeros((0, output_shape))
    with torch.no_grad():  # we do not need gradients
        model.eval()  # set the model to inference mode
        for i, (xA, xP, xN, k1, k2) in enumerate(data_loader):
            # copy the mini-batch to GPU
            xA = xA.to(device, dtype=torch.float)
            xP = xP.to(device, dtype=torch.float)
            xN = xN.to(device, dtype=torch.float)
            k1 = k1.to(device, dtype=torch.float)
            k2 = k2.to(device, dtype=torch.float)
            yA, yP1, yP2, yN1, yN2 = model(xA, xP, xN, k1, k2)
            output_A = np.concatenate((output_A, yA.cpu().numpy()))
            output_P1 = np.concatenate((output_P1, yP1.cpu().numpy()))
            output_P2 = np.concatenate((output_P2, yP2.cpu().numpy()))
            output_N1 = np.concatenate((output_N1, yN1.cpu().numpy()))
            output_N2 = np.concatenate((output_N2, yN2.cpu().numpy()))
            input_A.append(xA.cpu().numpy())
            input_k1.append(k1.cpu().numpy())
    input_A = np.concatenate(input_A)
    input_k1 = np.concatenate(input_k1)
    return output_A, output_P1, output_P2, output_N1, output_N2, input_A, input_k1


def compute_distances_triplet(yA, yP, yN):
    # Computes distances between triplet embeddings.
    yA = yA.cpu().numpy()
    dp = normalised_distance(yA, yP.cpu().numpy())
    dn = normalised_distance(yA, yN.cpu().numpy())
    return np.array([dp, dn])


def compute_distances_secure(yA, yP1, yP2, yN1, yN2):
    # Computes distances between secure triplet embeddings.
    yA = yA.cpu().numpy()
    dsp = normalised_distance(yA, yP1.cpu().numpy())
    ddp = normalised_distance(yA, yP2.cpu().numpy())
    dsn = normalised_distance(yA, yN1.cpu().numpy())
    ddn = normalised_distance(yA, yN2.cpu().numpy())
    return np.array([dsp, ddp, dsn, ddn])


def evaluate_triplet_model(model, data_loader, batch_size, device, debug=False, output_shape=100, N=1000, save_embeddings=False):
    # Gets triplets from a data_loader and sends to triplet_metrics to compute triplet loss metrics.
    yA, yP, yN, xA = get_triplet_outputs(model, data_loader, batch_size, device, output_shape)
    if save_embeddings:
        with open('embeddings.pk', 'wb') as hf:
            pk.dump((yA, yP, yN), hf)
    positives = normalised_distance(yA, yP)
    negatives = normalised_distance(yA, yN)
    distances = np.array([positives, negatives])
    out = (triplet_metrics(distances, debug=debug, N=N),)  # Transformed into a tuple
    out += (triplet_it_measures(xA, yA, subset=1000),)
    return out


def evaluate_secure_model(model, data_loader, batch_size, device, debug=False, output_shape=100, N=1000, save_embeddings=False):
    # Gets secure triplets from a data_loader and sends to secure_metrics to compute SecureTL metrics.
    yA, yP1, yP2, yN1, yN2, xA, k1 = get_secure_outputs(model, data_loader, batch_size, device, output_shape)
    if save_embeddings:
        with open('embeddings.pk', 'wb') as hf:
            pk.dump((yA, yP1, yP2, yN1, yN2), hf)
    positives_samekey = normalised_distance(yA, yP1)
    positives_diffkey = normalised_distance(yA, yP2)
    negatives_samekey = normalised_distance(yA, yN1)
    negatives_diffkey = normalised_distance(yA, yN2)
    distances = np.array([positives_samekey, positives_diffkey, negatives_samekey, negatives_diffkey])
    out = secure_metrics(distances, debug=debug, N=N)
    out += (secure_it_measures(xA, yA, k1, subset=1000),)
    return out


def triplet_metrics(distances, debug=False, N=1000):
    # Computes EER based on triplet loss distances.
    predictions = np.concatenate((distances[0], distances[1]))
    labels = np.concatenate((np.zeros((distances.shape[1],)), np.ones((distances.shape[1],))))
    result = evaluate_eer(predictions, labels, n=N)
    if debug:
        return result
    else:
        return result['eer']


def secure_metrics(distances, debug=False, N=1000):
    # Computes performance EER, cancelability EER, and linkability metrics based on SecureTL distances
    # Performance EER (dSP, dSN).
    predictions = np.concatenate((distances[0], distances[2]))
    labels = np.concatenate((np.zeros((distances.shape[1],)), np.ones((distances.shape[1],))))
    result_eer = evaluate_eer(predictions, labels, n=N)
    # Cancelability EER (dSP, dDP)
    predictions = np.concatenate((distances[0], distances[1]))
    labels = np.concatenate((np.zeros((distances.shape[1],)), np.ones((distances.shape[1],))))
    result_cancelability = evaluate_eer(predictions, labels, n=N)
    if debug:
        # Linkability (dDP, dDN)
        link = linkability(distances[1], distances[3], N=N, debug=True)
        return result_eer, result_cancelability, link
    else:
        # Linkability (dDP, dDN)
        d_sys = linkability(distances[1], distances[3], N=N)
        return result_eer['eer'][0], result_eer['eer'][1], result_cancelability['eer'][0], result_cancelability['eer'][1], d_sys


def evaluate_eer(predictions, labels, pos_label=0, smin=0, smax=1,
                 n=1000, positive='lower'):
    roc = roc_curve(labels, predictions, pos_label=pos_label, smin=smin,
                    smax=smax, n=n, positive=positive)
    eer = determine_equal_error_rate(roc)
    return {'eer': eer, 'roc': roc}


def linkability(mated, non_mated, N=1000, debug=False):
    # Computes D_s and D_sys based on mated and non-mated distances, to evaluate
    # template linkability. Set debug=False to return only D_sys, or debug=True
    # to return also D_s, p_sHm, and p_sHnm.
    # Based on M. Gomez-Barrero et al., "Unlinkable and irreversible biometric
    # template protection based on bloom filters", Information Sciences 370-371,
    # pp. 18–32, 2016.

    s_Hm = sorted(mated)  # dDP
    s_Hnm = sorted(non_mated)  # dDN

    p_sHm = stats.norm.pdf(np.linspace(0, 1, num=N), np.mean(s_Hm), np.std(s_Hm))
    p_sHnm = stats.norm.pdf(np.linspace(0, 1, num=N), np.mean(s_Hnm), np.std(s_Hnm)) + 1e-15  # Avoid divide by zero

    # Compute D_{<->}(s)
    lr_s = p_sHm/p_sHnm
    d_s = np.maximum((np.power(np.exp(-(lr_s - 1)) + 1, -1) - 0.5) * 2, 0)

    # Compute D_{<->}^{sys} 
    d_sys = np.trapz(d_s * p_sHm, dx=1.0/N)

    # Only return more than 
    if debug:
        return d_sys, d_s, p_sHm, p_sHnm
    else:
        return d_sys


def triplet_it_measures(xA, yA, k=10, subset=None):
    # Uses the entropy_estimators package to estimate information
    # theoretical security metrics.
    np.random.seed(42)  # Ensuring reproducibility
    if subset is not None:  # subset chooses only N random samples to speed up the process
        indices = np.random.choice(range(len(xA)), size=subset, replace=False)
        input = xA[indices]
        output = yA[indices]
    else:
        input = xA
        output = yA
    input = np.reshape(input, (len(input), -1))
    mi_xy = continuous.get_mi(input, output, k=k)
    h_x = continuous.get_h(input, k=k) 
    out = {'plr': 1.0 - mi_xy/h_x}
    return out


def secure_it_measures(xA, yA, k1, k=10, subset=None):
    # Uses the entropy_estimators package to estimate information
    # theoretical security metrics.
    np.random.seed(42)  # Ensuring reproducibility
    if subset is not None:  # subset chooses only N random samples to speed up the process
        indices = np.random.choice(range(len(xA)), size=subset, replace=False)
        input = xA[indices]
        output = yA[indices]
        keys = k1[indices]
    else:
        input = xA
        output = yA
        keys = k1   
    input = np.reshape(input, (len(input), -1))
    mi_xy = continuous.get_mi(input, output, k=k)
    h_x = continuous.get_h(input, k=k)
    mi_ky = continuous.get_mi(keys, output, k=k)
    out = {'plr': 1.0 - mi_xy/h_x, 'sl': mi_ky}
    return out


def determine_equal_error_rate(roc):
    # Computes the EER for a certain Receiver Operating Characteristic curve.
    fpr, fnr, thr = roc
    diff = fpr - fnr
    abs_diff = np.absolute(diff)
    min_diff = np.min(abs_diff)
    min_loc = np.argmin(abs_diff)
    if min_diff == 0:
        return (thr[min_loc], fpr[min_loc])
    elif diff[0] * diff[-1] > 0:
        print('Warning! FPR and FNR lines do not intersect.')
        return (np.nan, np.nan)
    else:
        if diff[np.argmin(abs_diff)] > 0:
            idx1 = min_loc - 1
            idx2 = min_loc
        else:
            idx1 = min_loc
            idx2 = min_loc + 1
        if diff[idx1] * diff[idx2] >= 0:
            idx1, idx2 = escape_parallel_plateaux(diff)
        A1 = [thr[idx1], fpr[idx1]]
        B1 = [thr[idx1], fnr[idx1]]
        A2 = [thr[idx2], fpr[idx2]]
        B2 = [thr[idx2], fnr[idx2]]
        lA = line(A1, A2)
        lB = line(B1, B2)
        eer = intersection(lA, lB)
        if eer is False:
            return (np.nan, np.nan)
        else:
            return eer


def roc_curve(labels, predictions, pos_label=0, smin=0, smax=1,
              n=1000, positive='lower'):
    # Computes a Receiver Operating Characteristic curve based on
    # true labels and prediction scores.
    positives = predictions[labels == pos_label]
    negatives = predictions[labels != pos_label]
    n_positives = len(positives)
    n_negatives = len(negatives)
    fpr = np.zeros((n,))
    fnr = np.zeros((n,))
    thr = np.linspace(smin, smax, num=n)
    if positive == 'lower':
        for tt in range(n):
            fpr[tt] = np.sum(negatives <= thr[tt])/n_negatives
            fnr[tt] = np.sum(positives > thr[tt])/n_positives
    elif positive == 'higher':
        for tt in range(n):
            fpr[tt] = np.sum(negatives >= thr[tt])/n_negatives
            fnr[tt] = np.sum(positives < thr[tt])/n_positives
    else:
        raise ValueError('Parameter \'positive\' can only be \'lower\' or \'higher\'.')
    return fpr, fnr, thr


def normalised_distance(yA, yX):
    # Normalised Euclidean distance between two embeddings.
    var_A = np.var(yA, axis=1)
    var_X = np.var(yX, axis=1)
    var_AX = np.var(yA-yX, axis=1)
    dist = 0.5 * var_AX / (var_A + var_X)  # d = 0.5 * var(x - y) / (var(x) + var(y))
    return dist


def line(p1, p2):
    # Computes line coefficients for intersection determination.
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C


def intersection(L1, L2):
    # Returns the intersection point between two lines L1 and L2.
    D = L1[0]*L2[1] - L1[1]*L2[0]
    Dx = L1[2]*L2[1] - L1[1]*L2[2]
    Dy = L1[0]*L2[2] - L1[2]*L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x, y
    else:
        return False


def escape_parallel_plateaux(diff):
    # Procedure to escale parallel plateaux.
    diff_t = diff[1:len(diff)] * diff[0:len(diff)-1]
    idx1 = np.argmin(diff_t)
    idx2 = np.argmin(diff_t) + 1
    return (idx1, idx2)
