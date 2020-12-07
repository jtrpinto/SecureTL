'''
Secure Triplet Loss Project Repository (https://github.com/jtrpinto/SecureTL)

File: losses.py
- Defines the loss function classes to be used during training: the original
  triplet loss [1], the original formulation of the Secure Triplet Loss [2],
  and the proposed SecureTL w/KLD and SecureTL w/SL.

  References:
  [1] G. Chechik et al., "Large scale online learning of image similarity
      through ranking", JMLR 11, pp. 1109–1135, 2010.
  [2] JR Pinto et al., "Secure Triplet Loss for End-to-End Deep Biometrics",
      in IWBF 2020.

"Secure Triplet Loss: Achieving Cancelability and Non-Linkability in End-to-End Deep Biometrics"
João Ribeiro Pinto, Miguel V. Correia, and Jaime S. Cardoso
IEEE Transactions on Biometrics, Behavior, and Identity Science

joao.t.pinto@inesctec.pt  |  https://jtrpinto.github.io
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence


class TripletLoss(nn.Module):
    # Original Triplet Loss

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        # Computing loss value based on a batch of triplet embeddings.
        distance_positive = torch.norm(anchor - positive, dim=1)
        distance_negative = torch.norm(anchor - negative, dim=1)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class SecureTripletLoss(nn.Module):
    # SecureTL (original formulation w/out linkability)

    def __init__(self, margin):
        super(SecureTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive1, positive2, negative1, negative2, size_average=True):
        # Computing loss value based on a batch of secure triplet embeddings.
        dSP = torch.norm(anchor - positive1, dim=1)
        dSN = torch.norm(anchor - negative1, dim=1)
        dDP = torch.norm(anchor - positive2, dim=1)
        dDN = torch.norm(anchor - negative2, dim=1)
        dneg = torch.min(dSN, dDP)
        dneg = torch.min(dneg, dDN)
        losses = F.relu(dSP - dneg + self.margin)
        return losses.mean() if size_average else losses.sum()


class SecureTripletLossKLD(nn.Module):
    # SecureTL w/KLD (with linkability through Kullback-Leibler Divergence)

    def __init__(self, margin, gamma):
        super(SecureTripletLossKLD, self).__init__()
        self.margin = margin
        self.gamma = gamma

    def forward(self, anchor, positive1, positive2, negative1, negative2):
        # Computing loss value based on a batch of secure triplet embeddings.
        
        # Original Secure Triplet Loss component:
        dSP = torch.norm(anchor - positive1, dim=1)
        dSN = torch.norm(anchor - negative1, dim=1)
        dDP = torch.norm(anchor - positive2, dim=1)
        dDN = torch.norm(anchor - negative2, dim=1)
        dneg = torch.min(dSN, dDP)
        dneg = torch.min(dneg, dDN)
        losses = F.relu(dSP - dneg + self.margin)
        loss = losses.mean()

        # Linkability component:
        dist_DP = Normal(torch.mean(dDP).view(1), torch.var(dDP).view(1))
        dist_DN = Normal(torch.mean(dDN).view(1), torch.var(dDN).view(1))
        linkability = kl_divergence(dist_DP, dist_DN)

        return self.gamma * loss + (1 - self.gamma) * linkability


class SecureTripletLossSL(nn.Module):
    # SecureTL w/SL (with linkability through Simple Statistics)

    def __init__(self, margin, gamma):
        super(SecureTripletLossSL, self).__init__()
        self.margin = margin
        self.gamma = gamma

    def forward(self, anchor, positive1, positive2, negative1, negative2):
        # Computing loss value based on a batch of secure triplet embeddings.

        # Original Secure Triplet Loss part
        dSP = torch.norm(anchor - positive1, dim=1)
        dSN = torch.norm(anchor - negative1, dim=1)
        dDP = torch.norm(anchor - positive2, dim=1)
        dDN = torch.norm(anchor - negative2, dim=1)
        dneg = torch.min(dSN, dDP)
        dneg = torch.min(dneg, dDN)
        losses = F.relu(dSP - dneg + self.margin)
        loss = losses.mean()

        # Linkability part
        mean_diff = torch.norm(torch.mean(dDP) - torch.mean(dDN))
        stdv_diff = torch.norm(torch.std(dDP) - torch.std(dDN))
        linkability = mean_diff + stdv_diff

        return self.gamma * loss + (1 - self.gamma) * linkability
