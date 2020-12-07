'''
Secure Triplet Loss Project Repository (https://github.com/jtrpinto/SecureTL)

File: models.py
- Defines the models: the unsecure models of ECG/face that can be trained with
  the original triplet loss; and the secure models (with cancellability keys)
  that can be trained with the proposed Secure Triplet Loss. The ECG model is
  based on our prior work [1] and the face model is based on the Inception 
  ResNet [2].

  References:
  [1] JR Pinto and JS Cardoso, "A end-to-end convolutional neural network for
      ECG based biometric authentication", in BTAS 2019.
  [2] C. Szegedy et al., "Inceptionv4, Inception-ResNet and the Impact of
      Residual Connections on Learning", in AAAI 2017.

"Secure Triplet Loss: Achieving Cancelability and Non-Linkability in End-to-End Deep Biometrics"
João Ribeiro Pinto, Miguel V. Correia, and Jaime S. Cardoso
IEEE Transactions on Biometrics, Behavior, and Identity Science

joao.t.pinto@inesctec.pt  |  https://jtrpinto.github.io
'''

import torch
from torch import nn
from torch.nn import functional as F


class TripletECGNetwork(nn.Module):
    # Defines the ECG network that processes a
    # single biometric sample. Is used with TripletModel
    # for training with the original triplet loss.

    def __init__(self, dropout_prob=0.5):
        # Defining the structure of the ECG network.
        super(TripletECGNetwork, self).__init__()
        self.convnet = nn.Sequential(nn.Conv1d(1, 16, 5),
                                     nn.ReLU(),
                                     nn.MaxPool1d(3, stride=3),
                                     nn.Conv1d(16, 16, 5),
                                     nn.ReLU(),
                                     nn.MaxPool1d(3, stride=3),
                                     nn.Conv1d(16, 32, 5),
                                     nn.ReLU(),
                                     nn.MaxPool1d(3, stride=3),
                                     nn.Conv1d(32, 32, 5),
                                     nn.ReLU(),
                                     nn.MaxPool1d(3, stride=3) )

        self.dropout = nn.Sequential(nn.Dropout(p=dropout_prob))

        self.fc = nn.Sequential(nn.Linear(320, 100),
                                nn.ReLU(),
                                nn.Linear(100, 100),
                                nn.ReLU() )

    def forward(self, x):
        # Network's inference routine.
        h = self.convnet(x)
        h = h.view(h.size()[0], -1)
        h = self.dropout(h)
        output = self.fc(h)
        return output

    def get_embedding(self, x):
        # To get an embedding (template).
        return self.forward(x)


class TripletFaceNetwork(nn.Module):
    # Defines the face network that processes a
    # single biometric sample. Is used with TripletModel
    # for training with the original triplet loss.

    def __init__(self, pretrained_model, dropout_prob=0.5):
        # Defining the structure of the network, based on
        # the Inception-ResNet model with pretrained weights.
        super(TripletFaceNetwork, self).__init__()
        self.conv2d_1a = pretrained_model.conv2d_1a
        self.conv2d_2a = pretrained_model.conv2d_2a
        self.conv2d_2b = pretrained_model.conv2d_2b
        self.maxpool_3a = pretrained_model.maxpool_3a
        self.conv2d_3b = pretrained_model.conv2d_3b
        self.conv2d_4a = pretrained_model.conv2d_4a
        self.conv2d_4b = pretrained_model.conv2d_4b
        self.repeat_1 = pretrained_model.repeat_1
        self.mixed_6a = pretrained_model.mixed_6a
        self.repeat_2 = pretrained_model.repeat_2
        self.mixed_7a = pretrained_model.mixed_7a
        self.repeat_3 = pretrained_model.repeat_3
        self.block8 = pretrained_model.block8
        self.avgpool_1a = pretrained_model.avgpool_1a
        self.relu = nn.Sequential(nn.ReLU())
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Sequential(nn.Linear(1792, 100), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(100, 100), nn.ReLU())

    def forward(self, x):
        # Network's inference routine.
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        x = self.repeat_1(x)
        x = self.mixed_6a(x)
        x = self.repeat_2(x)
        x = self.mixed_7a(x)
        x = self.repeat_3(x)
        x = self.block8(x)
        x = self.avgpool_1a(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def get_embedding(self, x):
        # To get an embedding (template).
        return self.forward(x)
        
    def freeze_parameters(self):
        # Freezes parameters in the first part of the
        # network to retain VGGFace2 pretrained weights.
        for param in self.conv2d_1a.parameters():
            param.requires_grad = False
        for param in self.conv2d_2a.parameters():
            param.requires_grad = False
        for param in self.conv2d_2b.parameters():
            param.requires_grad = False
        for param in self.conv2d_3b.parameters():
            param.requires_grad = False
        for param in self.conv2d_4a.parameters():
            param.requires_grad = False
        for param in self.conv2d_4b.parameters():
            param.requires_grad = False
        for param in self.repeat_1.parameters():
            param.requires_grad = False
        for param in self.mixed_6a.parameters():
            param.requires_grad = False
        for param in self.repeat_2.parameters():
            param.requires_grad = False
        for param in self.mixed_7a.parameters():
            param.requires_grad = False
        for param in self.repeat_3.parameters():
            param.requires_grad = False

    def unfreeze_parameters(self):
        # Unfreezes the first part of the network.
        for param in self.conv2d_1a.parameters():
            param.requires_grad = True
        for param in self.conv2d_2a.parameters():
            param.requires_grad = True
        for param in self.conv2d_2b.parameters():
            param.requires_grad = True
        for param in self.conv2d_3b.parameters():
            param.requires_grad = True
        for param in self.conv2d_4a.parameters():
            param.requires_grad = True
        for param in self.conv2d_4b.parameters():
            param.requires_grad = True
        for param in self.repeat_1.parameters():
            param.requires_grad = True
        for param in self.mixed_6a.parameters():
            param.requires_grad = True
        for param in self.repeat_2.parameters():
            param.requires_grad = True
        for param in self.mixed_7a.parameters():
            param.requires_grad = True
        for param in self.repeat_3.parameters():
            param.requires_grad = True


class SecureECGNetwork(nn.Module):
    # Defines the ECG secure network that processes a
    # single biometric sample and a key. Is used with
    # SecureModel for training with Secure Triplet Loss.

    def __init__(self):
        # Defining the structure of the ECG secure network.
        super(SecureECGNetwork, self).__init__()
        self.convnet = nn.Sequential(nn.Conv1d(1, 16, 5),
                                     nn.ReLU(),
                                     nn.MaxPool1d(3, stride=3),
                                     nn.Conv1d(16, 16, 5),
                                     nn.ReLU(),
                                     nn.MaxPool1d(3, stride=3),
                                     nn.Conv1d(16, 32, 5),
                                     nn.ReLU(),
                                     nn.MaxPool1d(3, stride=3),
                                     nn.Conv1d(32, 32, 5),
                                     nn.ReLU(),
                                     nn.MaxPool1d(3, stride=3) )

        self.dropout = nn.Sequential(nn.Dropout(p=DROPOUT))

        self.fc = nn.Sequential(nn.Linear(420, 100),
                                nn.ReLU(),
                                nn.Linear(100, 100),
                                nn.ReLU() )

    def forward(self, x, k):
        # Network's inference routine.
        h = self.convnet(x)
        h = h.view(h.size()[0], -1)
        h = self.dropout(h)
        h = torch.cat((h, k), dim=1)
        output = self.fc(h)
        return output

    def get_embedding(self, x, k):
        # To get a secure embedding (template).
        return self.forward(x, k)


class SecureFaceNetwork(nn.Module):
    # Defines the face secure network that processes a
    # single biometric sample and a key. Is used with
    # SecureModel for training with Secure Triplet Loss.

    def __init__(self, pretrained_model, dropout_prob=0.5):
        # Defining the structure of the secure network, based on
        # the Inception-ResNet model with pretrained weights.
        super(SecureFaceNetwork, self).__init__()
        self.conv2d_1a = pretrained_model.conv2d_1a
        self.conv2d_2a = pretrained_model.conv2d_2a
        self.conv2d_2b = pretrained_model.conv2d_2b
        self.maxpool_3a = pretrained_model.maxpool_3a
        self.conv2d_3b = pretrained_model.conv2d_3b
        self.conv2d_4a = pretrained_model.conv2d_4a
        self.conv2d_4b = pretrained_model.conv2d_4b
        self.repeat_1 = pretrained_model.repeat_1
        self.mixed_6a = pretrained_model.mixed_6a
        self.repeat_2 = pretrained_model.repeat_2
        self.mixed_7a = pretrained_model.mixed_7a
        self.repeat_3 = pretrained_model.repeat_3
        self.block8 = pretrained_model.block8
        self.avgpool_1a = pretrained_model.avgpool_1a
        self.relu = nn.Sequential(nn.ReLU())
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Sequential(nn.Linear(1892, 100), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(100, 100), nn.ReLU())

    def forward(self, x, k):
        # Network's inference routine.
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        x = self.repeat_1(x)
        x = self.mixed_6a(x)
        x = self.repeat_2(x)
        x = self.mixed_7a(x)
        x = self.repeat_3(x)
        x = self.block8(x)
        x = self.avgpool_1a(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        x = torch.cat((x, k), dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def get_embedding(self, x, k):
        # To get a secure embedding (template).
        return self.forward(x, k)
        
    def freeze_parameters(self):
        # Freezes parameters in the first part of the
        # network to retain VGGFace2 pretrained weights.
        for param in self.conv2d_1a.parameters():
            param.requires_grad = False
        for param in self.conv2d_2a.parameters():
            param.requires_grad = False
        for param in self.conv2d_2b.parameters():
            param.requires_grad = False
        for param in self.conv2d_3b.parameters():
            param.requires_grad = False
        for param in self.conv2d_4a.parameters():
            param.requires_grad = False
        for param in self.conv2d_4b.parameters():
            param.requires_grad = False
        for param in self.repeat_1.parameters():
            param.requires_grad = False
        for param in self.mixed_6a.parameters():
            param.requires_grad = False
        for param in self.repeat_2.parameters():
            param.requires_grad = False
        for param in self.mixed_7a.parameters():
            param.requires_grad = False
        for param in self.repeat_3.parameters():
            param.requires_grad = False

    def unfreeze_parameters(self):
        # Unfreezes the first part of the network.
        for param in self.conv2d_1a.parameters():
            param.requires_grad = True
        for param in self.conv2d_2a.parameters():
            param.requires_grad = True
        for param in self.conv2d_2b.parameters():
            param.requires_grad = True
        for param in self.conv2d_3b.parameters():
            param.requires_grad = True
        for param in self.conv2d_4a.parameters():
            param.requires_grad = True
        for param in self.conv2d_4b.parameters():
            param.requires_grad = True
        for param in self.repeat_1.parameters():
            param.requires_grad = True
        for param in self.mixed_6a.parameters():
            param.requires_grad = True
        for param in self.repeat_2.parameters():
            param.requires_grad = True
        for param in self.mixed_7a.parameters():
            param.requires_grad = True
        for param in self.repeat_3.parameters():
            param.requires_grad = True


class TripletModel(nn.Module):
    # Defines the model to be trained with the
    # original triplet loss. Can be based on either
    # TripletECGNetwork or TripletFaceNetwork.

    def __init__(self, network):
        super(TripletModel, self).__init__()
        self.network = network

    def forward(self, xA, xP, xN):
        # Triplet inference routine.
        output_a = self.network(xA)
        output_p = self.network(xP)
        output_n = self.network(xN)
        return output_a, output_p, output_n

    def get_embedding(self, x):
        # To get an embedding (template).
        return self.network(x)


class SecureModel(nn.Module):
    # Defines the model to be trained with the
    # Secure Triplet Loss. Can be based on either
    # SecureECGNetwork or SecureFaceNetwork.

    def __init__(self, network):
        super(SecureModel, self).__init__()
        self.network = network

    def forward(self, xA, xP, xN, k1, k2):
        # Secure triplet inference routine.
        output_a = self.network(xA, k1)
        output_p1 = self.network(xP, k1)
        output_p2 = self.network(xP, k2)
        output_n1 = self.network(xN, k1)
        output_n2 = self.network(xN, k2)
        return output_a, output_p1, output_p2, output_n1, output_n2

    def get_embedding(self, x, k):
        # To get a secure embedding (template).
        return self.network(x, k)
