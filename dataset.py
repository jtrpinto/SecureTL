'''
Secure Triplet Loss Project Repository (https://github.com/jtrpinto/SecureTL)

File: dataset.py
- Uses torch's Dataset class to import the Triplet or Secure datasets of ECG
  or Face to be used to train or test the models.

  Info on the dataset files needed:

  TripletFaceDataset and SecureFaceDataset require a numpy file,
  e.g. "face_data.npy", which stores a Nx3 or Nx5 numpy array,
  where the three first columns are the paths to the prepared images
  serving as anchor, positive, and negative samples, respectively.
  The fourth and fifth columns are the cancelability keys 1 and 2,
  respectively, not required for TripletFaceDataset.
  See "face_prepare_ytfdb.py" for more details.

  TripletECGDataset and SecureECGDataset require a pickle file,
  e.g. "ecg_data.pickle", which stores a tuple of six numpy arrays.
  The first three numpy arrays correspond to the anchors, the positives,
  and the negatives, respectively. The fourth array includes the IDs of
  the anchors on each triplet. The fifth and sixth arrays correspond to
  the cancelable keys of each triplet, k1 and k2, respectively.
  See "ecg_prepare_uoftdb.py" for more details.

"Secure Triplet Loss: Achieving Cancelability and Non-Linkability in End-to-End Deep Biometrics"
João Ribeiro Pinto, Miguel V. Correia, and Jaime S. Cardoso
IEEE Transactions on Biometrics, Behavior, and Identity Science

joao.t.pinto@inesctec.pt  |  https://jtrpinto.github.io
'''

import torch
import numpy as np
import pickle as pk
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class SecureECGDataset(Dataset):
    # Used to load ECG triplet data, including cancellability keys
    # for the SecureTL method.

    def __init__(self, data_path):
        with open(data_path, 'rb') as handle:
            data = pk.load(handle)
        self.xA = data[0]
        self.xP = data[1]
        self.xN = data[2]
        self.k1 = data[4]
        self.k2 = data[5]
        self.sample_shape = (1, 1000)

    def __getitem__(self, index):
        # Load anchor, positive, negative, and the two keys for a given index
        xA = self.xA[index].reshape(self.sample_shape)
        xP = self.xP[index].reshape(self.sample_shape)
        xN = self.xN[index].reshape(self.sample_shape)
        k1 = self.k1[index]
        k2 = self.k2[index]
        return (xA, xP, xN, k1, k2)

    def __len__(self):
        return len(self.xA)


class TripletECGDataset(Dataset):
    # Used to load ECG triplet data for the original triplet loss.

    def __init__(self, data_path):
        with open(data_path, 'rb') as handle:
            data = pk.load(handle)
        self.xA = data[0]
        self.xP = data[1]
        self.xN = data[2]
        self.sample_shape = (1, 1000)

    def __getitem__(self, index):
        # Load anchor, positive, and negative for a given index
        xA = self.xA[index].reshape(self.sample_shape)
        xP = self.xP[index].reshape(self.sample_shape)
        xN = self.xN[index].reshape(self.sample_shape)
        return (xA, xP, xN)

    def __len__(self):
        # Number of triplets on the dataset
        return len(self.xA)


class SecureFaceDataset(Dataset):
    # Used to load face triplet data, including cancellability keys
    # for the SecureTL method.

    def __init__(self, dataset):
        dset = np.load(dataset, allow_pickle=True)
        self.anchors = dset[:,0]
        self.positives = dset[:,1]
        self.negatives = dset[:,2]
        self.k1 = dset[:,3]
        self.k2 = dset[:,4]

    def __normalise__(self, img):
        # Normalise image pixel intensities
        return (img - 127.5) / 128.0

    def __loadimage__(self, imgpath):
        # Load image, apply d.a., transform to tensor, and normalise
        img = Image.open(imgpath)
        tf = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                 np.float32,
                                 transforms.ToTensor(),
                                 self.__normalise__])  # Adapted from facenet_pytorch
        img = tf(img)
        return img

    def __getitem__(self, index):
        # Load anchor, positive, negative, and the two keys for a given index
        xA = self.__loadimage__(self.anchors[index])
        xP = self.__loadimage__(self.positives[index])
        xN = self.__loadimage__(self.negatives[index])
        k1 = self.k1[index]
        k2 = self.k2[index]
        return (xA, xP, xN, k1, k2)

    def __len__(self):
        # Number of triplets on the dataset
        return len(self.anchors)


class TripletFaceDataset(Dataset):
    # Used to load face triplet data for the original triplet loss.

    def __init__(self, dataset):
        dset = np.load(dataset, allow_pickle=True)
        self.anchors = dset[:,0]
        self.positives = dset[:,1]
        self.negatives = dset[:,2]

    def __normalise__(self, img):
        return (img - 127.5) / 128.0

    def __loadimage__(self, imgpath):
        img = Image.open(imgpath)
        tf = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                 np.float32,
                                 transforms.ToTensor(),
                                 self.__normalise__])  # Adapted from facenet_pytorch
        img = tf(img)
        return img

    def __getitem__(self, index):
        # Load anchor, positive, and negative for a given index
        xA = self.__loadimage__(self.anchors[index])
        xP = self.__loadimage__(self.positives[index])
        xN = self.__loadimage__(self.negatives[index])
        return (xA, xP, xN)

    def __len__(self):
        # Number of triplets on the dataset
        return len(self.anchors)
