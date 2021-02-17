#
import torch
from torch.utils.data import Dataset, DataLoader
# 
from apps.mgc.gtzan_data_source import GtzanDataSource

class GtzanDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X_raw = self.X[idx]
        X = X_raw.reshape((X_raw.shape[0]*X_raw.shape[1],))
        #y = torch.zeros(10)
        #y[self.y[idx]] = 1.0
        return {'X': X, 'y': self.y[idx]}