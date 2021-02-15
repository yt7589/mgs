#
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
        return {'X': self.X[idx], 'y': self.y[idx]}