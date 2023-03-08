import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import glob


class PackedDataset(Dataset):
    def __init__(self, files,label):
        self.files = files
        self.label = label
        self.len = len(files)

    def __getitem__(self, idx):
        packed = np.load(self.files[idx])
        x = packed[:,:-2]
        if self.label == 'breed':
            y = packed[:,-2:-1].astype(int)
            y = torch.from_numpy(y).to(torch.long)
        elif self.label == 'age':
            y = packed[:,-1:].astype(int)
            y = torch.from_numpy(y).to(torch.float32)
        x = torch.from_numpy(x).to(torch.float32)
        return x,y.squeeze()

    def __len__(self):
        return self.len

def dimFix(t):
    return torch.cat([x for x in t])