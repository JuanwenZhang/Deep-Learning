#%%
import pandas
import torch
from loader import PackedDataset
from trainUtil import get_accuracy, get_loss, train, splitPacks
from catModel import CatAgeANN, CatBreedANN
import time
import torch.optim as optim 
import torch.nn as nn

#%%
net = CatAgeANN()
TR, VAL, TS = splitPacks("./featureOut",randseed=942)
trainDS = PackedDataset(TR, 'age')
valDS = PackedDataset(VAL, 'age')
optimizer = optim.Adam(net.parameters(),lr=0.001)
criterion = nn.MSELoss()
# %%
train(net, str(int(time.time())), trainDS, valDS, 30, 20, optimizer, criterion, get_loss)

#%%
net = CatBreedANN()
TR, VAL, TS = splitPacks("./featureOut",randseed=942)
trainDS = PackedDataset(TR, 'breed')
valDS = PackedDataset(VAL, 'breed')
optimizer = optim.Adam(net.parameters(),lr=0.001, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()
# %%
train(net, str(int(time.time())), trainDS, valDS, 4, 20, optimizer, criterion, get_accuracy)
# %%
