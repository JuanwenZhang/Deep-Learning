import glob
import random
import math
import os
import pandas
import torch
from loader import dimFix
from torch.utils.data import Dataset, DataLoader
import json
import torch.nn as nn

def splitPacks(packDir,ratios=[0.7,0.15,0.15],randseed=1):
    files = glob.glob(f"{packDir}/*.npy")
    random.seed(randseed)
    random.shuffle(files)
    breakPoints = [math.floor(len(files)*sum(ratios[:i])) for i in range(len(ratios)+1)]
    splits = [files[a:b] for a,b in list(zip(breakPoints[:-1],breakPoints[1:]))]
    return splits

def get_accuracy(model, dataset, criterion):
    correct = 0
    total = 0
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)
    for x, y in dataloader:
        x = dimFix(x)
        y = dimFix(y)
        output = model(x)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(y.view_as(pred)).sum().item()
        total += x.shape[0]
    return correct / total

def get_loss(model, dataset, criterion):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)
    losses = 0
    total = 0
    for x, y in dataloader:
        x = dimFix(x)
        y = dimFix(y)
        output = model(x)
        loss = criterion(output.squeeze(1), y)
        losses += float(loss)
        total += y.shape[0]
    return losses/total
    
def train(model, runId, trainData, valData, batch_size, num_epochs,optimizer,criterion, accFunc):
    mdPath = f'./run/{runId}/'
    os.makedirs(mdPath,exist_ok=True)
    print(f"Run {runId} started")
    train_loader = torch.utils.data.DataLoader(trainData, batch_size=batch_size,shuffle=True)
    t = "acc"

    avg_loss, train_acc, val_acc = [], [], []

    # training
    for epoch in range(num_epochs):
        losses = []
        for x, y in train_loader:
            x = dimFix(x)
            y = dimFix(y)
            out = model(x)             # forward pass
            
            if type(criterion) is nn.MSELoss:
                out = out.squeeze(1)
                t="loss"

            loss = criterion(out, y) # compute the total loss
            loss.backward()               # backward pass (compute parameter updates)
            optimizer.step()              # make the updates for each parameter
            optimizer.zero_grad()         # a clean up step for PyTorch
            losses.append(float(loss)/y.shape[0])

        accT = accFunc(model,trainData, criterion)
        accV = accFunc(model,valData, criterion)
        lossAvg = sum(losses)/len(losses)
        train_acc.append(accT) # compute training accuracy 
        val_acc.append(accV)  # compute validation accuracy
        avg_loss.append(lossAvg)
        print(f"E {epoch}, Avg loss: {lossAvg}, train {t}: {accT:.5f}, val {t}: {accV:.5f}")
        torch.save(model.state_dict(), mdPath+f"{epoch}.model")

    with open(mdPath+"result.json", 'w') as f:
        json.dump({'loss':avg_loss,'train':train_acc,'val':val_acc},f)
    print(f"Run {runId} finished")