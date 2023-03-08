#%%
import pandas as pd
import glob
import pandas as pd
import torch
import torch.nn as nn
import cv2
import glob
import shutil
import os
import numpy as np
from PIL import Image
import random
import time
random.seed(int(time.time()))

import torchvision.models

aw = torchvision.models.AlexNet_Weights.DEFAULT
alexnet = torchvision.models.alexnet(weights=aw)
alexnet.eval()

rw = torchvision.models.ResNet50_Weights.DEFAULT
res50 = torchvision.models.resnet50(weights=rw)
res50.fc = nn.Identity()
res50.eval()

iw = torchvision.models.Inception_V3_Weights.DEFAULT
incep = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', weights=iw)
incep.fc = nn.Identity()
incep.eval()

#%%
BREEDS = ['Domestic Short Hair', 'Domestic Long Hair', 'Calico', 'Ragdoll', 'Siamese', 'Russian Blue', 'Bengal', 'Persian']
AGES = ['Baby','Young','Adult','Senior']
BREEDS_REF = {b:BREEDS.index(b) for b in BREEDS}
AGES_REF = {a:AGES.index(a) for a in AGES}
BATCH = 32
QUOTA = 3000
COUNT_REF = {b:len(glob.glob(f"./out/{b}/*.png")) for b in BREEDS}

#%%
shutil.rmtree("./featureOut",ignore_errors=True)
os.makedirs("./featureOut")

cats = pd.read_csv("./data/cats.csv")
cats = cats.loc[cats.breed.isin(BREEDS)][['id','breed','age']]
cats.set_index(["id"],inplace=True)

start = time.time()
current=0
estTotal=sum([(x if x>QUOTA else QUOTA) for x in COUNT_REF.values()])
#%%
def saveBatch(imgs,labels,outdir,name):
    global current,estTotal
    current += BATCH
    s = len(imgs)
    imgsNP = np.moveaxis(np.stack(imgs),-1,1)
    with torch.no_grad():
        imgTS = torch.tensor(imgsNP).float()
        # featuresA = alexnet.features(torch.tensor(imgsNP).float())
        # featuresA =  torch.from_numpy(featuresA.detach().numpy())
        # featuresA = featuresA.reshape(s,-1)

        featuresR = res50(imgTS)
        featuresR = featuresR.detach().numpy()
        featuresR = featuresR.reshape(s,-1)

        featuresI = incep(imgTS)
        featuresI = featuresI.detach().numpy()
        featuresI = featuresI.reshape(s,-1)

        x = np.concatenate((featuresR, featuresI),1)

    y = np.stack(labels)
    pack = np.concatenate((x,y),axis=1)
    np.save(f"{outdir}/{name}",pack)
    print(f"Progress {current}/{estTotal} ({current*100/estTotal:.2f}%), time elapsed {time.time()-start:.2f}s")
#%%
files = glob.glob("./out/*/*.png")
random.shuffle(files)

#%%
imgs = []
labels = []
batchId = 0
for f in files:
    catId = int(f.split("/")[-1].split("_")[0])
    catBreed = cats.loc[catId].breed
    catAge = cats.loc[catId].age
    img = np.array(Image.open(f))/255
    img = (img-[0.485, 0.456, 0.406])/[0.229, 0.224, 0.225]
    label = np.array([BREEDS_REF[catBreed],AGES_REF[catAge]])

    imgs.append(img)
    labels.append(label)

    if COUNT_REF[catBreed]<QUOTA and len(imgs) < BATCH:
        imgs.append(np.fliplr(img))
        labels.append(label)
        COUNT_REF[catBreed]+=1

    if len(imgs) == BATCH:
        saveBatch(imgs,labels,"./featureOut",str(batchId))
        batchId+=1
        imgs = []
        labels = []

### DISCARD THE LAST BATCH!! WILL CAUSE ISSUE WITH TORCH IF BATCH SIZE DOES NOT EQUAL!
# if len(imgs):
#     saveBatch(imgs,labels,"./featureOut",str(batchId))
