#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import main
import random
import sys
import cv2
import os
import time
import torch
from torch import nn
from torchvision.utils import save_image
from pathlib import Path

class ConvAE(nn.Module):
    def __init__(self):
        super(ConvAE, self).__init__()

        self.encoderConv = nn.Sequential(
            nn.Conv2d(1, 1, 4, stride=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(1, 1, 4, stride=2, padding=0),
            nn.ReLU(True),
            nn.Conv2d(1, 1, 3, stride=2, padding=0),
            nn.Tanh(),
        )

        self.encoderLin = nn.Sequential(
            nn.Linear(30*30, 100),
            nn.Softplus()
        )

        self.decoderLin = nn.Sequential(
            #nn.Linear(100, 130*130),
            nn.Linear(30*30, 130*130),
            nn.Softplus()
        )

    def forward(self, bachedInputs):
        batchSize = bachedInputs.shape[0]
        encodedConv = self.encoderConv(bachedInputs)
        #encodedLin = self.encoderLin(encodedConv.view(batchSize, -1))
        #print(encodedLin.shape)

        #decodedLin = self.decoderLin(encodedLin)
        decodedLin = self.decoderLin(encodedConv.view((batchSize, -1)))
        decoded = decodedLin.view(batchSize, 1, 130, 130)
        #print(decoded.shape)
        return decoded


config = {
    "image_folder": "/media/Data/Dipterv_MRI/test/mri",
    "image_types": {
        "la": {
            "folder": "la",
            "separator_attr": "ImageOrientationPatient",
            "goal_amount": 3
        }
    },
    "pickle_folder": "/media/Data/Dipterv_MRI/test/pickle",
    "failed_pickle_folder": "/media/Data/Dipterv_MRI/test/failed",
    "pytorch_model_folder": "/media/Data/Dipterv_MRI/test/pytorch_model",
    "img_folder": "/media/Data/Dipterv_MRI/test/ae",
    "cuda_seed": -1,
    "batch_size": 6,
    "epoch": 500,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5
}

patients = main.unSerializePatients(config)

cropSize = 130
patientsImages = []
for p in patients:
    for imgType in config['image_types']:
        for img in p.ImageTypes[imgType].Views:
            h, w = img.PixelArray.shape
            y = int((h-cropSize)/2)
            x = int((w-cropSize)/2)
            crop_img = img.PixelArray[y:y+cropSize, x:x+cropSize]
            patientsImages.append(crop_img)

random.shuffle(patientsImages)
print('shape: {}, images: {}'.format(patientsImages[0].shape, len(patientsImages)))

# -----------------------------------------------------------------------------------------------------------------------

if config['cuda_seed'] >= 0:
    device = 'cuda:{}'.format(config['cuda_seed'])
else:
    device = 'cpu'

model = ConvAE().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

images = torch.Tensor(patientsImages)
height = width = cropSize
timestr = time.strftime("%Y%m%d-%H%M%S")
config["img_folder"] = os.path.join(config["img_folder"], timestr)
os.mkdir(config["img_folder"])
path = os.path.join(config["img_folder"], '{:04d}_{}_{}.png')
pathFinal = os.path.join(config["img_folder"], '{:04d}_final_{}.png')

original = images[0:config["batch_size"]].view(-1, 1, height, width)
save_image(original, os.path.join(config["img_folder"], 'original.png'), normalize=True)

l = sys.maxsize
epoch = 0
saveImage = False
tempImage = None
while l > 0.1:
    epoch = epoch + 1
    if epoch % 25 == 0:
        saveImage = True

    for i in range(0, len(images), config["batch_size"]):
        original = images[i:i + config["batch_size"]].view(-1, 1, height, width)
        data = original.to(device)
        # forward
        output = model(data)
        loss = criterion(output, data)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if saveImage:
            save_image(output.cpu().data, path.format(epoch, 'res', loss.data), normalize=True)
            # logger.info('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, config["epoch"], loss.data))
            saveImage = False
            tempImage = None
        elif i == 0:
            tempImage = output.cpu().data

    l = loss.data   #while
save_image(tempImage, pathFinal.format(epoch, l), normalize=True)

Path(config["pytorch_model_folder"]).mkdir(parents=True, exist_ok=True)
modelPath = os.path.join(config["pytorch_model_folder"], 'cae_la_{}.pt').format(timestr)
torch.save(model.state_dict(), modelPath)
