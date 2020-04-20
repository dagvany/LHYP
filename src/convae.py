#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torchvision.utils import save_image

import os
import sys
from pathlib import Path
import time
import numpy as np

from utils import get_logger

logger = get_logger(__name__)


class ConvAE(nn.Module):
    def __init__(self):
        super(ConvAE, self).__init__()

        self.encoderConv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=4, stride=1, padding=0),
            nn.ELU(),
            nn.Conv2d(1, 1, kernel_size=4, stride=2, padding=0),
            nn.ELU(),
            nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=0),
            nn.ELU()
        )

        self.encoderLin = nn.Sequential(
            nn.Linear(30*30, 100),
            nn.Softplus()
        )

        self.decoderLin = nn.Sequential(
            nn.Linear(30*30, 130*130),
            nn.Softplus()
        )

        self.decoderDeConv = nn.Sequential(
            nn.ConvTranspose2d(1, 1, kernel_size=5, stride=2, padding=0),
            nn.ELU(),
            nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=0),
            nn.ELU(),
            nn.ConvTranspose2d(1, 1, kernel_size=3, stride=1, padding=0),
            nn.ELU()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.uniform(m.bias)
                #nn.init.xavier_uniform(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    nn.init.uniform(m.bias)
                #nn.init.xavier_uniform(m.weight)


    def forward(self, bachedInputs):
        batchSize = bachedInputs.shape[0]
        encodedConv = self.encoderConv(bachedInputs)

        #decodedLin = self.decoderLin(encodedConv.view((batchSize, -1)))
        #decoded = decodedLin.view(batchSize, 1, 130, 130)
        decoded = self.decoderDeConv(encodedConv)
        #print("Encoded: {} Decoded: {}".format(encodedConv.shape, decoded.shape))
        return decoded


def run(patientsImages, config):
    if config['cuda_seed'] >= 0:
        device = 'cuda:{}'.format(config['cuda_seed'])
    else:
        device = 'cpu'

    model = ConvAE().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    images = torch.Tensor(patientsImages)
    height = patientsImages[0].shape[0]
    width = patientsImages[0].shape[1]
    trainSetSize = int(np.ceil((len(patientsImages)*0.75)/config['batch_size'])) * config['batch_size']
    trainSet = images[0: trainSetSize]
    testSet = images[trainSetSize+1: ]

    msg = 'shape: {}, trainSet: {}, testSet: {}'.format(patientsImages[0].shape, len(trainSet), len(testSet))
    print(msg)
    logger.info(msg)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    config["img_folder"] = os.path.join(config["img_folder"], timestr)
    os.mkdir(config["img_folder"])
    path = os.path.join(config["img_folder"], '{:04d}_{}_{}.png')
    pathFinal = os.path.join(config["img_folder"], '{:04d}_final_{}.png')

    original = trainSet[0:config["batch_size"]].view(-1, 1, height, width)
    save_image(original, os.path.join(config["img_folder"], 'original.png'), normalize=True)

    l = sys.maxsize
    epoch = 0
    saveImage = False
    tempImage = None
    while l > config['goal_loss']:
        if 1 < config['epoch'] == epoch:
            break
        epoch += 1

        if epoch % 100 == 0:
            saveImage = True

        for i in range(0, len(trainSet), config["batch_size"]):
            original = trainSet[i:i + config["batch_size"]].view(-1, 1, height, width)
            data = original.to(device)
            # forward
            output = model(data)
            loss = criterion(output, data)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            l = loss.data # while
            if saveImage:
                save_image(output.cpu().data, path.format(epoch, 'train', l), normalize=True)
                logger.info('epoch [{}/{}], loss:{:.4f}'.format(epoch, config["epoch"], loss.data))
                saveImage = False
            elif i == 0:
                tempImage = output.cpu().data


    save_image(tempImage, pathFinal.format(epoch, l), normalize=True)

    for i in range(len(testSet)):
        original = testSet[i].view(-1, 1, height, width)
        data = original.to(device)
         # forward
        output = model(data)
        loss = criterion(output, data)

        image = torch.cat([original, output.cpu().data], dim=0)
        save_image(image, path.format(epoch, 'test', loss.data), normalize=True)
        logger.info('test [{}/{}], loss:{:.4f}'.format(i + 1, testSetSize, loss.data))

    Path(config["pytorch_model_folder"]).mkdir(parents=True, exist_ok=True)
    modelPath = os.path.join(config["pytorch_model_folder"], 'cae_la_{}.pt').format(timestr)
    torch.save(model.state_dict(), modelPath)

