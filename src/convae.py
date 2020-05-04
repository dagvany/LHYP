#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

import os
import sys
import pickle
from pathlib import Path
import time
import numpy as np

from utils import get_logger

logger = get_logger(__name__)

class ConvAE(nn.Module):
    def __init__(self):
        super(ConvAE, self).__init__()

        self.encoderConv = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 8, kernel_size=2, stride=1, padding=0),
            nn.SELU(),

            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=2, stride=1, padding=0),
            nn.SELU(),

            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=0),
            nn.SELU(),

            nn.BatchNorm2d(32),
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=0),
            nn.SELU(),

            nn.BatchNorm2d(16),
            nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=0),
            nn.SELU(),

            # Instead of MaxPool
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0),
            nn.SELU(),
        )

        self.decoderDeConv = nn.Sequential(
            nn.ConvTranspose2d(1, 8, kernel_size=1, stride=1, padding=0),
            nn.SELU(),

            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=0),
            nn.SELU(),
            nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=0),
            nn.SELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=1, padding=0),
            nn.SELU(),
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=1, padding=0),
            nn.SELU(),
            nn.ConvTranspose2d(8, 1, kernel_size=2, stride=1, padding=0),
            nn.SELU(),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    #nn.init.uniform(m.bias)
                    nn.init.xavier_uniform(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    #nn.init.uniform(m.bias)
                    nn.init.xavier_uniform(m.weight)


    def forward(self, bachedInputs):
        latent = self.encoderConv(bachedInputs)
        decoded = self.decoderDeConv(latent)
        return decoded, latent


def run(patientsImages, config):
    if config['cuda_seed'] >= 0:
        device = 'cuda:{}'.format(config['cuda_seed'])
    else:
        device = 'cpu'

    timestr = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(os.path.join(config["tensoardboardx_folder"], timestr))
    model = ConvAE().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    images = torch.Tensor(patientsImages)
    height = patientsImages[0].shape[0]
    width = patientsImages[0].shape[1]
    trainSetSize = int(np.round((len(patientsImages)*0.75)/config['batch_size'])) * config['batch_size']
    trainSet = images[0: trainSetSize]
    testSet = images[trainSetSize+1: ]

    msg = 'shape: {}, trainSet: {}, testSet: {}'.format(patientsImages[0].shape, len(trainSet), len(testSet))
    print(msg)
    logger.info(msg)

    original = trainSet[0:config["batch_size"]].view(-1, 1, height, width)
    img = vutils.make_grid(original, normalize=True)
    writer.add_image('train', img, 0)

    l = sys.maxsize
    epoch = 0
    saveImage = False
    tempImage = None
    while l > config['goal_loss']:
        if 1 < config['epoch'] == epoch:
            break
        epoch += 1

        if epoch % 25 == 0:
            saveImage = True

        for i in range(0, len(trainSet), config["batch_size"]):
            original = trainSet[i:i + config["batch_size"]].view(-1, 1, height, width)
            data = original.to(device)
            # forward
            output, latent = model(data)
            loss = criterion(output, data)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            l = loss.data # while

            writer.add_scalar('Loss/train', l, epoch)

            if saveImage:
                img = vutils.make_grid(output.cpu().data, normalize=True)
                writer.add_image('train', img, epoch)
                logger.info('epoch [{}/{}], loss: {:.4f}'.format(epoch, config["epoch"], loss.data))
                saveImage = False
            elif i == 0:
                tempImage = output.cpu().data

    img = vutils.make_grid(tempImage, normalize=True)
    writer.add_image('train', img, epoch)

    writer.add_hparams(
        {
            'optim': 'adam',
            'bsize': config['batch_size'],
            'lr': config['learning_rate'],
            'wd': config['weight_decay'],
            'epoch': epoch
        },
        {
            'loss': l,
        }
    )

    latentVectors = []
    for i in range(len(testSet)):
        original = testSet[i].view(-1, 1, height, width)
        data = original.to(device)
         # forward
        output, latent = model(data)
        loss = criterion(output, data)
        latentVectors.append(latent[0])

        image = torch.cat([original, output.cpu().data], dim=0)
        img = vutils.make_grid(image, normalize=True)
        writer.add_image('test', img, loss.data)
        writer.add_scalar('Loss/test', loss.data, i)
        logger.info('test [{}/{}], loss:{:.4f}'.format(i + 1, trainSetSize, loss.data))

    print('latent size: {}'.format(len(latentVectors)))
    file_name = '{}_{}.pickle'.format('la', timestr)

    if not os.path.exists(config['latent_pickle']):
        os.makedirs(config['latent_pickle'])
    path = os.path.join(config['latent_pickle'], file_name)

    try:
        with open(path, 'wb') as fp:
            pickle.dump(latentVectors, fp)
    except Exception:
        msg = 'Latent: {} serialization (dump) is failed!'.format('la')
        logger.critical(msg, exc_info=True)
        print(msg)
    logger.info('Dumped: {}'.format(file_name))

    writer.close()

    Path(config["pytorch_model_folder"]).mkdir(parents=True, exist_ok=True)
    modelPath = os.path.join(config["pytorch_model_folder"], 'cae_la_{}.pt').format(timestr)
    torch.save(model.state_dict(), modelPath)