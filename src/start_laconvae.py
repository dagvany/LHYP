#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import time
import random
import pickle
import numpy as np
from pathlib import Path

from laconvae import LaConvAE

from utils import get_logger, progress_bar
from proc_utils import getConfiguration, unSerializePatients

import torch
from torch import nn
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

logger = get_logger(__name__)

if __name__ == '__main__':
    config = getConfiguration("config_laconvae.json")
    patients = unSerializePatients(config['pickle_folder'], config['failed_pickle_folder'])

    trainSetSize = int(np.round((len(patients) * config['train_rate'])))
    trainPatients = patients[0: trainSetSize]
    trainSet = []
    testSet = []
    for i, p in enumerate(patients):
        for imgType in config['image_types']:
            for img in p.ImageTypes[imgType].Views:
                # TODO put this part to preprocessor
                # -------------------------------------------------
                cropSize = 130
                h, w = img.PixelArray.shape
                y = int((h - cropSize) / 2)
                x = int((w - cropSize) / 2)
                crop_img = img.PixelArray[y:y + cropSize, x:x + cropSize]
                # -------------------------------------------------
                if i < trainSetSize:
                    trainSet.append(crop_img)
                else:
                    testSet.append(crop_img)
    trainSet = torch.Tensor(random(trainSet))
    testSet = torch.Tensor(testSet)

    if config['cuda_seed'] >= 0:
        device = 'cuda:{}'.format(config['cuda_seed'])
    else:
        device = 'cpu'

    timestr = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(os.path.join(config["tensoardboardx_folder"], timestr))
    model = LaConvAE().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    height = trainSet[0].shape[0]
    width = trainSet[0].shape[1]

    msg = 'shape: {}, trainSet: {}, testSet: {}'.format(trainSet[0].shape, len(trainSet), len(testSet))
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

            l = loss.data  # while

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

    for i in range(len(testSet)):
        original = testSet[i].view(-1, 1, height, width)
        data = original.to(device)
        # forward
        output, latent = model(data)
        loss = criterion(output, data)

        image = torch.cat([original, output.cpu().data], dim=0)
        img = vutils.make_grid(image, normalize=True)
        writer.add_image('test', img, loss.data)
        writer.add_scalar('Loss/test', loss.data, i)
        logger.info('test [{}/{}], loss:{:.4f}'.format(i + 1, trainSetSize, loss.data))

    writer.close()

    Path(config["pytorch_model_folder"]).mkdir(parents=True, exist_ok=True)
    modelPath = os.path.join(config["pytorch_model_folder"], 'cae_la_{}.pt').format(timestr)
    torch.save(model.state_dict(), modelPath)
