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
    patients = unSerializePatients(config)

    # TODO put this part to preprocessor
    cropSize = 130
    patientsImages = []
    for p in patients:
        for imgType in config['image_types']:
            for img in p.ImageTypes[imgType].Views:
                h, w = img.PixelArray.shape
                y = int((h - cropSize) / 2)
                x = int((w - cropSize) / 2)
                crop_img = img.PixelArray[y:y + cropSize, x:x + cropSize]
                patientsImages.append(crop_img)

    random.shuffle(patientsImages)

    if config['cuda_seed'] >= 0:
        device = 'cuda:{}'.format(config['cuda_seed'])
    else:
        device = 'cpu'

    timestr = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(os.path.join(config["tensoardboardx_folder"], timestr))
    model = LaConvAE().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    images = torch.Tensor(patientsImages)
    height = patientsImages[0].shape[0]
    width = patientsImages[0].shape[1]
    trainSetSize = int(np.round((len(patientsImages) * 0.75) / config['batch_size'])) * config['batch_size']
    trainSet = images[0: trainSetSize]
    testSet = images[trainSetSize + 1:]

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
