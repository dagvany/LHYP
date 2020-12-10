#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import time
import random
from itertools import chain

import numpy as np
from pathlib import Path

from laconvaesmall import LaConvAEsmall as LaConvAE

from utils import get_logger, progress_bar
from proc_utils import getConfiguration, unSerializePatients

import torch
from torch import nn
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

logger = get_logger(__name__)

if __name__ == '__main__':
    config = getConfiguration("config_laconvae.json")
    picklePath = os.path.join(config['root'], config['pickle_folder'])
    failedPath = os.path.join(config['root'], config['failed_pickle_folder'])

    patients = unSerializePatients(picklePath, failedPath)

    random.shuffle(patients['train'])
    trainSet = []
    for p in patients['train']:
        if config['image_type'] in p.ImageTypes:
            trainSet.append(p.ImageTypes[config['image_type']])
    trainSet = torch.Tensor(list(chain(*trainSet)))

    validateSet = []
    for p in patients['validate']:
        if config['image_type'] in p.ImageTypes:
            validateSet.append(p.ImageTypes[config['image_type']])
    validateSet = torch.Tensor(list(chain(*validateSet)))

    if config['cuda_seed'] >= 0:
        device = 'cuda:{}'.format(config['cuda_seed'])
    else:
        device = 'cpu'

    timestr = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(
        os.path.join(config["root"], config["tensoardboardx_folder"], timestr))
    model = LaConvAE().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'])

    height, width = trainSet[0].shape

    msg = 'shape: {}, trainSet: {}, validateSet: {}'.format(
        (height, width), len(trainSet), len(validateSet))
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

        if epoch % 10 == 0:
            saveImage = True

        for i in range(0, len(trainSet), config["batch_size"]):
            original = trainSet[i:i + config["batch_size"]]
            original = original.view(-1, 1, height, width)
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
                logger.info('epoch [{}/{}], loss: {:.4f}'.format(
                    epoch, config["epoch"], loss.data))
                saveImage = False
            elif i == 0:
                tempImage = output.cpu().data
                
        original = random.choice(validateSet)
        original = original.view(-1, 1, height, width)
        data = original.to(device)
        output, latent = model(data)
        loss = criterion(output, data)
        writer.add_scalar('Loss/test', loss.data, epoch)
                

    img = vutils.make_grid(tempImage, normalize=True)
    writer.add_image('train', img, epoch)

    writer.add_hparams(
        {
            'optim': 'adam',
            'bsize': config['batch_size'],
            'activation': model.getNameOfActivation(),
            'lr': config['learning_rate'],
            'wd': config['weight_decay'],
            'epoch': epoch
        },
        {
            'loss': l,
        }
    )

    for i in range(len(validateSet)):
        original = validateSet[i].view(-1, 1, height, width)
        data = original.to(device)
        # forward
        output, latent = model(data)
        loss = criterion(output, data)

        image = torch.cat([original, output.cpu().data], dim=0)
        img = vutils.make_grid(image, normalize=True)
        writer.add_image('test', img, i)
        writer.add_scalar('Loss_final/test', loss.data, i)
        logger.info('test [{}/{}], loss:{:.4f}'.format(
            i+1, len(validateSet), loss.data))

    writer.close()

    modelFolder = os.path.join(config["root"], config["pytorch_model_folder"])
    Path(modelFolder).mkdir(parents=True, exist_ok=True)
    modelPath = os.path.join(modelFolder, 'cae_la_{}.pt').format(timestr)
    torch.save(model.state_dict(), modelPath)
