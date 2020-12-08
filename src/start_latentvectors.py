#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import pickle
import torch

from laconvae import LaConvAE

from utils import get_logger, progress_bar
from proc_utils import getConfiguration, unSerializePatients

logger = get_logger(__name__)

if __name__ == '__main__':
    config = getConfiguration("config_latent.json")
    trainedModelPath = os.path.join(
        config["root"],
        config['pytorch_model_folder'],
        config['pytorch_model_file'])
    picklePath = os.path.join(config['root'], config['pickle_folder'])
    failedPath = os.path.join(config['root'], config['failed_pickle_folder'])
    latentPath = os.path.join(config['root'], config['latent_pickle'])
    imgType = config['image_type']
    timestr = config['pytorch_model_file'].split('_')[-1].split('.')[0]

    if config['cuda_seed'] >= 0:
        device = 'cuda:{}'.format(config['cuda_seed'])
    else:
        device = 'cpu'

    model = LaConvAE()
    model.load_state_dict(torch.load(
        trainedModelPath, map_location=torch.device(device)))
    model.eval()
    
    patients = unSerializePatients(picklePath, failedPath)
    height, width = patients['train'][0].ImageTypes[imgType][0].shape
    for setType in patients:
        latentData = []
        for p in patients[setType]:
            if imgType not in p.ImageTypes:
                continue
            for img in p.ImageTypes[imgType]:
                original = torch.Tensor(img)
                original = original.view(-1, 1, height, width)
                modelData = original.to(device)
                output, latent = model(modelData)
                labeledLatent = (p.Pathology, latent)
                latentData.append(labeledLatent)
   
        fileName = '{}_{}_{}.pickle'.format(imgType, setType, timestr)
        latentFilePath = os.path.join(latentPath, fileName)
        try:
            with open(latentFilePath, 'wb') as fp:
                pickle.dump(latentData, fp)
        except Exception:
            msg = 'Latent: {} serialization dump is failed!'.format(imgType)
            logger.critical(msg, exc_info=True)
            print(msg)
        logger.info('Dumped: {}'.format(fileName))
