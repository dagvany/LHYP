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
    path = os.path.join(config['pytorch_model_folder'], config['pytorch_model_file'])

    if config['cuda_seed'] >= 0:
        device = 'cuda:{}'.format(config['cuda_seed'])
    else:
        device = 'cpu'

    model = LaConvAE()
    model.load_state_dict(torch.load(path, map_location=torch.device(device)))

    for dirName in os.listdir(config['pickle_folder']):
        dirPath = os.path.join(config['pickle_folder'], dirName)
        logger.info(dirName)
        if os.path.isdir(dirPath):
            patients = unSerializePatients(dirPath, config['failed_pickle_folder'])
            imgType = config['image_type']

            latentData = []
            for p in patients:
                for img in p.ImageTypes[imgType].Views:
                    # TODO put this part to preprocessor
                    # -------------------------------------------------
                    cropSize = 130
                    h, w = img.PixelArray.shape
                    y = int((h - cropSize) / 2)
                    x = int((w - cropSize) / 2)
                    crop_img = img.PixelArray[y:y + cropSize, x:x + cropSize]
                    # -------------------------------------------------
                    original = torch.Tensor(crop_img)
                    original = original.view(-1, 1, cropSize, cropSize)
                    modelData = original.to(device)
                    # forward
                    output, latent = model(modelData)
                    data = [p.Pathology, latent]
                    latentData.append(data)

            timestr = config['pytorch_model_file'].split('_')[-1].split('.')[0]
            fileName = '{}_{}_{}.pickle'.format(imgType, dirName, timestr)
            path = os.path.join(config['latent_pickle'], fileName)
            try:
                with open(path, 'wb') as fp:
                    pickle.dump(latentData, fp)
            except Exception:
                msg = 'Latent: {} serialization (dump) is failed!'.format('la')
                logger.critical(msg, exc_info=True)
                print(msg)
            logger.info('Dumped: {}'.format(fileName))

        else:
            logger.error("Unexpected file: {}".format(dirPath))