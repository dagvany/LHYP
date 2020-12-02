#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from utils import get_logger
from models.patient_enums import Pathology
from proc_utils import getConfiguration, unSerializePatients

import os
import cv2
import random

logger = get_logger(__name__)

config = getConfiguration("config_augmentation.json")
angle = config['angle']
picklePath = os.path.join(config['root'], config['pickle_folder'])
augmentedPath = os.path.join(picklePath, 'train')
failedPath = os.path.join(config['root'], config['failed_pickle_folder'])

patients = unSerializePatients(picklePath, failedPath)

angles = [i for i in range(-1*angle, angle)]
del angles[angle] # delete zero

pathologys = [Pathology[i] for i in config['pathologys']]
for p in patients['train']:
    if p.Pathology in pathologys:
        p.ID = p.ID + '_augmented'
        for imgType in config['image_types']:
            if imgType in p.ImageTypes:
                for i in range(len(p.ImageTypes[imgType])):
                    img = p.ImageTypes[imgType][i]
                    (h1, w1) = img.shape[:2]
                    center = (w1 / 2, h1 / 2)
                    matrix = cv2.getRotationMatrix2D(
                        center, random.choice(angles), 1.0)
                    img = cv2.warpAffine(img, matrix, (w1, h1))
                    p.ImageTypes[imgType][i] = img
        p.serialize(augmentedPath)
