#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from utils import get_logger
from models.patient_enums import Pathology
from proc_utils import getConfiguration, unSerializePatients

import os
import cv2
import random

logger = get_logger(__name__)

config = getConfiguration("config_img_cropping.json")
cropSize = config['crop_size']
picklePath = os.path.join(config['root'], config['pickle_folder'])
failedPath = os.path.join(config['root'], config['failed_pickle_folder'])

patients = unSerializePatients(picklePath, failedPath)

for setType in patients.keys():
    for p in patients[setType]:
        for imgType in config['image_types']:
            if imgType in p.ImageTypes:
                for i in range(len(p.ImageTypes[imgType])):
                    img = p.ImageTypes[imgType][i]
                    h, w = img.shape
                    y = int((h - cropSize) / 2)
                    x = int((w - cropSize) / 2)
                    img = img[y:y + cropSize, x:x + cropSize]
                    p.ImageTypes[imgType][i] = img
        p.serialize(os.path.join(picklePath, setType))

