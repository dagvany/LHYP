#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle

from utils import get_logger
from .patient_enums import Pathology

logger = get_logger(__name__)


class Patient:
    def __init__(self, patientId):
        self.ID = patientId
        self.Pathology = Pathology.UNDEFINED
        self.Sex = None
        self.Weight = None
        self.Height = None
        self.ImageTypes = {}

    def hasAnyImage(self):
        return any([len(self.ImageTypes[t]) > 0 for t in self.ImageTypes])

    def serialize(self, dest_folder):
        file_name = self.ID + '.pickle'
        dest_folder = os.path.join(dest_folder, self.Pathology.name)
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
        path = os.path.join(dest_folder, file_name)

        try:
            with open(path, 'wb') as fp:
                pickle.dump(self, fp)
        except Exception:
            msg = 'Patient: {} serialization (dump) is failed!'.format(self.ID)
            logger.critical(msg, exc_info=True)
            print(msg)
            exit(1)
        logger.info('Dumped: {}'.format(file_name))

    @staticmethod
    def unSerialize(src_path):
        try:
            with open(src_path, 'rb') as fp:
                patient = pickle.load(fp)
            logger.info('Loaded: {}'.format(src_path))
        except Exception:
            msg = '{} serialization (load) is failed!'.format(src_path)
            logger.critical(msg, exc_info=True)
            print(msg)
        return patient
