#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Serialization
import os
import pickle

from utils import get_logger

from .patient_enums import Pathology
from .imageCollection import ImageCollection

logger = get_logger(__name__)

class Patient:
    def __init__(self, patientId):
        self.ID = patientId
        self.Pathology = Pathology.UNDEFINED
        self.ImageTypes = {}

    def hasAnyImage(self):
        return any([len(self.ImageTypes[t].Views) > 0 for t in self.ImageTypes])
    
    def organizeImageAttributes(self):
        for iType in self.ImageTypes:
            self.ImageTypes[iType].organiseAttributes()

    def serialize(self, dest_folder):
        file_name = self.ID + '.pickle'
        
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
