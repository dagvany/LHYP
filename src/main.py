#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from utils import get_logger, progress_bar
from preprocessor import Preprocessor

from models import Patient

# Time measure
from timeit import default_timer as timer
from datetime import timedelta

logger = get_logger(__name__)

def preprocessPatiens(config):
    print('Preprocessing Patients Images')
    start = timer()
    try:
        patientData = Preprocessor.getPatients(config)
    except Exception as ex:
        logger.critical(ex, exc_info=True)
        exit(1)
    end = timer()
    print('Measured time: {}'.format(str(timedelta(seconds=end-start))))
    
    return patientData

def printTable(patients):
    for p in patients:
        img_meta = []
        for t in p.ImageTypes:
            img_meta.append('{}: {:4d} commonAttr: {:4d}'.format(
                t,
                len(p.ImageTypes[t].Views),
                len(p.ImageTypes[t].CommonAttibutes)))
        
        print('\t'.join([
            p.ID, 
            str(p.Pathology),
            str(img_meta)]))
    print('\n\n')

def serializePatiens(destPath, patientData):
    for p in patientData:
        p.serialize(destPath)

def unSerializePatients(srcFolderPath):
    print('Reload from Pickle')
    start = timer()
    reloadedPatients = []
    fileList = os.listdir(config["pickle_folder"])
    for index, fileName in enumerate(fileList):
        progress_bar(index+1, len(fileList), 20)
        fullPath = os.path.join(config["pickle_folder"], fileName)
        logger.debug('Pickle load: {}'.format(fullPath))
        
        patient = Patient.unSerialize(fullPath)
        reloadedPatients.append(patient)

    end = timer()
    print('Measured time: {}'.format(str(timedelta(seconds=end-start))))

    return reloadedPatients

# Preprocessing
config = Preprocessor.getConfiguration("config.json")
processedPationts = preprocessPatiens(config)
printTable(processedPationts)

serializePatiens(config['pickle_folder'], processedPationts)

loadedPatients = unSerializePatients(config['pickle_folder'])
printTable(loadedPatients)
