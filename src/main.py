#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from utils import get_logger, progress_bar
from preprocessor import Preprocessor

from models import Patient

# Time measure
from timeit import default_timer as timer
from datetime import timedelta

# Unserialize Stat
import json
from models import patient_enums, ImageCollection, Image

logger = get_logger(__name__)

def preprocessPatiens(config):
    print('Preprocessing Patients Images')
    start = timer()
    try:
        patientStats = Preprocessor.preprocessPatientsByConfig(config)
    except Exception as ex:
        logger.critical(ex, exc_info=True)
        print(ex)
        exit(1)
    end = timer()
    print('Measured time: {}'.format(str(timedelta(seconds=end-start))))
    
    return patientStats

def unSerializePatients(srcFolderPath):
    print('Reload from Pickle')
    start = timer()
    reloadedPatients = []
    numOfEmpty = 0
    numOfError = 0
    pathologyNames = filter(lambda p: p[0] != '_', dir(patient_enums.Pathology))
    typeDict = dict.fromkeys(list(pathologyNames), 0)
    
    fileList = os.listdir(config["pickle_folder"])
    index = 0
    for index, fileName in enumerate(fileList):
        progress_bar(index+1, len(fileList), 20)
        fullPath = os.path.join(config["pickle_folder"], fileName)
        logger.debug('Pickle load: {}'.format(fullPath))
        
        try:
            patient = Patient.unSerialize(fullPath)
            if patient.hasAnyImage():
                pathName = patient.Pathology.name
                typeDict[pathName] = typeDict[pathName] + 1
                reloadedPatients.append(patient)
            else:
                logger.warning('Empty: {}'.format(fullPath))
                destPath = os.path.join(config['failed_pickle_folder'], fileName)
                os.replace(fullPath, destPath)
                numOfEmpty = numOfEmpty + 1
        except Exception:
            logger.error('Broken: {}'.format(fullPath), exc_info=True)
            destPath = os.path.join(config['failed_pickle_folder'], fileName)
            os.replace(fullPath, destPath)
            numOfError = numOfError + 1

    end = timer()
    
    print('Measured time: {}'.format(str(timedelta(seconds=end-start))))
    print('Total: {:5d}, Failed: {:5d}, Empty: {:5d}'.format(index+1, numOfError, numOfEmpty))
    try:
        import psutil
        patientSize = psutil.Process(os.getpid()).memory_info().rss /1024 /1024
        print("Memory size of patients = {} Mbytes".format(patientSize))
    except:
        pass
    print(json.dumps(typeDict, indent=4, sort_keys=True))

    return reloadedPatients

if __name__ == '__main__':
    config = Preprocessor.getConfiguration("config.json")

    # Preprocessing
    if config["patients_preprocessing"]:
        patientStats = preprocessPatiens(config)
        tldrStat = 'Total: {:5d}, Correct: {:5d}, Broken: {:5d}'.format(
            patientStats[1]+patientStats[2], patientStats[1], patientStats[2])
        print(tldrStat)
        logger.info(tldrStat)
        logger.info('\n'.join(patientStats[0]))
    else:
        patients = unSerializePatients(config['pickle_folder'])
