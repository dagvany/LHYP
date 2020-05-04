#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from timeit import default_timer as timer
from datetime import timedelta

from models import Patient
from models import patient_enums, ImageCollection, Image
from utils import get_logger, progress_bar

logger = get_logger(__name__)

def getConfiguration(jsonFileLocation="config.json"):
    try:
        with open(jsonFileLocation, 'r') as fp:
            config = json.load(fp)
    except FileNotFoundError:
        msg = jsonFileLocation + ' is not found!'
        logger.critical(msg)
        print(msg)
        exit(1)
    except ValueError:
        msg = jsonFileLocation + ' JSON format is not correct!'
        logger.critical(msg)
        print(msg)
        exit(1)

    return config

def unSerializePatients(config):
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
        progress_bar(index + 1, len(fileList), 20)
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

    print('Measured time: {}'.format(str(timedelta(seconds=end - start))))
    print('Total: {:5d}, Failed: {:5d}, Empty: {:5d}'.format(index + 1, numOfError, numOfEmpty))
    try:
        import psutil
        patientSize = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        print("Memory size of patients = {} Mbytes".format(patientSize))
    except:
        pass
    print(json.dumps(typeDict, indent=4, sort_keys=True))

    return reloadedPatients