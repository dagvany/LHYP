#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from timeit import default_timer as timer
from datetime import timedelta

from models import Patient
from models import patient_enums
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


def _loadPatients(folder, failedFolder, typeDict):
    reloadedPatients = []
    numOfEmpty = 0
    numOfError = 0
    progressTotal = sum(len(f[2]) for f in os.walk(folder))
    progressCounter = 0
    for pathology in os.listdir(folder):
        for fileName in os.listdir(os.path.join(folder, pathology)):
            progressCounter += 1
            progress_bar(progressCounter, progressTotal, 20)
            fullPath = os.path.join(folder, pathology, fileName)
            logger.debug('Pickle load: {}'.format(fullPath))
            try:
                patient = Patient.unSerialize(fullPath)
                if patient.hasAnyImage():
                    pathName = patient.Pathology.name
                    typeDict[pathName] = typeDict[pathName] + 1
                    reloadedPatients.append(patient)
                else:
                    logger.warning('Empty: {}'.format(fullPath))
                    destPath = os.path.join(failedFolder, fileName)
                    os.replace(fullPath, destPath)
                    numOfEmpty = numOfEmpty + 1
            except Exception:
                logger.error('Broken: {}'.format(fullPath), exc_info=True)
                destPath = os.path.join(failedFolder, fileName)
                os.replace(fullPath, destPath)
                numOfError = numOfError + 1
    return reloadedPatients, numOfEmpty, numOfError


def unSerializePatients(folder, failedFolder):
    print('Reload from Pickle')
    start = timer()
    reloadedPatients = {}
    numOfEmpty = 0
    numOfError = 0
    pathologyNames = filter(lambda p: p[0] != '_', dir(patient_enums.Pathology))
    typeDict = dict.fromkeys(list(pathologyNames), 0)

    for category in ['train', 'validate']:
        trainPath = os.path.join(folder, category)
        [patients, nEmpty, nError] = _loadPatients(
            trainPath, failedFolder, typeDict)
        reloadedPatients[category] = patients
        numOfEmpty += nEmpty
        numOfError += nError
    end = timer()

    numOfTotal = sum([len(reloadedPatients[i]) for i in reloadedPatients])
    numOfTotal = numOfTotal + numOfError + numOfEmpty
    print('\nMeasured time: {}'.format(str(timedelta(seconds=end - start))))
    print('Total: {:5d}, Failed: {:5d}, Empty: {:5d}'.format(
        numOfTotal, numOfError, numOfEmpty))
    print(json.dumps(typeDict, indent=4, sort_keys=True))

    return reloadedPatients
