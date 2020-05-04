#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math

from utils import get_logger, progress_bar
from preprocessor import Preprocessor
from proc_utils import getConfiguration

from models import Pathology

# Time measure
from timeit import default_timer as timer
from datetime import timedelta

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

if __name__ == '__main__':
    config = getConfiguration("config_preprocess.json")

    patientStats = preprocessPatiens(config)
    tldrStat = 'Total: {:5d}, Correct: {:5d}, Broken: {:5d}'.format(
        patientStats[1] + patientStats[2],
        patientStats[1],
        patientStats[2]
    )
    print(tldrStat)
    logger.info(tldrStat)
    print(patientStats[0])
    txt = ['\t'.join(line) for line in patientStats[0]]
    logger.info('\n'.join(txt))

    pStat = {}
    for p in patientStats[0]:
        pathology = Pathology[p[1].split('.')[1]]
        if pathology in pStat:
            pStat[pathology]['num'] += 1
            pStat[pathology]['ids'].append(p[0])
        else:
            pStat[pathology] = {'num': 1, 'ids': [p[0]]}

    validatePatientIds = []
    for pathology in pStat:
        if pathology is not Pathology.UNDEFINED:
            data = pStat[pathology]
            numOfValidate = math.ceil(data['num'] * config['validate_rate'])
            validatePatientIds += data['ids'][0: numOfValidate]
    print(validatePatientIds)

    validateFolder = os.path.join(config['pickle_folder'], 'validate')
    if not os.path.exists(validateFolder):
        os.makedirs(validateFolder)

    for pid in validatePatientIds:
        src = os.path.join(config['pickle_folder'], '{}.pickle'.format(pid))
        dst = os.path.join(validateFolder, '{}.pickle'.format(pid))
        os.replace(src, dst)

    logger.info('Preprocess is done.')

