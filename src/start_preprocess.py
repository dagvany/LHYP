#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from utils import get_logger, progress_bar
from preprocessor import Preprocessor

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
    config = Preprocessor.getConfiguration("config_preprocess.json")

    patientStats = preprocessPatiens(config)
    tldrStat = 'Total: {:5d}, Correct: {:5d}, Broken: {:5d}'.format(
        patientStats[1] + patientStats[2],
        patientStats[1],
        patientStats[2]
    )
    print(tldrStat)
    logger.info(tldrStat)
    logger.info('\n'.join(patientStats[0]))
