#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from utils import get_logger
from preprocessor import Preprocessor

# Serialization
import os
import pickle
from models import Patient

# Time measure
from timeit import default_timer as timer
from datetime import timedelta

logger = get_logger(__name__)

def get_configuration(json_file_location = "config.json"):
    try:
        with open(json_file_location, 'r') as fp:
            config = json.load(fp)
    except FileNotFoundError:
        logger.critical(json_file_location + ' is not found!')
        exit(1)
    except ValueError:
        logger.critical(json_file_location + ' JSON format is not correct!')
        exit(1) 
    
    return config

def print_table(patients):
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

config = get_configuration("config.json")
start = timer()
try:
    patients = Preprocessor.getPatients(config)
except Exception as ex:
    logger.critical(ex, exc_info=True)
    exit(1)
end = timer()
print('Preprocessing')
print('Measured time: {}'.format(str(timedelta(seconds=end-start))))

print_table(patients)

for p in patients:
    file_name = p.ID + '.pickle'
    if not os.path.exists(config["pickle_folder"]):
        os.makedirs(config["pickle_folder"])
    path = os.path.join(config["pickle_folder"], file_name)
    try:
        with open(path, 'wb') as fp:
            pickle.dump(p, fp)
    except Exception:
        msg = 'Patient: {} serialization (dump) is failed!'.format(p.ID)
        logger.critical(msg, exc_info=True)
        exit(1)
    logger.info('Dumped: {}'.format(file_name))

start = timer()
reloaded_patients = []
for file_name in os.listdir(config["pickle_folder"]):
    path = os.path.join(config["pickle_folder"], file_name)
    try:
        with open(path, 'rb') as fp:
            patient = pickle.load(fp)
        reloaded_patients.append(patient)
    except Exception:
        msg = 'Patient: {} serialization (load) is failed!'.format(p.ID)
        logger.critical(msg, exc_info=True)
        exit(1)
    logger.info('Loaded: {}'.format(logger.info('Pickled: {}'.format(file_name))))

end = timer()
print('Reload from Pickle')
print('Measured time: {}'.format(str(timedelta(seconds=end-start))))

print_table(reloaded_patients)