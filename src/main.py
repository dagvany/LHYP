#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from utils import get_logger
from preprocessor import Preprocessor

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

config = get_configuration("config.json")
try:
    patients = Preprocessor.getPatients(config)
except Exception as ex:
    logger.critical(ex, exc_info=True)
    exit(1)

for p in patients:
    img_meta = []
    for t in p.ImageTypes:
        img_meta.append('{}: {:4d} attr: {:4d}'.format(
            t,
            len(p.ImageTypes[t].views),
            len(p.ImageTypes[t].common_attibutes)))
    
    print('\t'.join([
        p.ID, 
        str(p.Pathology),
        str(img_meta)]))