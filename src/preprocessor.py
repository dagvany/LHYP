#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from utils import get_logger, progress_bar
import pydicom as dicom

from models import Patient, Pathology, Image, ImageCollection

logger = get_logger(__name__)

class Preprocessor:
    '''
    Abstract class for patients dcm-s preprocessing
    It has config.json driven behavior
    '''

    @staticmethod
    def getConfiguration(jsonFileLocation = "config.json"):
        try:
            with open(jsonFileLocation, 'r') as fp:
                config = json.load(fp)
        except FileNotFoundError:
            logger.critical(jsonFileLocation + ' is not found!')
            exit(1)
        except ValueError:
            logger.critical(jsonFileLocation + ' JSON format is not correct!')
            exit(1) 
        
        return config

    @staticmethod
    def getPatients(config):
        '''
        config: config.json -> dict
        return with preprocessed patient array
        '''
        patients = []
        dirs = os.listdir(config['image_folder'])

        for index, patientId in enumerate(dirs): # folder names = patientIds
            progress_bar(index+1, len(dirs), 20)
            logger.info('Start: ' + patientId)
            patient = Patient(patientId)
            metaFolder = os.path.join(config['image_folder'], patientId)
            patient.Pathology = Preprocessor._getPathology(metaFolder)
            
            for imgType in config['image_types']:
                path = os.path.join(
                    config['image_folder'],
                    patientId,
                    config['image_types'][imgType]["folder"])
                sepAttr = config['image_types'][imgType]["separator_attr"]
                patient.ImageTypes[imgType] = Preprocessor._getImages(path, sepAttr)
                
                if len(patient.ImageTypes[imgType].Views) == 0:
                    logger.error('{} doesnt contain correct {} images'.format(
                        patientId,
                        imgType.upper()))
            
            patients.append(patient)
            logger.info('Finished: ' + patientId)
        return patients

    @staticmethod
    def _getPathology(folder):
        metaFileLocation = os.path.join(folder, 'meta.txt')
        try:
            with open(metaFileLocation, 'r') as fp:
                meta = fp.read().replace('\r', '')
                meta = meta.replace(' ', '').split('\n')
                meta.remove('')
                if len(meta) == 1:
                    meta = meta[0].replace(' ', '').split(':')
                    pathology = Pathology[meta[1].upper()]
                    return pathology
                else:
                    unhandledMetaTags = []
                    for data in meta:
                        tag = data.split(':')[0]
                        unhandledMetaTags.append(tag)
                    exText = '{}\n\tFollowing meta data tags are not handled: {}'
                    tags = ', '.join(unhandledMetaTags)
                    exMsg = exText.format(metaFileLocation, tags)
                    logger.error(exMsg)
                    raise Exception(exMsg)
        except FileNotFoundError:
            logger.error(metaFileLocation + ' is not found!')
    
    @staticmethod
    def _getImages(imageFolder, separatorAttr):
        dcmFiles = os.listdir(imageFolder)
        if len(dcmFiles) == 0:
            logger.error(imageFolder + ' is empty!')
                
        iCollection = ImageCollection()
        separatorAttrVals = []

        for file in dcmFiles:
            filePath = os.path.join(imageFolder, file)
            if file.find('.dcm') == -1:
                logger.warning('Unwanted file: {}'.format(filePath))
                continue

            try:
                with dicom.dcmread(filePath) as tempDcm:
                    try:
                        if separatorAttr:
                            attrVal = getattr(tempDcm, separatorAttr)
                            #logger.debug('file: {}\nattr_val: {}, sepAttrVals: {}'.format(filePath, attrVal, separatorAttrVals))
                            if attrVal not in separatorAttrVals:
                                separatorAttrVals.append(attrVal)
                            else:
                                continue
                        img = Preprocessor._createImage(tempDcm)
                        iCollection.Views.append(img)
                    except AttributeError as ex:
                        # I found some images, what is not correct with MicroDicom,
                        # but processable with pydicom
                        # However error happens also with bad separator attr
                        msg = 'Separator ("{}") attr missing from img: {}'\
                            .format(separatorAttr, filePath)
                        logger.error(msg)
                        raise ex
            except Exception:
                logger.error('Broken: {}'.format(filePath), exc_info=True)

        iCollection.organiseAttributes()
        return iCollection

    @staticmethod
    def _createImage(dcmFile):
        img = Image()
        img.PixelArray = dcmFile.pixel_array

        fileAttrs = set(dir(dcmFile))
        # attributes start with uppercase
        filteredAttrs = [a for a in fileAttrs if a[0].islower()]
        filteredAttrs = filteredAttrs + dir(object)
        # PixelData is already copied to pixel_array
        filteredAttrs.append('PixelData')
        fileAttrs = fileAttrs.difference(filteredAttrs)

        for attr in fileAttrs:
            img.Attributes[attr] = getattr(dcmFile, attr)
        return img
