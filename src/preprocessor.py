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
            msg = jsonFileLocation + ' is not found!'
            logger.critical(msg)
            print(mgs)
            exit(1)
        except ValueError:
            msg = jsonFileLocation + ' JSON format is not correct!'
            logger.critical(msg)
            print(msg)
            exit(1) 
        
        return config

    @staticmethod
    def preprocessPatientsByConfig(config):
        '''
        config: config.json -> dict
        return with preprocessed patient array
        '''
        stat = []
        numOfCorrect = 0
        numOfBroken = 0
        dirs = os.listdir(config['image_folder'])

        for index, patientId in enumerate(dirs): # folder names = patientIds
            logger.info('Start: ' + patientId)
            progress_bar(index+1, len(dirs), 20)

            patient = Preprocessor.preprocessPatient(config, patientId)
            if patient.hasAnyImage():
                patient.organizeImageAttributes()
                patient.serialize(config["pickle_folder"])
                stat.append(Preprocessor._getStatFromPatient(patient))
                numOfCorrect = numOfCorrect + 1
            else:
                msg = '{} hasnt contains any images!'.format(patient.ID)
                stat.append(msg)
                logger.warning(msg)
                numOfBroken = numOfBroken + 1
            
            logger.info('Finished: ' + patientId)
        return stat, numOfCorrect, numOfBroken
    
    @staticmethod
    def preprocessPatient(config, patientId):
        patientFolder = os.path.join(config['image_folder'], patientId)
        patient = Patient(patientId)
        patient.Pathology = Preprocessor._getPathology(patientFolder)

        for imgType in config['image_types']:
            iTypeFolder = config['image_types'][imgType]["folder"]
            path = os.path.join(patientFolder, iTypeFolder)
            sepAttr = config['image_types'][imgType]["separator_attr"]
            patient.ImageTypes[imgType] = Preprocessor._getImages(path, sepAttr)
            
            if len(patient.ImageTypes[imgType].Views) == 0:
                msg = '{} doesnt contain correct {} images'.format(
                    patientId, imgType.upper())
                logger.error(msg)
        
        return patient
    
    @staticmethod
    def _getStatFromPatient(patient):
        imgStat = []
        for imgType in patient.ImageTypes:
            numOfViews = len(patient.ImageTypes[imgType].Views)
            numOfCommonAttrs = len(patient.ImageTypes[imgType].CommonAttibutes)
            stat = '{}: {:4d} commonAttr: {:4d}'.format(
                imgType, numOfViews, numOfCommonAttrs)
            imgStat.append(stat)
        return '\t'.join([patient.ID, str(patient.Pathology), str(imgStat)])

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
                    pathologyEnumName = meta[1].upper()
                    try:
                        pathology = Pathology[pathologyEnumName]
                    except ValueError:
                        msg = '{} is unknown Pathology!'.format(pathologyEnumName)
                        logger.critical(msg)
                        print(msg)
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
