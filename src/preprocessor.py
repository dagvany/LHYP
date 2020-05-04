#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from utils import get_logger, progress_bar
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut

# Attributes processing
import numpy as np
from decimal import Decimal

from models import Patient, Pathology, Image, ImageCollection

logger = get_logger(__name__)

class Preprocessor:
    '''
    Static class for patients dcm-s preprocessing
    It has config.json driven behavior
    '''

    @staticmethod
    def preprocessPatientsByConfig(config):
        stat = []
        numOfCorrect = 0
        numOfBroken = 0
        dirs = os.listdir(config['image_folder'])

        for index, patientId in enumerate(dirs):  # folder names = patientIds
            logger.info('Start: ' + patientId)
            progress_bar(index + 1, len(dirs), 20)

            folderPath = os.path.join(config['image_folder'], patientId)
            if not os.path.isdir(folderPath):
                logger.warning("Unwaited file: {}".format(folderPath))
                continue

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
            goalAmount = config['image_types'][imgType]["goal_amount"]
            patient.ImageTypes[imgType] = Preprocessor._getImages(path, sepAttr, goalAmount)

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
                except KeyError:
                    msg = '{} is unknown Pathology!'.format(pathologyEnumName)
                    logger.critical(msg)
                    print(msg)
                    exit(1)
                return pathology
            else:
                unhandledMetaTags = []
                for data in meta:
                    tag = data.split(':')[0]
                    unhandledMetaTags.append(tag)
                exText = '{}\n\tMeta data tags are not handled: {}'
                tags = ', '.join(unhandledMetaTags)
                exMsg = exText.format(metaFileLocation, tags)
                logger.critical(exMsg)
                print(exMsg)
                exit(1)
        except FileNotFoundError:
            exMsg = '{} is not found!'.format(metaFileLocation)
            logger.critical(exMsg)
            print(exMsg)
            exit(1)

    @staticmethod
    def _getImages(imageFolder, separatorAttr, goalAmount):
        files = os.listdir(imageFolder)
        dcmFiles = list(filter(lambda x: x.lower().endswith('.dcm'), files))
        if len(dcmFiles) == 0:
            logger.error(imageFolder + ' is empty!')
            return ImageCollection()

        iCol = ImageCollection()
        separatorAttrVals = []

        for file in dcmFiles:
            filePath = os.path.join(imageFolder, file)

            try:
                with pydicom.dcmread(filePath) as tempDcm:
                    try:
                        if separatorAttr:
                            attrVal = getattr(tempDcm, separatorAttr)
                            if attrVal not in separatorAttrVals:
                                separatorAttrVals.append(attrVal)
                            else:
                                continue
                        img = Preprocessor._createImage(tempDcm)
                        iCol.Views.append(img)
                    except AttributeError as ex:
                        # I found some images, what is not correct with MicroDicom,
                        # but processable with pydicom
                        # However error happens also with bad separator attr
                        msg = 'Separator ("{}") attr missing from img: {}' \
                            .format(separatorAttr, filePath)
                        logger.error(msg)
                        raise ex
            except Exception:
                logger.error('Broken: {}'.format(filePath), exc_info=True)

            iCol.Views = Preprocessor._filterCloseSameImages(iCol.Views, separatorAttr, goalAmount)
        return iCol

    @staticmethod
    def _createImage(dcmFile):
        img = Image()
        img.PixelArray = Preprocessor._convertDicomFloatToUint8(dcmFile)
        img.Attributes = Preprocessor._getAttributes(dcmFile)
        return img

    @staticmethod
    def _getAttributes(dcmFile):
        attributes = {}
        fileAttrs = set(dir(dcmFile))
        # attributes start with uppercase
        filteredAttrs = [a for a in fileAttrs if a[0].islower() or 'UID' in a]
        filteredAttrs = filteredAttrs + dir(object)
        # PixelData is already copied to pixel_array
        filteredAttrs.append('PixelData')
        fileAttrs = fileAttrs.difference(filteredAttrs)

        # everyelse will be droped
        enabledTypes = (int, str, Decimal, np.float)
        convertableTypes = {
            pydicom.valuerep.DSfloat: np.float,
            pydicom.valuerep.DSdecimal: Decimal,
            pydicom.valuerep.IS: np.int,
            pydicom.multival.MultiValue: list
        }

        for attr in fileAttrs:
            dcmAttr = getattr(dcmFile, attr)

            if isinstance(dcmAttr, tuple(convertableTypes.keys())):
                try:
                    attributes[attr] = Preprocessor._convertDicomType(convertableTypes, dcmAttr)
                except KeyError:
                    # pydicom inheritance
                    # eg.: Sequence is inheritance of MultiValue
                    pass
            elif isinstance(dcmAttr, enabledTypes):
                attributes[attr] = dcmAttr
        return attributes

    @staticmethod
    def _convertDicomType(convertableTypes, dcmAttr):
        if isinstance(dcmAttr, tuple(convertableTypes.keys())):
            newType = convertableTypes[type(dcmAttr)]
            dcmAttr = newType(dcmAttr)
            if isinstance(dcmAttr, list):
                return [Preprocessor._convertDicomType(convertableTypes, x) for x in dcmAttr]
        return dcmAttr

    @staticmethod
    def _convertDicomFloatToUint8(dcm):
        # http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.11.html#sect_C.11.1.1.2
        image = apply_modality_lut(dcm.pixel_array, dcm)
        image = (np.maximum(image, 0) / image.max()) * 255.0
        image = image.astype(np.uint8)

        return image

    @staticmethod
    def _filterCloseSameImages(views, sepAttr, goalAmount):
        lenViews = len(views)
        dropableAmount = lenViews - goalAmount
        if goalAmount >= lenViews or dropableAmount <= 0:
            return views

        testAttr = views[0].Attributes[sepAttr]
        if not isinstance(testAttr, list) \
                or not isinstance(testAttr[0], (int, float, np.float, Decimal)):
            return views

        ipValues = {}
        pairGenerator = ((x, y) for x in range(lenViews) for y in range(lenViews) if y > x)
        for a, b in pairGenerator:
            x = views[a].Attributes[sepAttr]
            y = views[b].Attributes[sepAttr]
            innerProduct = np.inner(x, y)
            ipValues.update({innerProduct: b})

        drop = []
        for ip in sorted(ipValues, reverse=True):
            if len(drop) == dropableAmount: break
            index = ipValues[ip]
            if index not in drop:
                drop.append(index)

        logger.debug('len(views): {} dropableAmount: {}, drop: {}'.format(len(views), dropableAmount, str(drop)))
        logger.debug(json.dumps(ipValues, indent=4, sort_keys=True))

        for i in sorted(drop, reverse=True):
            del views[i]

        return views
