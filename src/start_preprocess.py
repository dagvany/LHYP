#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import json
import numpy as np
from decimal import Decimal
from timeit import default_timer as timer
from datetime import timedelta

import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut

from utils import get_logger, progress_bar
from proc_utils import getConfiguration
from models import Patient
from models import Pathology


logger = get_logger(__name__)


def preprocessPatients(config):
    print('Preprocessing Patients Images')
    start = timer()
    try:
        patientStats = preprocessPatientsByConfig(config)
    except Exception as ex:
        logger.critical(ex, exc_info=True)
        print(ex)
        exit(1)
    end = timer()
    print('Measured time: {}'.format(str(timedelta(seconds=end-start))))
    return patientStats


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
            logger.warning("Unwanted file: {}".format(folderPath))
            continue
        patient = preprocessPatient(config, patientId)
        if patient.hasAnyImage():
            path = os.path.join(config['root'], config["pickle_folder"])
            patient.serialize(path)
            stat.append(getStatFromPatient(patient))
            numOfCorrect = numOfCorrect + 1
        else:
            txt = "hasn't contains any images!"
            msg = '{} {}'.format(patient.ID, txt)
            stat.append([patient.ID, str(Pathology.UNDEFINED), txt])
            logger.warning(msg)
            numOfBroken = numOfBroken + 1
        logger.info('Finished: ' + patientId)
    return stat, numOfCorrect, numOfBroken


def getStatFromPatient(patient):
    imgStat = []
    for imgType in patient.ImageTypes:
        numOfViews = len(patient.ImageTypes[imgType])
        stat = '{}: {:4d}'.format(imgType, numOfViews)
        imgStat.append(stat)
    data = [patient.ID, str(patient.Pathology), str(imgStat)]
    return data


def preprocessPatient(config, patientId):
    patientFolder = os.path.join(config['image_folder'], patientId)
    patient = Patient(patientId)
    patient.Pathology = getPathology(patientFolder)
    for imgType in config['image_types']:
        iTypeFolder = config['image_types'][imgType]["folder"]
        path = os.path.join(patientFolder, iTypeFolder)
        sepAttr = config['image_types'][imgType]["separator_attr"]
        goalAmount = config['image_types'][imgType]["goal_amount"]
        try:
            images, attrs = getImages(path, sepAttr, goalAmount)
            patient.ImageTypes[imgType] = images
            patient.Sex = attrs.setdefault('PatientSex', patient.Sex)
            patient.Weight = attrs.setdefault('PatientWeight', patient.Weight)
            if len(patient.ImageTypes[imgType]) == 0:
                msg = "{} doesn't contain correct {} images".format(
                    patientId, imgType.upper())
                logger.error(msg)
        except ValueError as ex:
            logger.error(ex)
    return patient


def getPathology(folder):
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


def convertDicomType(dcmAttr):
    convertableTypes = {
        pydicom.valuerep.DSfloat: np.float,
        pydicom.valuerep.DSdecimal: Decimal,
        pydicom.valuerep.IS: np.int,
        pydicom.multival.MultiValue: list
    }
    if isinstance(dcmAttr, tuple(convertableTypes.keys())):
        newType = convertableTypes[type(dcmAttr)]
        dcmAttr = newType(dcmAttr)
        if isinstance(dcmAttr, list):
            return [convertDicomType(x) for x in dcmAttr]
    return dcmAttr


def getImages(imageFolder, separatorAttr, goalAmount):
    files = os.listdir(imageFolder)
    dcmFiles = list(filter(lambda x: x.lower().endswith('.dcm'), files))
    if len(dcmFiles) == 0:
        msg = imageFolder + ' is empty!'
        logger.error(msg)
        raise ValueError(msg)

    attrs = ['PatientSex', 'PatientWeight']
    patientAttributes = {}.fromkeys(attrs)
    allImages = []
    for file in dcmFiles:
        filePath = os.path.join(imageFolder, file)
        try:
            with pydicom.dcmread(filePath) as tempDcm:
                try:
                    if separatorAttr:
                        attrVal = getattr(tempDcm, separatorAttr)
                        attrVal = convertDicomType(attrVal)
                        isContain = any([attrVal is x['sepVal']
                                        for x in allImages])
                        if not isContain and tempDcm.PixelData:
                            allImages.append({
                                'img': tempDcm.PixelData,
                                'sepVal': attrVal
                            })
                        else:
                            continue
                    if not all(patientAttributes.values()):
                        for attr in patientAttributes:
                            attrVal = getattr(tempDcm, attr)
                            if attrVal:
                                patientAttributes[attr] = attrVal
                except AttributeError as ex:
                    # I found some images, what is not correct with
                    # MicroDicom, but processable with pydicom
                    # However error happens also with bad separator attr
                    msg = 'Separator ("{}") attr missing from img: {}' \
                        .format(separatorAttr, filePath)
                    logger.error(msg)
                    raise ex
                except TypeError as ex:
                    logger.critical(ex, exc_info=True)
                    print("Critical error -> read log!")
                    exit(1)
        except Exception:
            logger.error('Broken: {}'.format(filePath), exc_info=True)

        images = filterCloseSameImages(allImages, goalAmount)
    return images, patientAttributes


def filterCloseSameImages(allImages, goalAmount):
    dropableAmount = len(allImages) - goalAmount
    if goalAmount >= len(allImages) or dropableAmount <= 0:
        return [x['img'] for x in allImages]

    ipValues = {}
    pairs = ((x, y) for x in range(len(allImages))
             for y in range(len(allImages)) if y > x)
    for a, b in pairs:
        x = allImages[a]['sepVal']
        y = allImages[b]['sepVal']
        innerProduct = np.inner(x, y)
        ipValues.update({innerProduct: b})

    drop = []
    for ip in sorted(ipValues, reverse=True):
        if len(drop) == dropableAmount:
            break
        index = ipValues[ip]
        if index not in drop:
            drop.append(index)

    for i in sorted(drop, reverse=True):
        del allImages[i]

    return [x['img'] for x in allImages]


if __name__ == '__main__':
    config = getConfiguration("config_preprocess.json")

    patientStats = preprocessPatients(config)
    tldrStat = 'Total: {:5d}, Correct: {:5d}, Broken: {:5d}'.format(
        patientStats[1] + patientStats[2],
        patientStats[1],
        patientStats[2]
    )
    print(tldrStat)
    logger.info(tldrStat)
    txt = ['\t'.join(line) for line in patientStats[0]]
    logger.debug('\n'.join(txt))

    pathologyStat = {}
    for p in patientStats[0]:
        pathology = Pathology[p[1].split('.')[1]]
        if pathology in pathologyStat:
            pathologyStat[pathology]['num'] += 1
            pathologyStat[pathology]['ids'].append(p[0])
        else:
            pathologyStat[pathology] = {'num': 1, 'ids': [p[0]]}
    logger.debug(pathologyStat)
    if Pathology.UNDEFINED in pathologyStat:
        del pathologyStat[Pathology.UNDEFINED]

    validatePatients = {}
    for pathology in pathologyStat:
        data = pathologyStat[pathology]
        numOfValidate = math.ceil(data['num'] * config['validate_rate'])
        validatePatients[pathology.name] = data['ids'][0: numOfValidate]

    path = os.path.join(config['root'], config["pickle_folder"])
    validateFolder = os.path.join(path, 'validate')

    for pathologyName in validatePatients:
        for pid in validatePatients[pathologyName]:
            src = os.path.join(path, pathologyName, '{}.pickle'.format(pid))
            dstFolderPath = os.path.join(validateFolder, pathologyName)
            dst = os.path.join(dstFolderPath, '{}.pickle'.format(pid))
            if not os.path.exists(dstFolderPath):
                os.makedirs(dstFolderPath)
            os.replace(src, dst)

    for pathology in pathologyStat:
        src = os.path.join(path, pathology.name)
        dst = os.path.join(path, 'train', pathology.name)
        if not os.path.exists(dst):
            os.makedirs(dst)
        os.replace(src, dst)

    logger.info('Preprocess is done.')
