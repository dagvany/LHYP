#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from utils import get_logger
import pydicom as dicom
from models import Patient, Pathology, Image, Image_collection

logger = get_logger(__name__)

class Preprocessor:
    '''
    Abstract Class
    TODO
    '''
    @staticmethod
    def getPatients(config):
        patients = []
        dirs = os.listdir(config['image_folder'])

        for patientId in dirs: # folder names = patientIds
            logger.info('Start: ' + patientId)
            patient = Patient(patientId)
            meta_folder = os.path.join(config['image_folder'], patientId)
            patient.Pathology = Preprocessor._getPathology(meta_folder)
            
            for img_type in config['image_types']:
                path = os.path.join(
                    config['image_folder'],
                    patientId,
                    config['image_types'][img_type]["folder"])
                sep_attr = config['image_types'][img_type]["separator_attr"]
                patient.ImageTypes[img_type] = Preprocessor._getImages(path, sep_attr)
                
                if len(patient.ImageTypes[img_type].Views) == 0:
                    logger.error('{} doesnt contain correct {} images'.format(
                        patientId,
                        img_type.upper()))
            
            patients.append(patient)
            logger.info('Finished: ' + patientId)
        return patients

    @staticmethod
    def _getPathology(folder):
        meta_file_location = os.path.join(folder, 'meta.txt')
        try:
            with open(meta_file_location, 'r') as fp:
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
                    ex_text = '{}\n\tFollowing meta data tags are not handled: {}'
                    tags = ', '.join(unhandledMetaTags)
                    ex_message = ex_text.format(meta_file_location, tags)
                    logger.error(ex_message)
                    raise Exception(ex_message)
        except FileNotFoundError:
            logger.error(meta_file_location + ' is not found!')
    
    @staticmethod
    def _getImages(image_folder, separator_attr):
        dcm_files = os.listdir(image_folder)
        if len(dcm_files) == 0:
            logger.error(image_folder + ' is empty!')
                
        i_collection = Image_collection()
        separator_attr_vals = []

        for file in dcm_files:
            file_path = os.path.join(image_folder, file)
            if file.find('.dcm') == -1:
                logger.warning('Unwanted file: {}'.format(file_path))
                continue

            try:
                with dicom.dcmread(file_path) as temp_dcm:
                    try:
                        if separator_attr:
                            attr_val = getattr(temp_dcm, separator_attr)
                            #logger.debug('file: {}\nattr_val: {}, sep_attr_vals: {}'.format(file_path, attr_val, separator_attr_vals))
                            if attr_val not in separator_attr_vals:
                                separator_attr_vals.append(attr_val)
                            else:
                                continue
                        img = Preprocessor._createImage(temp_dcm)
                        i_collection.Views.append(img)
                    except AttributeError as ex:
                        # I found some images, what is not correct with MicroDicom,
                        # but processable with pydicom
                        # However error happens also with bad separator attr
                        msg = 'Separator ("{}") attr missing from img: {}'\
                            .format(separator_attr, file_path)
                        logger.error(msg)
                        raise ex
            except Exception:
                logger.error('Broken: {}'.format(file_path), exc_info=True)

        i_collection.organiseAttributes()
        return i_collection

    @staticmethod
    def _createImage(dcm_file):
        img = Image()
        img.PixelArray = dcm_file.pixel_array

        file_attrs = set(dir(dcm_file))
        # attributes start with uppercase
        filteredAttrs = [a for a in file_attrs if a[0].islower()]
        filteredAttrs = filteredAttrs + dir(object)
        # PixelData is already copied to pixel_array
        filteredAttrs.append('PixelData')
        file_attrs = file_attrs.difference(filteredAttrs)

        for attr in file_attrs:
            img.Attributes[attr] = getattr(dcm_file, attr)
        return img
