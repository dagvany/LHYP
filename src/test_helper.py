#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import main
from preprocessor import Preprocessor
from matplotlib import pyplot as plt
import math
import csv
import autoencoder
import random
import sys
import cv2


def drawPatientImageType(imgType, patients, labelAttr=None):
    sepAttr = config['image_types'][imgType]["separator_attr"]
    for p in patients:
        cols = 3
        rows = 2
        title = '{} -> {}'.format(p.ID, sepAttr)
        fig = plt.figure(title)
        fig.suptitle(title)

        numOfImgs = len(p.ImageTypes[imgType].Views)
        if numOfImgs > 6:
            rows = math.ceil(math.sqrt(numOfImgs))
            cols = rows

        for i, img in enumerate(p.ImageTypes[imgType].Views):
            fig.add_subplot(rows, cols, i + 1)
            if labelAttr:
                plt.xlabel(str(img.Attributes[labelAttr]))
            else:
                plt.xlabel('\n'.join(str(img.Attributes[sepAttr]).split(',')))
            plt.imshow(img.PixelArray, cmap='gray')
        plt.show()


def createCSV(imgType, patients):
    fieldnames = []
    for p in patients:
        fieldnames = fieldnames + list(p.ImageTypes[imgType].Views[0].Attributes.keys())
    fieldnames = list(set(fieldnames))
    fieldnames = ['ID'] + fieldnames

    with open('/media/Data/Dipterv_MRI/test/patients.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for p in patients:
            for img in p.ImageTypes[imgType].Views:
                row = img.Attributes
                row.update({'ID': p.ID})
                writer.writerow(row)


config = Preprocessor.getConfiguration("config_test.json")


def preprocess(config):
    for line in main.preprocessPatiens(config)[0]:
        print(line)


# preprocess(config)
patients = main.unSerializePatients(config)
# drawPatientImageType('la', patients)
# createCSV('la', patients)

height = sys.maxsize
width = sys.maxsize
patientsImages = []
for p in patients:
    for imgType in config['image_types']:
        for img in p.ImageTypes[imgType].Views:
            patientsImages.append(img.PixelArray)
            h, w = img.PixelArray.shape
            if h < height: height = h
            if w < width: width = w

height = width = 10
for i in range(len(patientsImages)):
    patientsImages[i] = cv2.resize(patientsImages[i], (height, width), interpolation=cv2.INTER_LANCZOS4)

random.shuffle(patientsImages)
print('h: {}, w: {}, images: {}'.format(height, width, len(patientsImages)))

autoencoder.run(patientsImages, height, width,
                intermediateLayerSize=100,
                latentLayerSize=50,
                numEpochs=config['epoch'],
                batchSize=config['batch_size'],
                learningRate=1e-3,
                imgFolder=config['img_folder'],
                modelFolder=config['pytorch_model_folder'],
                cudaSeed=config['cuda_seed'])
