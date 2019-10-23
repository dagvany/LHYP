#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .patient_enums import Pathology, Gender
from .image_collection import Image_collection

class Patient:
    def __init__(self, patientId):
        self.ID = patientId
        self.Pathology = Pathology.UNDEFINED
        self.ImageTypes = {}