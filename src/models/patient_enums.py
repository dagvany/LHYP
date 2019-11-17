#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from enum import Enum

class Pathology(Enum):
    UNDEFINED     = -1
    NORM          = 0
    HCM           = 1
    AMY           = 2
    AMYLOIDOSIS   = 2
    EMF           = 3
    ADULT_M_SPORT = 4
    ADULT_F_SPORT = 5
    U18_F         = 6
    U18_M         = 7
    AORTASTENOSIS = 8
    FABRY         = 9
