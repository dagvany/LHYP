#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from enum import Enum

class Pathology(Enum):
    UNDEFINED = None
    NORM      = 0
    HCM       = 1
    AMY       = 2
    EMF       = 3

class Gender(Enum):
    UNDEFINED = None
    MALE      = 0
    FEMALE    = 1