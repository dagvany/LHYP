#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import functools
import operator

from utils import get_logger
from .image import Image

logger = get_logger(__name__)

class ImageCollection:
    def __init__(self):
        self.Views = []
        self.CommonAttibutes = {}

    def organiseAttributes(self):
        logger.debug('len(Views) = {}'.format(len(self.Views)))
        if len(self.Views) < 1:
            return

        allAttr = [list(img.Attributes.keys()) for img in self.Views]
        logger.debug('raw allAttr: {}'.format(allAttr))
        allAttr = set(functools.reduce(operator.add, allAttr))
        allAttr = allAttr.difference(set(dir(dict)))
        deletableAttrs = list(filter(lambda x: x.startswith('_'), allAttr))
        allAttr = list(filter(lambda x: x not in deletableAttrs, allAttr))
        logger.debug('deletableAttrs: {}'.format(deletableAttrs))
        logger.debug('final allAttr: {}'.format(allAttr))
        
        commonAttr = []
        for attr in allAttr:
            attrVal = None
            for img in self.Views:
                if attr not in img.Attributes:
                    break
                elif attrVal is None:
                    attrVal = img.Attributes[attr]
                elif img.Attributes[attr] != attrVal:
                    break
            else:
                commonAttr.append(attr)
        logger.debug('commonAttr: {}'.format(commonAttr))
        
        for attr in commonAttr:
            self.CommonAttibutes[attr] = self.Views[0].Attributes[attr]
        
        for img in self.Views:
            for attr in deletableAttrs + commonAttr:
                del img.Attributes[attr]
        