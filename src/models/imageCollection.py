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

        all_attr = [list(img.Attributes.keys()) for img in self.Views]
        logger.debug('raw all_attr: {}'.format(all_attr))
        all_attr = set(functools.reduce(operator.add, all_attr))
        all_attr = all_attr.difference(set(dir(dict)))
        logger.debug('final all_attr: {}'.format(all_attr))
        
        common_attr = []
        for attr in all_attr:
            attr_val = None
            for img in self.Views:
                if attr not in img.Attributes:
                    break
                elif attr_val is None:
                    attr_val = img.Attributes[attr]
                elif img.Attributes[attr] != attr_val:
                    break
            else:
                common_attr.append(attr)
        logger.debug('common_attr: {}'.format(common_attr))
        
        for attr in common_attr:
            self.CommonAttibutes[attr] = self.Views[0].Attributes[attr]
            for img in self.Views:
                del img.Attributes[attr]