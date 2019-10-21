#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import functools
import operator

from .image import Image

class Image_collection:
    def __init__(self):
        self.views = []
        self.common_attibutes = {}

    def organiseAttributes(self):
        all_attr = [list(element.keys()) for element in self.views]
        all_attr = set(functools.reduce(operator.add, all_attr))
        
        common_attr = []
        for attr in all_attr:
            # TODO: refactoring
            if attr in self.views[0]:
                for img in self.views[1:]:
                    if attr in img.attributes:
                        if self.views[0][attr] != img.attributes[attr]:
                            break
                    else:
                        break
                else:
                    common_attr.append(attr)
            else:
                common_attr.append(attr)
        
        for attr in common_attr:
            self.common_attibutes[attr] = self.views[0][attr]
            for img in self.views:
                del img[attr]