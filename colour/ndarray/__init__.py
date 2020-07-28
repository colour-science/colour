# -*- coding: utf-8 -*-

from __future__ import absolute_import

import sys

from .backend import NDimensionalArrayBackend


class ndarray(NDimensionalArrayBackend):
    def __getattr__(self, attribute):
        return super(ndarray, self).__getattr__(attribute)


sys.modules['colour.ndarray'] = ndarray()

del NDimensionalArrayBackend, sys
