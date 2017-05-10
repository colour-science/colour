#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

from .dataset import *  # noqa
from . import dataset
from .meng2015 import XYZ_to_spectral_Meng2015
from .smits1999 import RGB_to_spectral_Smits1999

__all__ = []
__all__ += dataset.__all__
__all__ += ['XYZ_to_spectral_Meng2015']
__all__ += ['RGB_to_spectral_Smits1999']
