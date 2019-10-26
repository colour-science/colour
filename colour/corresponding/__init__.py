# -*- coding: utf-8 -*-

from __future__ import absolute_import

from .datasets import *  # noqa
from . import datasets
from .prediction import (CorrespondingColourDataset,
                         CorrespondingChromaticitiesPrediction,
                         corresponding_chromaticities_prediction_CIE1994,
                         corresponding_chromaticities_prediction_CMCCAT2000,
                         corresponding_chromaticities_prediction_Fairchild1990,
                         corresponding_chromaticities_prediction_VonKries,
                         CORRESPONDING_CHROMATICITIES_PREDICTION_MODELS,
                         corresponding_chromaticities_prediction)

__all__ = []
__all__ += datasets.__all__
__all__ += [
    'CorrespondingColourDataset', 'CorrespondingChromaticitiesPrediction',
    'corresponding_chromaticities_prediction_CIE1994',
    'corresponding_chromaticities_prediction_CMCCAT2000',
    'corresponding_chromaticities_prediction_Fairchild1990',
    'corresponding_chromaticities_prediction_VonKries',
    'CORRESPONDING_CHROMATICITIES_PREDICTION_MODELS',
    'corresponding_chromaticities_prediction'
]
