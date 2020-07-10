# -*- coding: utf-8 -*-

from __future__ import absolute_import

from .mallett2019 import (SPECTRAL_SHAPE_sRGB_MALLETT2019,
                          BASIS_FUNCTIONS_sRGB_MALLETT2019)
from .otsu2018 import (OTSU_2018_SPECTRAL_SHAPE, OTSU_2018_BASIS_FUNCTIONS,
                       OTSU_2018_MEANS, select_cluster_Otsu2018)
from .smits1999 import SDS_SMITS1999

__all__ = [
    'SPECTRAL_SHAPE_sRGB_MALLETT2019', 'BASIS_FUNCTIONS_sRGB_MALLETT2019'
]
__all__ += [
    'OTSU_2018_SPECTRAL_SHAPE', 'OTSU_2018_BASIS_FUNCTIONS', 'OTSU_2018_MEANS',
    'select_cluster_Otsu2018'
]
__all__ += ['SDS_SMITS1999']
