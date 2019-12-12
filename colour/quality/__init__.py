# -*- coding: utf-8 -*-

from __future__ import absolute_import

from .datasets import *  # noqa
from . import datasets
from .cri import CRI_Specification, colour_rendering_index
from .cqs import (CQS_Specification, COLOUR_QUALITY_SCALE_METHODS,
                  colour_quality_scale)
from .ssi import spectral_similarity_index

__all__ = []
__all__ += datasets.__all__
__all__ += ['CRI_Specification', 'colour_rendering_index']
__all__ += [
    'CQS_Specification', 'COLOUR_QUALITY_SCALE_METHODS', 'colour_quality_scale'
]
__all__ += ['spectral_similarity_index']
