# -*- coding: utf-8 -*-

from __future__ import absolute_import

from .cameras import RGB_SpectralSensitivities
from .displays import RGB_DisplayPrimaries
from .datasets import *  # noqa
from . import datasets
from .correction import (
    augmented_matrix_Cheung2004, polynomial_expansion_Finlayson2015,
    polynomial_expansion_Vandermonde, POLYNOMIAL_EXPANSION_METHODS,
    polynomial_expansion, colour_correction_matrix_Cheung2004,
    colour_correction_matrix_Finlayson2015,
    colour_correction_matrix_Vandermonde, COLOUR_CORRECTION_MATRIX_METHODS,
    colour_correction_matrix, colour_correction_Cheung2004,
    colour_correction_Finlayson2015, colour_correction_Vandermonde,
    COLOUR_CORRECTION_METHODS, colour_correction)

__all__ = []
__all__ += ['RGB_SpectralSensitivities']
__all__ += ['RGB_DisplayPrimaries']
__all__ += datasets.__all__
__all__ += [
    'augmented_matrix_Cheung2004', 'polynomial_expansion_Finlayson2015',
    'polynomial_expansion_Vandermonde', 'POLYNOMIAL_EXPANSION_METHODS',
    'polynomial_expansion', 'colour_correction_matrix_Cheung2004',
    'colour_correction_matrix_Finlayson2015',
    'colour_correction_matrix_Vandermonde', 'COLOUR_CORRECTION_MATRIX_METHODS',
    'colour_correction_matrix', 'colour_correction_Cheung2004',
    'colour_correction_Finlayson2015', 'colour_correction_Vandermonde',
    'COLOUR_CORRECTION_METHODS', 'colour_correction'
]
