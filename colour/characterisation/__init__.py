# -*- coding: utf-8 -*-

from __future__ import absolute_import

from .cameras import RGB_SpectralSensitivities
from .displays import RGB_DisplayPrimaries
from .datasets import *  # noqa
from . import datasets
from .aces_it import (
    sd_to_aces_relative_exposure_values, read_training_data_rawtoaces_v1,
    generate_illuminants_rawtoaces_v1, white_balance_multipliers,
    best_illuminant, normalise_illuminant, training_data_sds_to_RGB,
    training_data_sds_to_XYZ, optimisation_factory_rawtoaces_v1,
    optimisation_factory_JzAzBz, idt_matrix)
from .correction import (
    augmented_matrix_Cheung2004, polynomial_expansion_Finlayson2015,
    polynomial_expansion_Vandermonde, POLYNOMIAL_EXPANSION_METHODS,
    polynomial_expansion, colour_correction_matrix_Cheung2004,
    colour_correction_matrix_Finlayson2015,
    colour_correction_matrix_Vandermonde, COLOUR_CORRECTION_MATRIX_METHODS,
    colour_correction_matrix, colour_correction_Cheung2004,
    colour_correction_Finlayson2015, colour_correction_Vandermonde,
    COLOUR_CORRECTION_METHODS, colour_correction)

__all__ = ['RGB_SpectralSensitivities']
__all__ += ['RGB_DisplayPrimaries']
__all__ += datasets.__all__
__all__ += [
    'sd_to_aces_relative_exposure_values', 'read_training_data_rawtoaces_v1',
    'generate_illuminants_rawtoaces_v1', 'white_balance_multipliers',
    'best_illuminant', 'normalise_illuminant', 'training_data_sds_to_RGB',
    'training_data_sds_to_XYZ', 'optimisation_factory_rawtoaces_v1',
    'optimisation_factory_JzAzBz', 'idt_matrix'
]
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
