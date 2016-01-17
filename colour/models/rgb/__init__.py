#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

from .rgb_colourspace import RGB_Colourspace
from .rgb_colourspace import XYZ_to_RGB, RGB_to_XYZ
from .rgb_colourspace import RGB_to_RGB
from .derivation import (
    normalised_primary_matrix,
    chromatically_adapted_primaries,
    primaries_whitepoint,
    RGB_luminance_equation,
    RGB_luminance)
from .dataset import *  # noqa
from . import dataset
from .conversion_functions import (
    LINEAR_TO_LOG_METHODS,
    LOG_TO_LINEAR_METHODS)
from .conversion_functions import linear_to_log, log_to_linear
from .conversion_functions import (
    linear_to_cineon,
    cineon_to_linear,
    linear_to_panalog,
    panalog_to_linear,
    linear_to_red_log_film,
    red_log_film_to_linear,
    linear_to_viper_log,
    viper_log_to_linear,
    linear_to_pivoted_log,
    pivoted_log_to_linear,
    linear_to_c_log,
    c_log_to_linear,
    linear_to_aces_cc,
    aces_cc_to_linear,
    linear_to_alexa_log_c,
    alexa_log_c_to_linear,
    linear_to_dci_p3_log,
    dci_p3_log_to_linear,
    linear_to_s_log,
    s_log_to_linear,
    linear_to_s_log2,
    s_log2_to_linear,
    linear_to_s_log3,
    s_log3_to_linear,
    linear_to_v_log,
    v_log_to_linear)
from .common import XYZ_to_sRGB, sRGB_to_XYZ
from .aces_it import spectral_to_aces_relative_exposure_values

__all__ = ['RGB_Colourspace']
__all__ += ['XYZ_to_RGB', 'RGB_to_XYZ']
__all__ += ['RGB_to_RGB']
__all__ += ['normalised_primary_matrix',
            'chromatically_adapted_primaries',
            'primaries_whitepoint',
            'RGB_luminance_equation',
            'RGB_luminance']
__all__ += dataset.__all__
__all__ += ['LINEAR_TO_LOG_METHODS', 'LOG_TO_LINEAR_METHODS']
__all__ += ['linear_to_log', 'log_to_linear']
__all__ += ['linear_to_cineon',
            'cineon_to_linear',
            'linear_to_panalog',
            'panalog_to_linear',
            'linear_to_red_log_film',
            'red_log_film_to_linear',
            'linear_to_viper_log',
            'viper_log_to_linear',
            'linear_to_pivoted_log',
            'pivoted_log_to_linear',
            'linear_to_c_log',
            'c_log_to_linear',
            'linear_to_aces_cc',
            'aces_cc_to_linear',
            'linear_to_alexa_log_c',
            'alexa_log_c_to_linear',
            'linear_to_s_log',
            'linear_to_dci_p3_log',
            'dci_p3_log_to_linear',
            's_log_to_linear',
            'linear_to_s_log2',
            's_log2_to_linear',
            'linear_to_s_log3',
            's_log3_to_linear',
            'linear_to_v_log',
            'v_log_to_linear']
__all__ += ['XYZ_to_sRGB', 'sRGB_to_XYZ']
__all__ += ['spectral_to_aces_relative_exposure_values']
