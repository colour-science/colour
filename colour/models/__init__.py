#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

from .cie_xyy import (
    XYZ_to_xyY,
    xyY_to_XYZ,
    xy_to_xyY,
    xyY_to_xy,
    xy_to_XYZ,
    XYZ_to_xy)
from .rgb import RGB_Colourspace
from .derivation import (
    normalised_primary_matrix,
    chromatically_adapted_primaries,
    primaries_whitepoint,
    RGB_luminance_equation,
    RGB_luminance)
from .dataset import *  # noqa
from . import dataset
from .cie_lab import XYZ_to_Lab, Lab_to_XYZ, Lab_to_LCHab, LCHab_to_Lab
from .cie_luv import (
    XYZ_to_Luv,
    Luv_to_XYZ,
    Luv_to_uv,
    Luv_uv_to_xy,
    Luv_to_LCHuv,
    LCHuv_to_Luv)
from .cie_ucs import XYZ_to_UCS, UCS_to_XYZ, UCS_to_uv, UCS_uv_to_xy
from .cie_uvw import XYZ_to_UVW
from .ipt import XYZ_to_IPT, IPT_to_XYZ, IPT_hue_angle
from .log import LINEAR_TO_LOG_METHODS, LOG_TO_LINEAR_METHODS
from .log import linear_to_log, log_to_linear
from .log import (
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
from .rgb import XYZ_to_RGB, RGB_to_XYZ
from .rgb import RGB_to_RGB
from .common import XYZ_to_sRGB, sRGB_to_XYZ
from .aces_it import spectral_to_aces_relative_exposure_values

__all__ = ['XYZ_to_xyY',
           'xyY_to_XYZ',
           'xy_to_xyY',
           'xyY_to_xy',
           'xy_to_XYZ',
           'XYZ_to_xy']
__all__ += ['RGB_Colourspace']
__all__ += ['normalised_primary_matrix',
            'chromatically_adapted_primaries',
            'primaries_whitepoint',
            'RGB_luminance_equation',
            'RGB_luminance']
__all__ += dataset.__all__
__all__ += ['XYZ_to_Lab', 'Lab_to_XYZ', 'Lab_to_LCHab', 'LCHab_to_Lab']
__all__ += ['XYZ_to_Luv',
            'Luv_to_XYZ',
            'Luv_to_uv',
            'Luv_uv_to_xy',
            'Luv_to_LCHuv',
            'LCHuv_to_Luv']
__all__ += ['XYZ_to_UCS', 'UCS_to_XYZ', 'UCS_to_uv', 'UCS_uv_to_xy']
__all__ += ['XYZ_to_UVW']
__all__ += ['XYZ_to_IPT', 'IPT_to_XYZ', 'IPT_hue_angle']
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
__all__ += ['XYZ_to_RGB', 'RGB_to_XYZ']
__all__ += ['RGB_to_RGB']
__all__ += ['XYZ_to_sRGB', 'sRGB_to_XYZ']
__all__ += ['spectral_to_aces_relative_exposure_values']
