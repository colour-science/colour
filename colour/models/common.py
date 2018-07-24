# -*- coding: utf-8 -*-
"""
Common Colour Models Utilities
==============================

Defines various colour models common utilities.
"""

from __future__ import division, unicode_literals

from colour.models import (Lab_to_LCHab, Luv_to_LCHuv, Luv_to_uv, UCS_to_uv,
                           XYZ_to_IPT, XYZ_to_Hunter_Lab, XYZ_to_Hunter_Rdab,
                           XYZ_to_Lab, XYZ_to_Luv, XYZ_to_UCS, XYZ_to_UVW,
                           XYZ_to_xy, XYZ_to_xyY, xy_to_XYZ)
from colour.utilities import domain_range_scale

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'COLOURSPACE_MODELS', 'COLOURSPACE_MODELS_LABELS',
    'XYZ_to_colourspace_model'
]

COLOURSPACE_MODELS = ('CIE XYZ', 'CIE xyY', 'CIE Lab', 'CIE LCHab', 'CIE Luv',
                      'CIE Luv uv', 'CIE LCHuv', 'CIE UCS', 'CIE UCS uv',
                      'CIE UVW', 'IPT', 'Hunter Lab', 'Hunter Rdab')

COLOURSPACE_MODELS_LABELS = {
    'CIE XYZ': ('X', 'Y', 'Z'),
    'CIE xyY': ('x', 'y', 'Y'),
    'CIE Lab': ('$a^*$', '$b^*$', '$L^*$'),
    'CIE LCHab': ('CH', 'ab', '$L^*$'),
    'CIE Luv': ('$u^\prime$', '$v^\prime$', '$L^*$'),
    'CIE Luv uv': ('$u^\prime$', '$v^\prime$'),
    'CIE LCHuv': ('CH', 'uv', '$L^*$'),
    'CIE UCS': ('U', 'V', 'W'),
    'CIE UCS uv': ('$u$', '$v$'),
    'CIE UVW': ('U', 'V', 'W'),
    'IPT': ('P', 'T', 'I'),
    'Hunter Lab': ('$a^*$', '$b^*$', '$L^*$'),
    'Hunter Rdab': ('$a$', '$b$', '$Rd$')
}
"""
Colourspace models labels mapping.

COLOURSPACE_MODELS_LABELS : dict
    **{'CIE XYZ', 'CIE xyY', 'CIE Lab', 'CIE LCHab, 'CIE Luv', 'CIE Luv uv',
    'CIE LCHuv', 'CIE UCS', 'CIE UCS uv', 'CIE UVW', 'IPT', 'Hunter Lab',
    'Hunter Rdab'}**
"""


def XYZ_to_colourspace_model(XYZ, illuminant, model):
    """
    Converts from *CIE XYZ* tristimulus values to given colourspace model.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values.
    illuminant : array_like
        *CIE XYZ* tristimulus values *illuminant* *xy* chromaticity
        coordinates.
    model : unicode
        **{'CIE XYZ', 'CIE xyY', 'CIE xy', 'CIE Lab', 'CIE LCHab', 'CIE Luv',
        'CIE Luv uv', 'CIE LCHuv', 'CIE UCS', 'CIE UCS uv', 'CIE UVW', 'IPT',
        'Hunter Lab', 'Hunter Rdab'}**,
        Colourspace model to convert the *CIE XYZ* tristimulus values to.

    Returns
    -------
    ndarray
        Colourspace model values.

    Examples
    --------
    >>> import numpy as np
    >>> XYZ = np.array([0.07049534, 0.10080000, 0.09558313])
    >>> W = np.array([0.34570, 0.35850])
    >>> XYZ_to_colourspace_model(  # doctest: +ELLIPSIS
    ... XYZ, W, 'CIE XYZ')
    array([ 0.0704953...,  0.1008    ,  0.0955831...])
    >>> XYZ_to_colourspace_model(  # doctest: +ELLIPSIS
    ... XYZ, W, 'CIE xyY')
    array([ 0.2641477...,  0.3777000...,  0.1008    ])
    >>> XYZ_to_colourspace_model(  # doctest: +ELLIPSIS
    ... XYZ, W, 'CIE xy')
    array([ 0.2641477...,  0.3777000...])
    >>> XYZ_to_colourspace_model(  # doctest: +ELLIPSIS
    ... XYZ, W, 'CIE Lab')
    array([ 0.3798562..., -0.2362907..., -0.0441746...])
    >>> XYZ_to_colourspace_model(  # doctest: +ELLIPSIS
    ... XYZ, W, 'CIE LCHab')
    array([ 0.3798562...,  0.2403845...,  0.5294145...])
    >>> XYZ_to_colourspace_model(  # doctest: +ELLIPSIS
    ... XYZ, W, 'CIE Luv')
    array([ 0.3798562..., -0.2880219..., -0.0135800...])
    >>> XYZ_to_colourspace_model(  # doctest: +ELLIPSIS
    ... XYZ, W, 'CIE Luv uv')
    array([ 0.1508531...,  0.4853297...])
    >>> XYZ_to_colourspace_model(  # doctest: +ELLIPSIS
    ... XYZ, W, 'CIE LCHuv')
    array([ 0.37985629...,  0.2883419...,  0.5074985...])
    >>> XYZ_to_colourspace_model(  # doctest: +ELLIPSIS
    ... XYZ, W, 'CIE UCS uv')
    array([ 0.1508531...,  0.32355314...])
    >>> XYZ_to_colourspace_model(  # doctest: +ELLIPSIS
    ... XYZ, W, 'CIE UVW')
    array([-0.2805797..., -0.0088194...,  0.3700411...])
    >>> XYZ_to_colourspace_model(  # doctest: +ELLIPSIS
    ... XYZ, W, 'IPT')
    array([ 0.3657112..., -0.1111479...,  0.0159474...])
    >>> XYZ_to_colourspace_model(  # doctest: +ELLIPSIS
    ... XYZ, W, 'Hunter Lab')
    array([ 0.3174901..., -0.1513517..., -0.0277096...])
    >>> XYZ_to_colourspace_model(  # doctest: +ELLIPSIS
    ... XYZ, W, 'Hunter Rdab')
    array([ 0.1008..., -0.1870192..., -0.0342396...])
    """

    with domain_range_scale(1):
        values = None
        if model == 'CIE XYZ':
            values = XYZ
        elif model == 'CIE xyY':
            values = XYZ_to_xyY(XYZ, illuminant)
        elif model == 'CIE xy':
            values = XYZ_to_xy(XYZ, illuminant)
        elif model == 'CIE Lab':
            values = XYZ_to_Lab(XYZ, illuminant)
        elif model == 'CIE LCHab':
            values = Lab_to_LCHab(XYZ_to_Lab(XYZ, illuminant))
        elif model == 'CIE Luv':
            values = XYZ_to_Luv(XYZ, illuminant)
        elif model == 'CIE Luv uv':
            values = Luv_to_uv(XYZ_to_Luv(XYZ, illuminant), illuminant)
        elif model == 'CIE LCHuv':
            values = Luv_to_LCHuv(XYZ_to_Luv(XYZ, illuminant))
        elif model == 'CIE UCS':
            values = XYZ_to_UCS(XYZ)
        elif model == 'CIE UCS uv':
            values = UCS_to_uv(XYZ_to_UCS(XYZ))
        elif model == 'CIE UVW':
            values = XYZ_to_UVW(XYZ, illuminant)
        elif model == 'IPT':
            values = XYZ_to_IPT(XYZ)
        elif model == 'Hunter Lab':
            values = XYZ_to_Hunter_Lab(XYZ, xy_to_XYZ(illuminant))
        elif model == 'Hunter Rdab':
            values = XYZ_to_Hunter_Rdab(XYZ, xy_to_XYZ(illuminant))

        if values is None:
            raise ValueError(
                '"{0}" not found in colourspace models: "{1}".'.format(
                    model, ', '.join(COLOURSPACE_MODELS)))

        return values
