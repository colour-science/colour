# -*- coding: utf-8 -*-
"""
Common Colour Models Utilities
==============================

Defines various colour models common utilities:

-   :func:`colour.models.Jab_to_JCh`
-   :func:`colour.models.JCh_to_Jab`

References
----------
-   :cite:`CIETC1-482004m` : CIE TC 1-48. (2004). CIE 1976 uniform colour
    spaces. In CIE 015:2004 Colorimetry, 3rd Edition (p. 24).
    ISBN:978-3-901-90633-6
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.algebra import cartesian_to_polar, polar_to_cartesian
from colour.utilities import (domain_range_scale, from_range_degrees,
                              to_domain_degrees, tsplit, tstack)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'Jab_to_JCh', 'JCh_to_Jab', 'COLOURSPACE_MODELS',
    'COLOURSPACE_MODELS_AXIS_LABELS', 'XYZ_to_colourspace_model'
]


def Jab_to_JCh(Jab):
    """
    Converts from *Jab** colour representation to *JCh* colour representation.

    This definition is used to perform conversion from *CIE L\\*a\\*b\\**
    colourspace to *CIE L\\*C\\*Hab* colourspace and for other similar
    conversions. It implements a generic transformation from Lightness
    :math:`J`, :math:`a` and :math:`b` opponent colour dimensions to the
    correlates of Lightness :math:`J`, chroma :math:`C` and hue angle
    :math:`h`.

    Parameters
    ----------
    Jab : array_like
        *Jab** colour representation array.

    Returns
    -------
    ndarray
        *JCh* colour representation array.

    Notes
    -----

    +------------+-----------------------+-----------------+
    | **Domain** | **Scale - Reference** | **Scale - 1**   |
    +============+=======================+=================+
    | ``Jab``    | ``J`` : [0, 100]      | ``J`` : [0, 1]  |
    |            |                       |                 |
    |            | ``a`` : [-100, 100]   | ``a`` : [-1, 1] |
    |            |                       |                 |
    |            | ``b`` : [-100, 100]   | ``b`` : [-1, 1] |
    +------------+-----------------------+-----------------+

    +------------+-----------------------+-----------------+
    | **Range**  | **Scale - Reference** | **Scale - 1**   |
    +============+=======================+=================+
    | ``JCh``    | ``J``  : [0, 100]     | ``J`` : [0, 1]  |
    |            |                       |                 |
    |            | ``C``  : [0, 100]     | ``C`` : [0, 1]  |
    |            |                       |                 |
    |            | ``h`` : [0, 360]      | ``h`` : [0, 1]  |
    +------------+-----------------------+-----------------+

    References
    ----------
    :cite:`CIETC1-482004m`

    Examples
    --------
    >>> Jab = np.array([41.52787529, 52.63858304, 26.92317922])
    >>> Jab_to_JCh(Jab)  # doctest: +ELLIPSIS
    array([ 41.5278752...,  59.1242590...,  27.0884878...])
    """

    L, a, b = tsplit(Jab)

    C, H = tsplit(cartesian_to_polar(tstack([a, b])))

    JCh = tstack([L, C, from_range_degrees(np.degrees(H) % 360)])

    return JCh


def JCh_to_Jab(JCh):
    """
    Converts from *JCh* colour representation to *Jab** colour representation.

    This definition is used to perform conversion from *CIE L\\*C\\*Hab*
    colourspace to *CIE L\\*a\\*b\\** colourspace and for other similar
    conversions. It implements a generic transformation from the correlates of
    Lightness :math:`J`, chroma :math:`C` and hue angle :math:`h` to Lightness
    :math:`J`, :math:`a` and :math:`b` opponent colour dimensions.

    Parameters
    ----------
    JCh : array_like
        *JCh* colour representation array.

    Returns
    -------
    ndarray
        *Jab** colour representation array.

    Notes
    -----

    +-------------+-----------------------+-----------------+
    | **Domain**  | **Scale - Reference** | **Scale - 1**   |
    +=============+=======================+=================+
    | ``JCh``     | ``J``  : [0, 100]     | ``J``  : [0, 1] |
    |             |                       |                 |
    |             | ``C``  : [0, 100]     | ``C``  : [0, 1] |
    |             |                       |                 |
    |             | ``h`` : [0, 360]      | ``h`` : [0, 1]  |
    +-------------+-----------------------+-----------------+

    +-------------+-----------------------+-----------------+
    | **Range**   | **Scale - Reference** | **Scale - 1**   |
    +=============+=======================+=================+
    | ``Jab``     | ``J`` : [0, 100]      | ``J`` : [0, 1]  |
    |             |                       |                 |
    |             | ``a`` : [-100, 100]   | ``a`` : [-1, 1] |
    |             |                       |                 |
    |             | ``b`` : [-100, 100]   | ``b`` : [-1, 1] |
    +-------------+-----------------------+-----------------+

    References
    ----------
    :cite:`CIETC1-482004m`

    Examples
    --------
    >>> JCh = np.array([41.52787529, 59.12425901, 27.08848784])
    >>> JCh_to_Jab(JCh)  # doctest: +ELLIPSIS
    array([ 41.5278752...,  52.6385830...,  26.9231792...])
    """

    L, C, H = tsplit(JCh)

    a, b = tsplit(
        polar_to_cartesian(tstack([C, np.radians(to_domain_degrees(H))])))

    Jab = tstack([L, a, b])

    return Jab


COLOURSPACE_MODELS = ('CIE XYZ', 'CIE xyY', 'CIE Lab', 'CIE LCHab', 'CIE Luv',
                      'CIE Luv uv', 'CIE LCHuv', 'CIE UCS', 'CIE UCS uv',
                      'CIE UVW', 'DIN 99', 'Hunter Lab', 'Hunter Rdab', 'IPT',
                      'JzAzBz', 'OSA UCS', 'hdr-CIELAB', 'hdr-IPT')

COLOURSPACE_MODELS_AXIS_LABELS = {
    'CIE XYZ': ('X', 'Y', 'Z'),
    'CIE xyY': ('x', 'y', 'Y'),
    'CIE Lab': ('$L^*$', '$a^*$', '$b^*$'),
    'CIE LCHab': ('$L^*$', 'CH', 'ab'),
    'CIE Luv': ('$L^*$', '$u^\\prime$', '$v^\\prime$'),
    'CIE Luv uv': ('$u^\\prime$', '$v^\\prime$'),
    'CIE LCHuv': ('$L^*$', 'CH', 'uv'),
    'CIE UCS': ('U', 'V', 'W'),
    'CIE UCS uv': ('u', 'v'),
    'CIE UVW': ('U', 'V', 'W'),
    'DIN 99': ('L99', 'a99', 'b99'),
    'Hunter Lab': ('$L^*$', '$a^*$', '$b^*$'),
    'Hunter Rdab': ('Rd', 'a', 'b'),
    'IPT': ('I', 'P', 'T'),
    'JzAzBz': ('$J_z$', '$A_z$', '$B_z$'),
    'OSA UCS': ('L', 'j', 'g'),
    'hdr-CIELAB': ('L hdr', 'a hdr', 'b hdr'),
    'hdr-IPT': ('I hdr', 'P hdr', 'T hdr'),
}
"""
Colourspace models labels mapping.

COLOURSPACE_MODELS_AXIS_LABELS : dict
    **{'CIE XYZ', 'CIE xyY', 'CIE Lab', 'CIE LCHab, 'CIE Luv', 'CIE Luv uv',
    'CIE LCHuv', 'CIE UCS', 'CIE UCS uv', 'CIE UVW', 'DIN 99', 'Hunter Lab',
    'Hunter Rdab','IPT', 'JzAzBz', 'OSA UCS', 'hdr-CIELAB', 'hdr-IPT'}**
"""


def XYZ_to_colourspace_model(XYZ, illuminant, model, **kwargs):
    """
    Converts from *CIE XYZ* tristimulus values to given colourspace model.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values.
    illuminant : array_like
        Reference *illuminant* *CIE xy* chromaticity coordinates or *CIE xyY*
        colourspace array.
    model : unicode
        **{'CIE XYZ', 'CIE xyY', 'CIE xy', 'CIE Lab', 'CIE LCHab', 'CIE Luv',
        'CIE Luv uv', 'CIE LCHuv', 'CIE UCS', 'CIE UCS uv', 'CIE UVW',
        'DIN 99', 'Hunter Lab', 'Hunter Rdab', 'IPT', 'JzAzBz, 'OSA UCS',
        'hdr-CIELAB', 'hdr-IPT'}**,
        Colourspace model to convert the *CIE XYZ* tristimulus values to.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        Keywords arguments.

    Returns
    -------
    ndarray
        Colourspace model values.

    Warnings
    --------
    This definition is is deprecated and will be removed in a future release.
    :func:`colour.convert` definition should be used instead.

    Examples
    --------
    >>> import numpy as np
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> W = np.array([0.31270, 0.32900])
    >>> XYZ_to_colourspace_model(  # doctest: +ELLIPSIS
    ... XYZ, W, 'CIE XYZ')
    array([ 0.2065400...,  0.1219722...,  0.0513695...])
    >>> XYZ_to_colourspace_model(  # doctest: +ELLIPSIS
    ... XYZ, W, 'CIE xyY')
    array([ 0.5436955...,  0.3210794...,  0.1219722...])
    >>> XYZ_to_colourspace_model(  # doctest: +ELLIPSIS
    ... XYZ, W, 'CIE xy')
    array([ 0.5436955...,  0.3210794...])
    >>> XYZ_to_colourspace_model(  # doctest: +ELLIPSIS
    ... XYZ, W, 'CIE Lab')
    array([ 0.4152787...,  0.5263858...,  0.2692317...])
    >>> XYZ_to_colourspace_model(  # doctest: +ELLIPSIS
    ... XYZ, W, 'CIE LCHab')
    array([ 0.4152787...,  0.5912425...,  0.0752458...])
    >>> XYZ_to_colourspace_model(  # doctest: +ELLIPSIS
    ... XYZ, W, 'CIE Luv')
    array([ 0.4152787...,  0.9683626...,  0.1775210...])
    >>> XYZ_to_colourspace_model(  # doctest: +ELLIPSIS
    ... XYZ, W, 'CIE Luv uv')
    array([ 0.3772021...,  0.5012026...])
    >>> XYZ_to_colourspace_model(  # doctest: +ELLIPSIS
    ... XYZ, W, 'CIE LCHuv')
    array([ 0.4152787...,  0.9844997...,  0.0288560...])
    >>> XYZ_to_colourspace_model(  # doctest: +ELLIPSIS
    ... XYZ, W, 'CIE UCS')
    array([ 0.1376933...,  0.1219722...,  0.1053731...])
    >>> XYZ_to_colourspace_model(  # doctest: +ELLIPSIS
    ... XYZ, W, 'CIE UCS uv')
    array([ 0.3772021...,  0.3341350...])
    >>> XYZ_to_colourspace_model(  # doctest: +ELLIPSIS
    ... XYZ, W, 'CIE UVW')
    array([ 0.9455035...,  0.1155536...,  0.4054757...])
    >>> XYZ_to_colourspace_model(  # doctest: +ELLIPSIS
    ... XYZ, W, 'DIN 99')
    array([ 0.5322822...,  0.2841634...,  0.0389839...])
    >>> XYZ_to_colourspace_model(  # doctest: +ELLIPSIS
    ... XYZ, W, 'Hunter Lab')
    array([ 0.3492452...,  0.4703302...,  0.1439330...])
    >>> XYZ_to_colourspace_model(  # doctest: +ELLIPSIS
    ... XYZ, W, 'Hunter Rdab')
    array([ 0.1219722...,  0.5709032...,  0.1747109...])
    >>> XYZ_to_colourspace_model(  # doctest: +ELLIPSIS
    ... XYZ, W, 'IPT')
    array([ 0.3842619...,  0.3848730...,  0.1888683...])
    >>> XYZ_to_colourspace_model(  # doctest: +ELLIPSIS
    ... XYZ, W, 'JzAzBz')
    array([ 0.0053504...,  0.0092430...,  0.0052600...])
    >>> XYZ_to_colourspace_model(  # doctest: +ELLIPSIS
    ... XYZ, W, 'OSA UCS')
    array([-0.0300499...,  0.0299713..., -0.0966784...])
    >>> XYZ_to_colourspace_model(  # doctest: +ELLIPSIS
    ... XYZ, W, 'hdr-CIELAB')
    array([ 0.5187002...,  0.6047633...,  0.3214551...])
    >>> XYZ_to_colourspace_model(  # doctest: +ELLIPSIS
    ... XYZ, W, 'hdr-IPT')
    array([ 0.4839376...,  0.4244990...,  0.2201954...])
    >>> try:
    ...     XYZ_to_colourspace_model(XYZ, W, 'Undefined')
    ... except ValueError as error:
    ...     print(error)
    "Undefined" not found in colourspace models: "CIE XYZ, CIE xyY, CIE Lab, \
CIE LCHab, CIE Luv, CIE Luv uv, CIE LCHuv, CIE UCS, CIE UCS uv, CIE UVW, \
DIN 99, Hunter Lab, Hunter Rdab, IPT, JzAzBz, OSA UCS, hdr-CIELAB, hdr-IPT".
    """

    from colour.models import (
        Lab_to_DIN99, Lab_to_LCHab, Luv_to_LCHuv, Luv_to_uv, UCS_to_uv,
        XYZ_to_IPT, XYZ_to_Hunter_Lab, XYZ_to_Hunter_Rdab, XYZ_to_Lab,
        XYZ_to_Luv, XYZ_to_OSA_UCS, XYZ_to_UCS, XYZ_to_UVW, XYZ_to_hdr_CIELab,
        XYZ_to_hdr_IPT, XYZ_to_JzAzBz, XYZ_to_xy, XYZ_to_xyY, xy_to_XYZ)

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
        elif model == 'DIN 99':
            values = Lab_to_DIN99(XYZ_to_Lab(XYZ, illuminant))
        elif model == 'Hunter Lab':
            values = XYZ_to_Hunter_Lab(XYZ, xy_to_XYZ(illuminant))
        elif model == 'Hunter Rdab':
            values = XYZ_to_Hunter_Rdab(XYZ, xy_to_XYZ(illuminant))
        elif model == 'IPT':
            values = XYZ_to_IPT(XYZ)
        elif model == 'JzAzBz':
            values = XYZ_to_JzAzBz(XYZ)
        elif model == 'OSA UCS':
            values = XYZ_to_OSA_UCS(XYZ)
        elif model == 'hdr-CIELAB':
            values = XYZ_to_hdr_CIELab(XYZ, illuminant, **kwargs)
        elif model == 'hdr-IPT':
            values = XYZ_to_hdr_IPT(XYZ, **kwargs)

        if values is None:
            raise ValueError(
                '"{0}" not found in colourspace models: "{1}".'.format(
                    model, ', '.join(COLOURSPACE_MODELS)))

        return values
