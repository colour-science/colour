# -*- coding: utf-8 -*-
"""
Common Colour Models Utilities
==============================

Defines various colour models common utilities:

-   :func:`colour.models.Jab_to_JCh`
-   :func:`colour.models.JCh_to_Jab`
-   :attr:`colour.COLOURSPACE_MODELS`

References
----------
-   :cite:`CIETC1-482004m` : CIE TC 1-48. (2004). CIE 1976 uniform colour
    spaces. In CIE 015:2004 Colorimetry, 3rd Edition (p. 24).
    ISBN:978-3-901906-33-6
"""

import numpy as np

from colour.algebra import cartesian_to_polar, polar_to_cartesian
from colour.utilities import (from_range_degrees, to_domain_degrees, tsplit,
                              tstack)
from colour.utilities.documentation import (DocstringTuple,
                                            is_documentation_building)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'Jab_to_JCh', 'JCh_to_Jab', 'COLOURSPACE_MODELS',
    'COLOURSPACE_MODELS_AXIS_LABELS'
]


def Jab_to_JCh(Jab):
    """
    Converts from *Jab** colour representation to *JCh* colour representation.

    This definition is used to perform conversion from *CIE L\\*a\\*b\\**
    colourspace to *CIE L\\*C\\*Hab* colourspace and for other similar
    conversions. It implements a generic transformation from *Lightness*
    :math:`J`, :math:`a` and :math:`b` opponent colour dimensions to the
    correlates of *Lightness* :math:`J`, chroma :math:`C` and hue angle
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
    *Lightness* :math:`J`, chroma :math:`C` and hue angle :math:`h` to
    *Lightness* :math:`J`, :math:`a` and :math:`b` opponent colour dimensions.

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
                      'CIE UVW', 'DIN 99', 'Hunter Lab', 'Hunter Rdab',
                      'ICtCp', 'IPT', 'IgPgTg', 'JzAzBz', 'OSA UCS', 'Oklab',
                      'hdr-CIELAB', 'hdr-IPT')

if is_documentation_building():  # pragma: no cover
    COLOURSPACE_MODELS = DocstringTuple(COLOURSPACE_MODELS)
    COLOURSPACE_MODELS.__doc__ = """
Colourspace models supporting a direct conversion to *CIE XYZ* tristimulus
values.

COLOURSPACE_MODELS : Tuple
"""

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
    'ICtCp': ('$I$', '$C_T$', '$C_P$'),
    'IPT': ('I', 'P', 'T'),
    'IgPgTg': ('$I_G$', '$P_G$', '$T_G$'),
    'JzAzBz': ('$J_z$', '$A_z$', '$B_z$'),
    'OSA UCS': ('L', 'j', 'g'),
    'Oklab': ('$L$', '$a$', '$b$'),
    'hdr-CIELAB': ('L hdr', 'a hdr', 'b hdr'),
    'hdr-IPT': ('I hdr', 'P hdr', 'T hdr'),
}
"""
Colourspace models labels mapping.

COLOURSPACE_MODELS_AXIS_LABELS : dict
    **{'CIE XYZ', 'CIE xyY', 'CIE Lab', 'CIE LCHab, 'CIE Luv', 'CIE Luv uv',
    'CIE LCHuv', 'CIE UCS', 'CIE UCS uv', 'CIE UVW', 'DIN 99', 'Hunter Lab',
    'Hunter Rdab', 'ICtCp', 'IPT', 'IgPgTg','JzAzBz', 'OSA UCS', 'Oklab',
    'hdr-CIELAB', 'hdr-IPT'}**
"""
