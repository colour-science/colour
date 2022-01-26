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

from __future__ import annotations

import numpy as np

from colour.algebra import cartesian_to_polar, polar_to_cartesian
from colour.hints import ArrayLike, NDArray, Tuple
from colour.utilities import (
    CaseInsensitiveMapping,
    attest,
    from_range_degrees,
    to_domain_degrees,
    tsplit,
    tstack,
)
from colour.utilities.documentation import (
    DocstringTuple,
    is_documentation_building,
)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'Jab_to_JCh',
    'JCh_to_Jab',
    'COLOURSPACE_MODELS',
    'COLOURSPACE_MODELS_AXIS_LABELS',
    'COLOURSPACE_MODELS_DOMAIN_RANGE_SCALE_1_TO_REFERENCE',
]


def Jab_to_JCh(Jab: ArrayLike) -> NDArray:
    """
    Converts from *Jab* colour representation to *JCh* colour representation.

    This definition is used to perform conversion from *CIE L\\*a\\*b\\**
    colourspace to *CIE L\\*C\\*Hab* colourspace and for other similar
    conversions. It implements a generic transformation from *Lightness*
    :math:`J`, :math:`a` and :math:`b` opponent colour dimensions to the
    correlates of *Lightness* :math:`J`, chroma :math:`C` and hue angle
    :math:`h`.

    Parameters
    ----------
    Jab
        *Jab* colour representation array.

    Returns
    -------
    :class:`numpy.ndarray`
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


def JCh_to_Jab(JCh: ArrayLike) -> NDArray:
    """
    Converts from *JCh* colour representation to *Jab* colour representation.

    This definition is used to perform conversion from *CIE L\\*C\\*Hab*
    colourspace to *CIE L\\*a\\*b\\** colourspace and for other similar
    conversions. It implements a generic transformation from the correlates of
    *Lightness* :math:`J`, chroma :math:`C` and hue angle :math:`h` to
    *Lightness* :math:`J`, :math:`a` and :math:`b` opponent colour dimensions.

    Parameters
    ----------
    JCh
        *JCh* colour representation array.

    Returns
    -------
    :class:`numpy.ndarray`
        *Jab* colour representation array.

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


COLOURSPACE_MODELS: Tuple = ('CAM02LCD', 'CAM02SCD', 'CAM02UCS', 'CAM16LCD',
                             'CAM16SCD', 'CAM16UCS', 'CIE XYZ', 'CIE xyY',
                             'CIE Lab', 'CIE Luv', 'CIE UCS', 'CIE UVW',
                             'DIN99', 'Hunter Lab', 'Hunter Rdab', 'ICaCb',
                             'ICtCp', 'IPT', 'IgPgTg', 'Jzazbz', 'OSA UCS',
                             'Oklab', 'hdr-CIELAB', 'hdr-IPT')
if is_documentation_building():  # pragma: no cover
    COLOURSPACE_MODELS = DocstringTuple(COLOURSPACE_MODELS)
    COLOURSPACE_MODELS.__doc__ = """
Colourspace models supporting a direct conversion to *CIE XYZ* tristimulus
values.
"""

COLOURSPACE_MODELS_AXIS_LABELS: CaseInsensitiveMapping = (
    CaseInsensitiveMapping({
        'CAM02LCD': ('$J^\\prime$', '$a^\\prime$', '$b^\\prime$'),
        'CAM02SCD': ('$J^\\prime$', '$a^\\prime$', '$b^\\prime$'),
        'CAM02UCS': ('$J^\\prime$', '$a^\\prime$', '$b^\\prime$'),
        'CAM16LCD': ('$J^\\prime$', '$a^\\prime$', '$b^\\prime$'),
        'CAM16SCD': ('$J^\\prime$', '$a^\\prime$', '$b^\\prime$'),
        'CAM16UCS': ('$J^\\prime$', '$a^\\prime$', '$b^\\prime$'),
        'CIE XYZ': ('X', 'Y', 'Z'),
        'CIE xyY': ('x', 'y', 'Y'),
        'CIE Lab': ('$L^*$', '$a^*$', '$b^*$'),
        'CIE Luv': ('$L^*$', '$u^\\prime$', '$v^\\prime$'),
        'CIE UCS': ('U', 'V', 'W'),
        'CIE UVW': ('U', 'V', 'W'),
        'DIN99': ('$L_{99}$', '$a_{99}$', '$b_{99}$'),
        'Hunter Lab': ('$L^*$', '$a^*$', '$b^*$'),
        'Hunter Rdab': ('Rd', 'a', 'b'),
        'ICaCb': ('$I$', '$C_a$', '$C_b$'),
        'ICtCp': ('$I$', '$C_T$', '$C_P$'),
        'IPT': ('I', 'P', 'T'),
        'IgPgTg': ('$I_G$', '$P_G$', '$T_G$'),
        'Jzazbz': ('$J_z$', '$a_z$', '$b_z$'),
        'OSA UCS': ('L', 'j', 'g'),
        'Oklab': ('$L$', '$a$', '$b$'),
        'hdr-CIELAB': ('L hdr', 'a hdr', 'b hdr'),
        'hdr-IPT': ('I hdr', 'P hdr', 'T hdr'),
    }))
"""
Colourspace models labels mapping.
"""

attest(COLOURSPACE_MODELS == tuple(COLOURSPACE_MODELS_AXIS_LABELS.keys()))

COLOURSPACE_MODELS_DOMAIN_RANGE_SCALE_1_TO_REFERENCE: (
    CaseInsensitiveMapping) = CaseInsensitiveMapping({
        'CAM02LCD': np.array([100, 100, 100]),
        'CAM02SCD': np.array([100, 100, 100]),
        'CAM02UCS': np.array([100, 100, 100]),
        'CAM16LCD': np.array([100, 100, 100]),
        'CAM16SCD': np.array([100, 100, 100]),
        'CAM16UCS': np.array([100, 100, 100]),
        'CIE XYZ': np.array([1, 1, 1]),
        'CIE xyY': np.array([1, 1, 1]),
        'CIE Lab': np.array([100, 100, 100]),
        'CIE Luv': np.array([100, 100, 100]),
        'CIE UCS': np.array([1, 1, 1]),
        'CIE UVW': np.array([100, 100, 100]),
        'DIN99': np.array([100, 100, 100]),
        'Hunter Lab': np.array([100, 100, 100]),
        'Hunter Rdab': np.array([100, 100, 100]),
        'ICaCb': np.array([1, 1, 1]),
        'ICtCp': np.array([1, 1, 1]),
        'IPT': np.array([1, 1, 1]),
        'IgPgTg': np.array([1, 1, 1]),
        'Jzazbz': np.array([1, 1, 1]),
        'OSA UCS': np.array([100, 100, 100]),
        'Oklab': np.array([1, 1, 1]),
        'hdr-CIELAB': np.array([100, 100, 100]),
        'hdr-IPT': np.array([100, 100, 100]),
    })
"""
Colourspace models domain-range scale **'1'** to **'Reference'** mapping.
"""
