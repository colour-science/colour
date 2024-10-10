"""
Common Colour Models Utilities
==============================

Define various colour models common utilities:

-   :attr:`colour.COLOURSPACE_MODELS`
-   :func:`colour.models.Jab_to_JCh`
-   :func:`colour.models.JCh_to_Jab`
-   :func:`colour.models.XYZ_to_Iab`
-   :func:`colour.models.Iab_to_XYZ`

References
----------
-   :cite:`CIETC1-482004m` : CIE TC 1-48. (2004). CIE 1976 uniform colour
    spaces. In CIE 015:2004 Colorimetry, 3rd Edition (p. 24).
    ISBN:978-3-901906-33-6
"""

from __future__ import annotations

import numpy as np

from colour.algebra import cartesian_to_polar, polar_to_cartesian, vecmul
from colour.hints import ArrayLike, Callable, NDArrayFloat
from colour.utilities import (
    CanonicalMapping,
    attest,
    from_range_1,
    from_range_degrees,
    to_domain_1,
    to_domain_degrees,
    tsplit,
    tstack,
)
from colour.utilities.documentation import (
    DocstringTuple,
    is_documentation_building,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "COLOURSPACE_MODELS",
    "COLOURSPACE_MODELS_AXIS_LABELS",
    "COLOURSPACE_MODELS_DOMAIN_RANGE_SCALE_1_TO_REFERENCE",
    "Jab_to_JCh",
    "JCh_to_Jab",
    "XYZ_to_Iab",
    "Iab_to_XYZ",
]


COLOURSPACE_MODELS: tuple = (
    "CAM02LCD",
    "CAM02SCD",
    "CAM02UCS",
    "CAM16LCD",
    "CAM16SCD",
    "CAM16UCS",
    "CIE 1931",
    "CIE 1960 UCS",
    "CIE 1976 UCS",
    "CIE Lab",
    "CIE Luv",
    "CIE UCS",
    "CIE UVW",
    "CIE XYZ",
    "CIE xyY",
    "DIN99",
    "HCL",
    "HSL",
    "HSV",
    "Hunter Lab",
    "Hunter Rdab",
    "ICaCb",
    "ICtCp",
    "IHLS",
    "IPT Ragoo 2021",
    "IPT",
    "IgPgTg",
    "Jzazbz",
    "OSA UCS",
    "Oklab",
    "RGB",
    "YCbCr",
    "YCoCg",
    "Yrg",
    "hdr-CIELAB",
    "hdr-IPT",
)
if is_documentation_building():  # pragma: no cover
    COLOURSPACE_MODELS = DocstringTuple(COLOURSPACE_MODELS)
    COLOURSPACE_MODELS.__doc__ = """
Colourspace models supporting a direct conversion to *CIE XYZ* tristimulus
values.
"""

COLOURSPACE_MODELS_AXIS_LABELS: CanonicalMapping = CanonicalMapping(
    {
        "CAM02LCD": ("$J^'$", "$a^'$", "$b^'$"),
        "CAM02SCD": ("$J^'$", "$a^'$", "$b^'$"),
        "CAM02UCS": ("$J^'$", "$a^'$", "$b^'$"),
        "CAM16LCD": ("$J^'$", "$a^'$", "$b^'$"),
        "CAM16SCD": ("$J^'$", "$a^'$", "$b^'$"),
        "CAM16UCS": ("$J^'$", "$a^'$", "$b^'$"),
        "CIE 1931": ("x", "y", "Y"),
        "CIE 1960 UCS": ("$u^'$", "$v^'$", "$L^*$"),
        "CIE 1976 UCS": ("$u^'$", "$v^'$", "$L^*$"),
        "CIE Lab": ("$L^*$", "$a^*$", "$b^*$"),
        "CIE Luv": ("$L^*$", "$u^'$", "$v^'$"),
        "CIE UCS": ("U", "V", "W"),
        "CIE UVW": ("U", "V", "W"),
        "CIE XYZ": ("X", "Y", "Z"),
        "CIE xyY": ("x", "y", "Y"),
        "DIN99": ("$L_{99}$", "$a_{99}$", "$b_{99}$"),
        "HCL": ("H", "C", "L"),
        "HSL": ("H", "S", "L"),
        "HSV": ("H", "S", "V"),
        "Hunter Lab": ("$L^*$", "$a^*$", "$b^*$"),
        "Hunter Rdab": ("Rd", "a", "b"),
        "ICaCb": ("$I$", "$C_a$", "$C_b$"),
        "ICtCp": ("$I$", "$C_T$", "$C_P$"),
        "IHLS": ("H", "Y", "S"),
        "IPT Ragoo 2021": ("I", "P", "T"),
        "IPT": ("I", "P", "T"),
        "IgPgTg": ("$I_G$", "$P_G$", "$T_G$"),
        "Jzazbz": ("$J_z$", "$a_z$", "$b_z$"),
        "OSA UCS": ("L", "j", "g"),
        "Oklab": ("$L$", "$a$", "$b$"),
        "RGB": ("R", "G", "B"),
        "YCbCr": ("Y", "$C_b$", "$C_r$"),
        "YCoCg": ("Y", "$C_o$", "$C_g$"),
        "Yrg": ("Y", "r", "g"),
        "hdr-CIELAB": ("L hdr", "a hdr", "b hdr"),
        "hdr-IPT": ("I hdr", "P hdr", "T hdr"),
    }
)
"""Colourspace models labels mapping."""

attest(tuple(COLOURSPACE_MODELS_AXIS_LABELS.keys()) == COLOURSPACE_MODELS)

COLOURSPACE_MODELS_DOMAIN_RANGE_SCALE_1_TO_REFERENCE: CanonicalMapping = (
    CanonicalMapping(
        {
            "CAM02LCD": np.array([100, 100, 100]),
            "CAM02SCD": np.array([100, 100, 100]),
            "CAM02UCS": np.array([100, 100, 100]),
            "CAM16LCD": np.array([100, 100, 100]),
            "CAM16SCD": np.array([100, 100, 100]),
            "CAM16UCS": np.array([100, 100, 100]),
            "CIE 1931": np.array([1, 1, 1]),
            "CIE 1960 UCS": np.array([1, 1, 100]),
            "CIE 1976 UCS": np.array([1, 1, 100]),
            "CIE Lab": np.array([100, 100, 100]),
            "CIE Luv": np.array([100, 100, 100]),
            "CIE UCS": np.array([1, 1, 1]),
            "CIE UVW": np.array([100, 100, 100]),
            "CIE XYZ": np.array([1, 1, 1]),
            "CIE xyY": np.array([1, 1, 1]),
            "DIN99": np.array([100, 100, 100]),
            "HCL": np.array([1, 1, 1]),
            "HSL": np.array([1, 1, 1]),
            "HSV": np.array([1, 1, 1]),
            "Hunter Lab": np.array([100, 100, 100]),
            "Hunter Rdab": np.array([100, 100, 100]),
            "ICaCb": np.array([1, 1, 1]),
            "ICtCp": np.array([1, 1, 1]),
            "IHLS": np.array([1, 1, 1]),
            "IPT Ragoo 2021": np.array([1, 1, 1]),
            "IPT": np.array([1, 1, 1]),
            "IgPgTg": np.array([1, 1, 1]),
            "Jzazbz": np.array([1, 1, 1]),
            "OSA UCS": np.array([100, 100, 100]),
            "Oklab": np.array([1, 1, 1]),
            "RGB": np.array([1, 1, 1]),
            "YCbCr": np.array([1, 1, 1]),
            "YCoCg": np.array([1, 1, 1]),
            "Yrg": np.array([1, 1, 1]),
            "hdr-CIELAB": np.array([100, 100, 100]),
            "hdr-IPT": np.array([100, 100, 100]),
        }
    )
)
"""Colourspace models domain-range scale **'1'** to **'Reference'** mapping."""


def Jab_to_JCh(Jab: ArrayLike) -> NDArrayFloat:
    """
    Convert from *Jab* colour representation to *JCh* colour representation.

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


def JCh_to_Jab(JCh: ArrayLike) -> NDArrayFloat:
    """
    Convert from *JCh* colour representation to *Jab* colour representation.

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

    a, b = tsplit(polar_to_cartesian(tstack([C, np.radians(to_domain_degrees(H))])))

    Jab = tstack([L, a, b])

    return Jab


def XYZ_to_Iab(
    XYZ: ArrayLike,
    LMS_to_LMS_p_callable: Callable,
    matrix_XYZ_to_LMS: ArrayLike,
    matrix_LMS_p_to_Iab: ArrayLike,
) -> NDArrayFloat:
    """
    Convert from *CIE XYZ* tristimulus values to *IPT*-like :math:`Iab` colour
    representation.

    This definition is used to perform conversion from *CIE XYZ* tristimulus
    values to *IPT* colourspace and for other similar conversions. It
    implements a generic transformation from *CIE XYZ* tristimulus values to
    *Lightness* :math:`I`, :math:`a` and :math:`b` representing
    red-green dimension, i.e., the dimension lost by protanopes and
    the yellow-blue dimension, i.e., the dimension lost by tritanopes,
    respectively.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values.
    LMS_to_LMS_p_callable
        Callable applying the forward non-linearity to the :math:`LMS`
        colourspace array.
    matrix_XYZ_to_LMS
        Matrix converting from *CIE XYZ* tristimulus values to :math:`LMS`
        colourspace.
    matrix_LMS_p_to_Iab
        Matrix converting from non-linear :math:`LMS_p` colourspace to
        *IPT*-like :math:`Iab` colour representation.

    Returns
    -------
    :class:`numpy.ndarray`
        *IPT*-like :math:`Iab` colour representation.

    Notes
    -----
    +------------+-----------------------+-----------------+
    | **Domain** | **Scale - Reference** | **Scale - 1**   |
    +============+=======================+=================+
    | ``XYZ``    | [0, 1]                | [0, 1]          |
    +------------+-----------------------+-----------------+

    +------------+-----------------------+-----------------+
    | **Range**  | **Scale - Reference** | **Scale - 1**   |
    +============+=======================+=================+
    | ``Iab``    | ``I`` : [0, 1]        | ``I`` : [0, 1]  |
    |            |                       |                 |
    |            | ``a`` : [-1, 1]       | ``a`` : [-1, 1] |
    |            |                       |                 |
    |            | ``b`` : [-1, 1]       | ``b`` : [-1, 1] |
    +------------+-----------------------+-----------------+

    Examples
    --------
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> LMS_to_LMS_p = lambda x: x**0.43
    >>> M_XYZ_to_LMS = np.array(
    ...     [
    ...         [0.4002, 0.7075, -0.0807],
    ...         [-0.2280, 1.1500, 0.0612],
    ...         [0.0000, 0.0000, 0.9184],
    ...     ]
    ... )
    >>> M_LMS_p_to_Iab = np.array(
    ...     [
    ...         [0.4000, 0.4000, 0.2000],
    ...         [4.4550, -4.8510, 0.3960],
    ...         [0.8056, 0.3572, -1.1628],
    ...     ]
    ... )
    >>> XYZ_to_Iab(XYZ, LMS_to_LMS_p, M_XYZ_to_LMS, M_LMS_p_to_Iab)
    ... # doctest: +ELLIPSIS
    array([ 0.3842619...,  0.3848730...,  0.1888683...])
    """

    XYZ = to_domain_1(XYZ)

    LMS = vecmul(matrix_XYZ_to_LMS, XYZ)
    LMS_p = LMS_to_LMS_p_callable(LMS)
    Iab = vecmul(matrix_LMS_p_to_Iab, LMS_p)

    return from_range_1(Iab)


def Iab_to_XYZ(
    Iab: ArrayLike,
    LMS_p_to_LMS_callable: Callable,
    matrix_Iab_to_LMS_p: ArrayLike,
    matrix_LMS_to_XYZ: ArrayLike,
) -> NDArrayFloat:
    """
    Convert from *IPT*-like :math:`Iab` colour representation to *CIE XYZ*
    tristimulus values.

    This definition is used to perform conversion from *IPT* colourspace to
    *CIE XYZ* tristimulus values and for other similar conversions. It
    implements a generic transformation from *Lightness* :math:`I`, :math:`a`
    and :math:`b` representing red-green dimension, i.e., the dimension lost by
    protanopes and the yellow-blue dimension, i.e., the dimension lost by
    tritanopes, respectively to *CIE XYZ* tristimulus values.

    Parameters
    ----------
    Iab
        *IPT*-like :math:`Iab` colour representation.
    LMS_p_to_LMS_callable
        Callable applying the reverse non-linearity to the :math:`LMS_p`
        colourspace array.
    matrix_Iab_to_LMS_p
        Matrix converting from *IPT*-like :math:`Iab` colour representation to
        non-linear :math:`LMS_p` colourspace.
    matrix_LMS_to_XYZ
        Matrix converting from :math:`LMS` colourspace to *CIE XYZ* tristimulus
        values.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ* tristimulus values.

    Notes
    -----
    +------------+-----------------------+-----------------+
    | **Domain** | **Scale - Reference** | **Scale - 1**   |
    +============+=======================+=================+
    | ``Iab``    | ``I`` : [0, 1]        | ``I`` : [0, 1]  |
    |            |                       |                 |
    |            | ``a`` : [-1, 1]       | ``a`` : [-1, 1] |
    |            |                       |                 |
    |            | ``b`` : [-1, 1]       | ``b`` : [-1, 1] |
    +------------+-----------------------+-----------------+

    +------------+-----------------------+-----------------+
    | **Range**  | **Scale - Reference** | **Scale - 1**   |
    +============+=======================+=================+
    | ``XYZ``    | [0, 1]                | [0, 1]          |
    +------------+-----------------------+-----------------+

    Examples
    --------
    >>> Iab = np.array([0.38426191, 0.38487306, 0.18886838])
    >>> LMS_p_to_LMS = lambda x: x ** (1 / 0.43)
    >>> M_Iab_to_LMS_p = np.linalg.inv(
    ...     np.array(
    ...         [
    ...             [0.4000, 0.4000, 0.2000],
    ...             [4.4550, -4.8510, 0.3960],
    ...             [0.8056, 0.3572, -1.1628],
    ...         ]
    ...     )
    ... )
    >>> M_LMS_to_XYZ = np.linalg.inv(
    ...     np.array(
    ...         [
    ...             [0.4002, 0.7075, -0.0807],
    ...             [-0.2280, 1.1500, 0.0612],
    ...             [0.0000, 0.0000, 0.9184],
    ...         ]
    ...     )
    ... )
    >>> Iab_to_XYZ(Iab, LMS_p_to_LMS, M_Iab_to_LMS_p, M_LMS_to_XYZ)
    ... # doctest: +ELLIPSIS
    array([ 0.2065400...,  0.1219722...,  0.0513695...])
    """

    Iab = to_domain_1(Iab)

    LMS = vecmul(matrix_Iab_to_LMS_p, Iab)
    LMS_p = LMS_p_to_LMS_callable(LMS)
    XYZ = vecmul(matrix_LMS_to_XYZ, LMS_p)

    return from_range_1(XYZ)
