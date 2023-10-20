"""
Yellowness Index :math:`Y`
==========================

Defines the *yellowness* index :math:`Y` computation objects:

-   :func:`colour.colorimetry.yellowness_ASTMD1925`: *Yellowness* index
    :math:`YI` computation of given sample *CIE XYZ* tristimulus values using
    *ASTM D1925* method.
-   :func:`colour.colorimetry.yellowness_ASTME313_alternative`: *Yellowness*
    index :math:`YI` computation of given sample *CIE XYZ* tristimulus values
    using the alternative *ASTM E313* method.
-   :func:`colour.colorimetry.yellowness_ASTME313`: *Yellowness*
    index :math:`YI` computation of given sample *CIE XYZ* tristimulus values
    using the recommended *ASTM E313* method.
-   :attr:`colour.YELLOWNESS_METHODS`: Supported *yellowness* computations
    methods.
-   :func:`colour.yellownes`: *Yellowness* :math:`YI` computation using given
    method.

References
----------
-   :cite:`ASTMInternational2015` : ASTM International. (2015). ASTM E313-15e1
    - Standard Practice for Calculating Yellowness and Whiteness Indices from
    Instrumentally Measured Color Coordinates. doi:10.1520/E0313-20
-   :cite:`X-Rite2012a` : X-Rite, & Pantone. (2012). Color iQC and Color iMatch
    Color Calculations Guide.
    https://www.xrite.com/-/media/xrite/files/apps_engineering_techdocuments/\
c/09_color_calculations_en.pdf
"""

from __future__ import annotations

import numpy as np

from colour.algebra import sdiv, sdiv_mode
from colour.hints import Any, ArrayLike, Literal, NDArrayFloat
from colour.utilities import (
    CanonicalMapping,
    as_float,
    filter_kwargs,
    from_range_100,
    to_domain_100,
    tsplit,
    validate_method,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "yellowness_ASTMD1925",
    "yellowness_ASTME313_alternative",
    "YELLOWNESS_COEFFICIENTS_ASTME313",
    "yellowness_ASTME313",
    "YELLOWNESS_METHODS",
    "yellowness",
]


def yellowness_ASTMD1925(XYZ: ArrayLike) -> NDArrayFloat:
    """
    Return the *yellowness* index :math:`YI` of given sample *CIE XYZ*
    tristimulus values using *ASTM D1925* method.

    ASTM D1925 has been specifically developed for the definition of the
    yellowness of homogeneous, non-fluorescent, almost neutral-transparent,
    white-scattering or opaque plastics as they will be reviewed under daylight
    condition. It can be other materials as well, as long as they fit into this
    description.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values of the sample.

    Returns
    -------
    :class:`np.float` or :class:`numpy.ndarray`
        *Yellowness* :math:`YI`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ``    | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``YI``     | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    -   Input *CIE XYZ* tristimulus values must be adapted to
        *CIE Illuminant C*.

    References
    ----------
    :cite:`ASTMInternational2015`, :cite:`X-Rite2012a`

    Examples
    --------
    >>> XYZ = np.array([95.00000000, 100.00000000, 105.00000000])
    >>> yellowness_ASTMD1925(XYZ)  # doctest: +ELLIPSIS
    10.2999999...
    """

    X, Y, Z = tsplit(to_domain_100(XYZ))

    with sdiv_mode():
        YI = sdiv(100 * (1.28 * X - 1.06 * Z), Y)

    return as_float(from_range_100(YI))


def yellowness_ASTME313_alternative(XYZ: ArrayLike) -> NDArrayFloat:
    """
    Return the *yellowness* index :math:`YI` of given sample *CIE XYZ*
    tristimulus values using the alternative *ASTM E313* method.

    In the original form of *Test Method E313*, an alternative equation was
    recommended for a *yellowness* index. In terms of colorimeter readings,
    it was :math:`YI = 100(1 - B/G)` where :math:`B` and :math:`G` are,
    respectively, blue and green colorimeter readings. Its derivation assumed
    that, because of the limitation of the concept to yellow (or blue) colors,
    it was not necessary to take account of variations in the amber or red
    colorimeter reading :math:`A`. This equation is no longer recommended.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values of the sample.

    Returns
    -------
    :class:`np.float` or :class:`numpy.ndarray`
        *Yellowness* :math:`YI`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ``    | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``YI``     | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    -   Input *CIE XYZ* tristimulus values must be adapted to
        *CIE Illuminant C*.

    References
    ----------
    :cite:`ASTMInternational2015`, :cite:`X-Rite2012a`

    Examples
    --------
    >>> XYZ = np.array([95.00000000, 100.00000000, 105.00000000])
    >>> yellowness_ASTME313_alternative(XYZ)  # doctest: +ELLIPSIS
    11.0650000...
    """

    _X, Y, Z = tsplit(to_domain_100(XYZ))

    with sdiv_mode():
        WI = 100 * (1 - sdiv(0.847 * Z, Y))

    return as_float(from_range_100(WI))


YELLOWNESS_COEFFICIENTS_ASTME313: CanonicalMapping = CanonicalMapping(
    {
        "CIE 1931 2 Degree Standard Observer": CanonicalMapping(
            {
                "C": np.array([1.2769, 1.0592]),
                "D65": np.array([1.2985, 1.1335]),
            }
        ),
        "CIE 1964 10 Degree Standard Observer": CanonicalMapping(
            {
                "C": np.array([1.2871, 1.0781]),
                "D65": np.array([1.3013, 1.1498]),
            }
        ),
    }
)
YELLOWNESS_COEFFICIENTS_ASTME313.__doc__ = """
Coefficients :math:`C_X` and :math:`C_Z` for the *ASTM E313* *yellowness* index
:math:`YI` computation method.

References
----------
:cite:`ASTMInternational2015`

Aliases:

-   'cie_2_1931': 'CIE 1931 2 Degree Standard Observer'
-   'cie_10_1964': 'CIE 1964 10 Degree Standard Observer'
"""
YELLOWNESS_COEFFICIENTS_ASTME313[
    "cie_2_1931"
] = YELLOWNESS_COEFFICIENTS_ASTME313["CIE 1931 2 Degree Standard Observer"]
YELLOWNESS_COEFFICIENTS_ASTME313[
    "cie_10_1964"
] = YELLOWNESS_COEFFICIENTS_ASTME313["CIE 1964 10 Degree Standard Observer"]


def yellowness_ASTME313(
    XYZ: ArrayLike,
    C_XZ: ArrayLike = YELLOWNESS_COEFFICIENTS_ASTME313[
        "CIE 1931 2 Degree Standard Observer"
    ]["D65"],
) -> NDArrayFloat:
    """
    Return the *yellowness* index :math:`YI` of given sample *CIE XYZ*
    tristimulus values using *ASTM E313* method.

    ASTM E313 has successfully been used for a variety of white or near white
    materials. This includes coatings, plastics, textiles.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values of the sample.
    C_XZ
        Coefficients :math:`C_X` and :math:`C_Z` for the
        *CIE 1931 2 Degree Standard Observer* and
        *CIE 1964 10 Degree Standard Observer* and *CIE Illuminant C* and
        *CIE Standard Illuminant D65*.

    Returns
    -------
    :class:`np.float` or :class:`numpy.ndarray`
        *Yellowness* :math:`YI`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ``    | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``YI``     | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`ASTMInternational2015`

    Examples
    --------
    >>> XYZ = np.array([95.00000000, 100.00000000, 105.00000000])
    >>> yellowness_ASTME313(XYZ)  # doctest: +ELLIPSIS
    4.3400000...
    """

    X, Y, Z = tsplit(to_domain_100(XYZ))
    C_X, C_Z = tsplit(C_XZ)

    with sdiv_mode():
        WI = 100 * sdiv(C_X * X - C_Z * Z, Y)

    return as_float(from_range_100(WI))


YELLOWNESS_METHODS = CanonicalMapping(
    {
        "ASTM D1925": yellowness_ASTMD1925,
        "ASTM E313 Alternative": yellowness_ASTME313_alternative,
        "ASTM E313": yellowness_ASTME313,
    }
)
YELLOWNESS_METHODS.__doc__ = """
Supported *yellowness* computation methods.

References
----------
:cite:`ASTMInternational2015`, :cite:`X-Rite2012a`
"""


def yellowness(
    XYZ: ArrayLike,
    method: Literal["ASTM D1925", "ASTM E313", "ASTM E313 Alternative"]
    | str = "ASTM E313",
    **kwargs: Any,
) -> NDArrayFloat:
    """
    Return the *yellowness* :math:`W` using given method.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values of the sample.
    method
        Computation method.

    Other Parameters
    ----------------
    C_XZ
        {:func:`colour.colorimetry.yellowness_ASTME313`},
        Coefficients :math:`C_X` and :math:`C_Z` for the
        *CIE 1931 2 Degree Standard Observer* and
        *CIE 1964 10 Degree Standard Observer* and *CIE Illuminant C* and
        *CIE Standard Illuminant D65*.

    Returns
    -------
    :class:`np.float` or :class:`numpy.ndarray`
        *Yellowness* :math:`Y`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ``    | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``YI``     | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`ASTMInternational2015`, :cite:`X-Rite2012a`

    Examples
    --------
    >>> XYZ = np.array([95.00000000, 100.00000000, 105.00000000])
    >>> yellowness(XYZ)  # doctest: +ELLIPSIS
    4.3400000...
    >>> yellowness(XYZ, method="ASTM E313 Alternative")  # doctest: +ELLIPSIS
    11.0650000...
    >>> yellowness(XYZ, method="ASTM D1925")  # doctest: +ELLIPSIS
    10.2999999...
    """

    method = validate_method(method, tuple(YELLOWNESS_METHODS))

    function = YELLOWNESS_METHODS[method]

    return function(XYZ, **filter_kwargs(function, **kwargs))
