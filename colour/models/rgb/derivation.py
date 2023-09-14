"""
RGB Colourspace Derivation
==========================

Defines the objects related to *RGB* colourspace derivation, essentially
calculating the normalised primary matrix for given *RGB* colourspace primaries
and whitepoint:

-   :func:`colour.normalised_primary_matrix`
-   :func:`colour.chromatically_adapted_primaries`
-   :func:`colour.primaries_whitepoint`
-   :func:`colour.RGB_luminance_equation`
-   :func:`colour.RGB_luminance`

References
----------
-   :cite:`SocietyofMotionPictureandTelevisionEngineers1993a` : Society of
    Motion Picture and Television Engineers. (1993). RP 177:1993 - Derivation
    of Basic Television Color Equations. In RP 177:1993: Vol. RP 177:199. The
    Society of Motion Picture and Television Engineers.
    doi:10.5594/S9781614821915
-   :cite:`Trieu2015a` : Borer, T. (2017). Private Discussion with Mansencal,
    T. and Shaw, N.
"""

from __future__ import annotations

import numpy as np

from colour.adaptation import chromatic_adaptation_VonKries
from colour.hints import (
    ArrayLike,
    LiteralChromaticAdaptationTransform,
    NDArrayFloat,
    Tuple,
)
from colour.models import XYZ_to_xy, XYZ_to_xyY, xy_to_XYZ
from colour.utilities import as_float, as_float_array, ones, tsplit

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "xy_to_z",
    "normalised_primary_matrix",
    "chromatically_adapted_primaries",
    "primaries_whitepoint",
    "RGB_luminance_equation",
    "RGB_luminance",
]


def xy_to_z(xy: ArrayLike) -> float:
    """
    Return the *z* coordinate using given :math:`xy` chromaticity coordinates.

    Parameters
    ----------
    xy
        :math:`xy` chromaticity coordinates.

    Returns
    -------
    :class:`float`
        *z* coordinate.

    Examples
    --------
    >>> xy_to_z(np.array([0.25, 0.25]))
    0.5
    """

    x, y = tsplit(xy)

    z = 1 - x - y

    return z


def normalised_primary_matrix(
    primaries: ArrayLike, whitepoint: ArrayLike
) -> NDArrayFloat:
    """
    Compute the *Normalised Primary Matrix* (NPM) converting a *RGB*
    colourspace array to *CIE XYZ* tristimulus values using given *primaries*
    and *whitepoint* :math:`xy` chromaticity coordinates.

    Parameters
    ----------
    primaries
        Primaries :math:`xy` chromaticity coordinates.
    whitepoint
        Illuminant / whitepoint :math:`xy` chromaticity coordinates.

    Returns
    -------
    :class:`numpy.ndarray`
        *Normalised Primary Matrix* (NPM).

    References
    ----------
    :cite:`SocietyofMotionPictureandTelevisionEngineers1993a`

    Examples
    --------
    >>> p = np.array([0.73470, 0.26530, 0.00000, 1.00000, 0.00010, -0.07700])
    >>> w = np.array([0.32168, 0.33767])
    >>> normalised_primary_matrix(p, w)  # doctest: +ELLIPSIS
    array([[  9.5255239...e-01,   0.0000000...e+00,   9.3678631...e-05],
           [  3.4396645...e-01,   7.2816609...e-01,  -7.2132546...e-02],
           [  0.0000000...e+00,   0.0000000...e+00,   1.0088251...e+00]])
    """

    primaries = np.reshape(primaries, (3, 2))

    z = as_float_array(xy_to_z(primaries))[..., None]
    primaries = np.transpose(np.hstack([primaries, z]))

    whitepoint = xy_to_XYZ(whitepoint)

    coefficients = np.dot(np.linalg.inv(primaries), whitepoint)
    coefficients = np.diagflat(coefficients)

    npm = np.dot(primaries, coefficients)

    return npm


def chromatically_adapted_primaries(
    primaries: ArrayLike,
    whitepoint_t: ArrayLike,
    whitepoint_r: ArrayLike,
    chromatic_adaptation_transform: LiteralChromaticAdaptationTransform
    | str = "CAT02",
) -> NDArrayFloat:
    """
    Chromatically adapt given *primaries* :math:`xy` chromaticity coordinates
    from test ``whitepoint_t`` to reference ``whitepoint_r``.

    Parameters
    ----------
    primaries
        Primaries :math:`xy` chromaticity coordinates.
    whitepoint_t
        Test illuminant / whitepoint :math:`xy` chromaticity coordinates.
    whitepoint_r
        Reference illuminant / whitepoint :math:`xy` chromaticity coordinates.
    chromatic_adaptation_transform
        *Chromatic adaptation* transform.

    Returns
    -------
    :class:`numpy.ndarray`
        Chromatically adapted primaries :math:`xy` chromaticity coordinates.

    Examples
    --------
    >>> p = np.array([0.64, 0.33, 0.30, 0.60, 0.15, 0.06])
    >>> w_t = np.array([0.31270, 0.32900])
    >>> w_r = np.array([0.34570, 0.35850])
    >>> chromatic_adaptation_transform = "Bradford"
    >>> chromatically_adapted_primaries(
    ...     p, w_t, w_r, chromatic_adaptation_transform
    ... )
    ... # doctest: +ELLIPSIS
    array([[ 0.6484414...,  0.3308533...],
           [ 0.3211951...,  0.5978443...],
           [ 0.1558932...,  0.0660492...]])
    """

    primaries = np.reshape(primaries, (3, 2))

    XYZ_a = chromatic_adaptation_VonKries(
        xy_to_XYZ(primaries),
        xy_to_XYZ(whitepoint_t),
        xy_to_XYZ(whitepoint_r),
        chromatic_adaptation_transform,
    )

    P_a = XYZ_to_xyY(XYZ_a)[..., 0:2]

    return P_a


def primaries_whitepoint(npm: ArrayLike) -> Tuple[NDArrayFloat, NDArrayFloat]:
    """
    Compute the *primaries* and *whitepoint* :math:`xy` chromaticity
    coordinates using given *Normalised Primary Matrix* (NPM).

    Parameters
    ----------
    npm
        *Normalised Primary Matrix*.

    Returns
    -------
    :class:`tuple`
        *Primaries* and *whitepoint* :math:`xy` chromaticity coordinates.

    References
    ----------
    :cite:`Trieu2015a`

    Examples
    --------
    >>> npm = np.array(
    ...     [
    ...         [9.52552396e-01, 0.00000000e00, 9.36786317e-05],
    ...         [3.43966450e-01, 7.28166097e-01, -7.21325464e-02],
    ...         [0.00000000e00, 0.00000000e00, 1.00882518e00],
    ...     ]
    ... )
    >>> p, w = primaries_whitepoint(npm)
    >>> p  # doctest: +ELLIPSIS
    array([[  7.3470000...e-01,   2.6530000...e-01],
           [  0.0000000...e+00,   1.0000000...e+00],
           [  1.0000000...e-04,  -7.7000000...e-02]])
    >>> w  # doctest: +ELLIPSIS
    array([ 0.32168,  0.33767])
    """

    npm = np.reshape(npm, (3, 3))

    primaries = XYZ_to_xy(np.transpose(np.dot(npm, np.identity(3))))
    whitepoint = np.squeeze(XYZ_to_xy(np.transpose(np.dot(npm, ones((3, 1))))))

    # TODO: Investigate if we return an ndarray here with primaries and
    # whitepoint stacked together.
    return primaries, whitepoint


def RGB_luminance_equation(primaries: ArrayLike, whitepoint: ArrayLike) -> str:
    """
    Return the *luminance equation* from given *primaries* and *whitepoint*.

    Parameters
    ----------
    primaries
        Primaries chromaticity coordinates.
    whitepoint
        Illuminant / whitepoint chromaticity coordinates.

    Returns
    -------
    :class:`str`
        *Luminance* equation.

    Examples
    --------
    >>> p = np.array([0.73470, 0.26530, 0.00000, 1.00000, 0.00010, -0.07700])
    >>> whitepoint = np.array([0.32168, 0.33767])
    >>> RGB_luminance_equation(p, whitepoint)  # doctest: +ELLIPSIS
    'Y = 0.3439664...(R) + 0.7281660...(G) + -0.0721325...(B)'
    """

    return "Y = {}(R) + {}(G) + {}(B)".format(
        *np.ravel(normalised_primary_matrix(primaries, whitepoint))[3:6]
    )


def RGB_luminance(
    RGB: ArrayLike, primaries: ArrayLike, whitepoint: ArrayLike
) -> NDArrayFloat:
    """
    Return the *luminance* :math:`Y` of given *RGB* components from given
    *primaries* and *whitepoint*.

    Parameters
    ----------
    RGB
        *RGB* chromaticity coordinate matrix.
    primaries
        Primaries chromaticity coordinate matrix.
    whitepoint
        Illuminant / whitepoint chromaticity coordinates.

    Returns
    -------
    :class:`numpy.ndarray`
        *Luminance* :math:`Y`.

    Examples
    --------
    >>> RGB = np.array([0.21959402, 0.06986677, 0.04703877])
    >>> p = np.array([0.73470, 0.26530, 0.00000, 1.00000, 0.00010, -0.07700])
    >>> whitepoint = np.array([0.32168, 0.33767])
    >>> RGB_luminance(RGB, p, whitepoint)  # doctest: +ELLIPSIS
    0.1230145...
    """

    Y = np.sum(
        normalised_primary_matrix(primaries, whitepoint)[1] * RGB, axis=-1
    )

    return as_float(Y)
