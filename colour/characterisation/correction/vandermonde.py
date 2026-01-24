"""
Vandermonde Colour Correction
=============================

Define the *Vandermonde* colour correction objects:

-   :func:`colour.characterisation.polynomial_expansion_Vandermonde`
-   :func:`colour.characterisation.matrix_colour_correction_Vandermonde`
-   :func:`colour.characterisation.apply_matrix_colour_correction_Vandermonde`
-   :func:`colour.characterisation.colour_correction_Vandermonde`

References
----------
-   :cite:`Wikipedia2003e` : Wikipedia. (2003). Vandermonde matrix. Retrieved
    May 2, 2018, from https://en.wikipedia.org/wiki/Vandermonde_matrix
"""

from __future__ import annotations

import typing

import numpy as np

from colour.algebra import least_square_mapping_MoorePenrose

if typing.TYPE_CHECKING:
    from colour.hints import ArrayLike, NDArrayFloat

from colour.utilities import as_float_array

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "polynomial_expansion_Vandermonde",
    "matrix_colour_correction_Vandermonde",
    "apply_matrix_colour_correction_Vandermonde",
    "colour_correction_Vandermonde",
]


def polynomial_expansion_Vandermonde(a: ArrayLike, degree: int = 1) -> NDArrayFloat:
    """
    Perform polynomial expansion of the specified :math:`a` array using the
    *Vandermonde* method.

    Parameters
    ----------
    a
        Array :math:`a` to expand using polynomial expansion.
    degree
        Degree of the expanded polynomial.

    Returns
    -------
    :class:`numpy.ndarray`
        Polynomial-expanded :math:`a` array.

    References
    ----------
    :cite:`Wikipedia2003e`

    Examples
    --------
    >>> RGB = np.array([0.17224810, 0.09170660, 0.06416938])
    >>> polynomial_expansion_Vandermonde(RGB)  # doctest: +ELLIPSIS
    array([0.1722481..., 0.0917066..., 0.0641693..., 1...])
    """

    a = as_float_array(a)

    a_e = np.transpose(np.vander(np.ravel(a), int(degree) + 1))
    a_e = np.hstack(list(np.reshape(a_e, (a_e.shape[0], -1, 3))))

    return np.squeeze(a_e[:, 0 : a_e.shape[-1] - a.shape[-1] + 1])


def matrix_colour_correction_Vandermonde(
    M_T: ArrayLike, M_R: ArrayLike, degree: int = 1
) -> NDArrayFloat:
    """
    Compute a colour correction matrix from :math:`M_T` test colour array
    to :math:`M_R` reference colour array using the *Vandermonde* method.

    Parameters
    ----------
    M_T
        Test array :math:`M_T` to fit onto array :math:`M_R`.
    M_R
        Reference array the array :math:`M_T` will be colour fitted against.
    degree
        Expanded polynomial degree.

    Returns
    -------
    :class:`numpy.ndarray`
        Colour correction matrix mapping expanded test colours to reference
        colours.

    References
    ----------
    :cite:`Wikipedia2003e`

    Examples
    --------
    >>> prng = np.random.RandomState(2)
    >>> M_T = prng.random_sample((24, 3))
    >>> M_R = M_T + (prng.random_sample((24, 3)) - 0.5) * 0.5
    >>> matrix_colour_correction_Vandermonde(M_T, M_R)  # doctest: +ELLIPSIS
    array([[ 1.0300256...,  0.1141770..., -0.2621816...,  0.0418022...],
           [ 0.0670209...,  1.0221494..., -0.1166108...,  0.0128250...],
           [ 0.0744612..., -0.1872819...,  1.1278078..., -0.0318085...]])
    """

    return least_square_mapping_MoorePenrose(
        polynomial_expansion_Vandermonde(M_T, degree), M_R
    )


def apply_matrix_colour_correction_Vandermonde(
    RGB: ArrayLike, CCM: ArrayLike, degree: int = 1
) -> NDArrayFloat:
    """
    Apply colour correction matrix :math:`CCM` computed using the
    *Vandermonde* method to the specified *RGB* colourspace array.

    Parameters
    ----------
    RGB
        *RGB* colourspace array to which the colour correction matrix
        :math:`CCM` is applied.
    CCM
        Colour correction matrix :math:`CCM`.
    degree
        Expanded polynomial degree.

    Returns
    -------
    :class:`numpy.ndarray`
        Colour corrected *RGB* colourspace array.

    References
    ----------
    :cite:`Wikipedia2003e`

    Examples
    --------
    >>> RGB = np.array([0.17224810, 0.09170660, 0.06416938])
    >>> CCM = np.array(
    ...     [
    ...         [1.0300256, 0.11417701, -0.26218168, 0.04180222],
    ...         [0.06702098, 1.02214943, -0.11661082, 0.01282503],
    ...         [0.07446128, -0.18728192, 1.12780782, -0.03180856],
    ...     ]
    ... )
    >>> apply_matrix_colour_correction_Vandermonde(RGB, CCM)  # doctest: +ELLIPSIS
    array([0.2128689..., 0.1106242..., 0.0362129...])
    """

    RGB = as_float_array(RGB)
    shape = RGB.shape

    RGB = np.reshape(RGB, (-1, 3))

    RGB_e = polynomial_expansion_Vandermonde(RGB, degree)

    return np.reshape(np.transpose(np.dot(CCM, np.transpose(RGB_e))), shape)


def colour_correction_Vandermonde(
    RGB: ArrayLike, M_T: ArrayLike, M_R: ArrayLike, degree: int = 1
) -> NDArrayFloat:
    """
    Perform colour correction of *RGB* colourspace array using the colour
    correction matrix from :math:`M_T` colour array to :math:`M_R` colour
    array using *Vandermonde* method.

    Parameters
    ----------
    RGB
        *RGB* colourspace array to colour correct.
    M_T
        Test array :math:`M_T` to fit onto array :math:`M_R`.
    M_R
        Reference array the array :math:`M_T` will be colour fitted against.
    degree
        Expanded polynomial degree.

    Returns
    -------
    :class:`numpy.ndarray`
        Colour corrected *RGB* colourspace array.

    References
    ----------
    :cite:`Wikipedia2003e`

    Examples
    --------
    >>> RGB = np.array([0.17224810, 0.09170660, 0.06416938])
    >>> prng = np.random.RandomState(2)
    >>> M_T = prng.random_sample((24, 3))
    >>> M_R = M_T + (prng.random_sample((24, 3)) - 0.5) * 0.5
    >>> colour_correction_Vandermonde(RGB, M_T, M_R)  # doctest: +ELLIPSIS
    array([0.2128689..., 0.1106242..., 0.036213...])
    """

    return apply_matrix_colour_correction_Vandermonde(
        RGB, matrix_colour_correction_Vandermonde(M_T, M_R, degree), degree
    )
