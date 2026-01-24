"""
Cheung et al. (2004) Colour Correction
======================================

Define the *Cheung et al. (2004)* colour correction objects:

-   :func:`colour.characterisation.matrix_augmented_Cheung2004`
-   :func:`colour.characterisation.matrix_colour_correction_Cheung2004`
-   :func:`colour.characterisation.apply_matrix_colour_correction_Cheung2004`
-   :func:`colour.characterisation.colour_correction_Cheung2004`

References
----------
-   :cite:`Cheung2004` : Cheung, V., Westland, S., Connah, D., & Ripamonti, C.
    (2004). A comparative study of the characterisation of colour cameras by
    means of neural networks and polynomial transforms. Coloration Technology,
    120(1), 19-25. doi:10.1111/j.1478-4408.2004.tb00201.x
-   :cite:`Westland2004` : Westland, S., & Ripamonti, C. (2004). Table 8.2. In
    Computational Colour Science Using MATLAB (1st ed., p. 137). John Wiley &
    Sons, Ltd. doi:10.1002/0470020326
"""

from __future__ import annotations

import typing

import numpy as np

from colour.algebra import least_square_mapping_MoorePenrose

if typing.TYPE_CHECKING:
    from colour.hints import ArrayLike, Literal, NDArrayFloat

from colour.utilities import (
    as_float_array,
    as_int,
    closest,
    ones,
    tsplit,
    tstack,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "matrix_augmented_Cheung2004",
    "matrix_colour_correction_Cheung2004",
    "apply_matrix_colour_correction_Cheung2004",
    "colour_correction_Cheung2004",
]


def matrix_augmented_Cheung2004(
    RGB: ArrayLike,
    terms: Literal[3, 4, 5, 7, 8, 10, 11, 14, 16, 17, 19, 20, 22, 35] | int = 3,
) -> NDArrayFloat:
    """
    Perform polynomial expansion of *RGB* colourspace array using
    *Cheung et al. (2004)* method.

    Parameters
    ----------
    RGB
        *RGB* colourspace array to expand using polynomial expansion.
    terms
        Number of terms of the expanded polynomial.

    Returns
    -------
    :class:`numpy.ndarray`
        Polynomial-expanded *RGB* colourspace array.

    Notes
    -----
    -   This definition combines the augmented matrices specified in
        :cite:`Cheung2004` and :cite:`Westland2004`.

    References
    ----------
    :cite:`Cheung2004`, :cite:`Westland2004`

    Examples
    --------
    >>> RGB = np.array([0.17224810, 0.09170660, 0.06416938])
    >>> matrix_augmented_Cheung2004(RGB, terms=5)  # doctest: +ELLIPSIS
    array([0.1722481..., 0.0917066..., 0.0641693..., 0.0010136..., 1...])
    """

    RGB = as_float_array(RGB)

    R, G, B = tsplit(RGB)
    tail = ones(R.shape)

    existing_terms = np.array([3, 4, 5, 7, 8, 10, 11, 14, 16, 17, 19, 20, 22, 35])
    closest_terms = as_int(closest(existing_terms, terms))
    if closest_terms != terms:
        error = (
            f'"Cheung et al. (2004)" method does not define an augmented '
            f"matrix with {terms} terms, closest augmented matrix has "
            f"{closest_terms} terms!"
        )

        raise ValueError(error)

    if terms == 3:
        expansion = RGB
    elif terms == 4:
        expansion = tstack([R, G, B, tail])
    elif terms == 5:
        expansion = tstack(
            [
                R,
                G,
                B,
                R * G * B,
                tail,
            ]
        )
    elif terms == 7:
        expansion = tstack(
            [
                R,
                G,
                B,
                R * G,
                R * B,
                G * B,
                tail,
            ]
        )
    elif terms == 8:
        expansion = tstack(
            [
                R,
                G,
                B,
                R * G,
                R * B,
                G * B,
                R * G * B,
                tail,
            ]
        )
    elif terms == 10:
        expansion = tstack(
            [
                R,
                G,
                B,
                R * G,
                R * B,
                G * B,
                R**2,
                G**2,
                B**2,
                tail,
            ]
        )
    elif terms == 11:
        expansion = tstack(
            [
                R,
                G,
                B,
                R * G,
                R * B,
                G * B,
                R**2,
                G**2,
                B**2,
                R * G * B,
                tail,
            ]
        )
    elif terms == 14:
        expansion = tstack(
            [
                R,
                G,
                B,
                R * G,
                R * B,
                G * B,
                R**2,
                G**2,
                B**2,
                R * G * B,
                R**3,
                G**3,
                B**3,
                tail,
            ]
        )
    elif terms == 16:
        expansion = tstack(
            [
                R,
                G,
                B,
                R * G,
                R * B,
                G * B,
                R**2,
                G**2,
                B**2,
                R * G * B,
                R**2 * G,
                G**2 * B,
                B**2 * R,
                R**3,
                G**3,
                B**3,
            ]
        )
    elif terms == 17:
        expansion = tstack(
            [
                R,
                G,
                B,
                R * G,
                R * B,
                G * B,
                R**2,
                G**2,
                B**2,
                R * G * B,
                R**2 * G,
                G**2 * B,
                B**2 * R,
                R**3,
                G**3,
                B**3,
                tail,
            ]
        )
    elif terms == 19:
        expansion = tstack(
            [
                R,
                G,
                B,
                R * G,
                R * B,
                G * B,
                R**2,
                G**2,
                B**2,
                R * G * B,
                R**2 * G,
                G**2 * B,
                B**2 * R,
                R**2 * B,
                G**2 * R,
                B**2 * G,
                R**3,
                G**3,
                B**3,
            ]
        )
    elif terms == 20:
        expansion = tstack(
            [
                R,
                G,
                B,
                R * G,
                R * B,
                G * B,
                R**2,
                G**2,
                B**2,
                R * G * B,
                R**2 * G,
                G**2 * B,
                B**2 * R,
                R**2 * B,
                G**2 * R,
                B**2 * G,
                R**3,
                G**3,
                B**3,
                tail,
            ]
        )
    elif terms == 22:
        expansion = tstack(
            [
                R,
                G,
                B,
                R * G,
                R * B,
                G * B,
                R**2,
                G**2,
                B**2,
                R * G * B,
                R**2 * G,
                G**2 * B,
                B**2 * R,
                R**2 * B,
                G**2 * R,
                B**2 * G,
                R**3,
                G**3,
                B**3,
                R**2 * G * B,
                R * G**2 * B,
                R * G * B**2,
            ]
        )
    elif terms == 35:
        expansion = tstack(
            [
                R,
                G,
                B,
                R * G,
                R * B,
                G * B,
                R**2,
                G**2,
                B**2,
                R * G * B,
                R**2 * G,
                G**2 * B,
                B**2 * R,
                R**2 * B,
                G**2 * R,
                B**2 * G,
                R**3,
                G**3,
                B**3,
                R**3 * G,
                R**3 * B,
                G**3 * R,
                G**3 * B,
                B**3 * R,
                B**3 * G,
                R**2 * G * B,
                R * G**2 * B,
                R * G * B**2,
                R**2 * G**2,
                R**2 * B**2,
                G**2 * B**2,
                R**4,
                G**4,
                B**4,
                tail,
            ]
        )

    return expansion


def matrix_colour_correction_Cheung2004(
    M_T: ArrayLike,
    M_R: ArrayLike,
    terms: Literal[3, 4, 5, 7, 8, 10, 11, 14, 16, 17, 19, 20, 22, 35] | int = 3,
) -> NDArrayFloat:
    """
    Compute a colour correction matrix from test array :math:`M_T` to
    reference array :math:`M_R` using the *Cheung et al. (2004)* polynomial
    expansion method.

    Parameters
    ----------
    M_T
        Test array :math:`M_T` to fit onto reference array :math:`M_R`.
    M_R
        Reference array that the test array :math:`M_T` will be colour
        fitted against.
    terms
        Number of terms of the expanded polynomial. The value must be one
        of the supported term counts: 3, 4, 5, 7, 8, 10, 11, 14, 16, 17,
        19, 20, 22, or 35.

    Returns
    -------
    :class:`numpy.ndarray`
        Colour correction matrix mapping expanded test colours to reference
        colours.

    References
    ----------
    :cite:`Cheung2004`, :cite:`Westland2004`

    Examples
    --------
    >>> prng = np.random.RandomState(2)
    >>> M_T = prng.random_sample((24, 3))
    >>> M_R = M_T + (prng.random_sample((24, 3)) - 0.5) * 0.5
    >>> matrix_colour_correction_Cheung2004(M_T, M_R)  # doctest: +ELLIPSIS
    array([[ 1.0526376...,  0.1378078..., -0.2276339...],
           [ 0.0739584...,  1.0293994..., -0.1060115...],
           [ 0.0572550..., -0.2052633...,  1.1015194...]])
    """

    return least_square_mapping_MoorePenrose(
        matrix_augmented_Cheung2004(M_T, terms), M_R
    )


def apply_matrix_colour_correction_Cheung2004(
    RGB: ArrayLike,
    CCM: ArrayLike,
    terms: Literal[3, 4, 5, 7, 8, 10, 11, 14, 16, 17, 19, 20, 22, 35] | int = 3,
) -> NDArrayFloat:
    """
    Apply colour correction matrix :math:`CCM` computed using *Cheung et al.
    (2004)* method to the specified *RGB* colourspace array.

    Parameters
    ----------
    RGB
        *RGB* colourspace array to which the colour correction matrix
        :math:`CCM` is applied.
    CCM
        Colour correction matrix :math:`CCM`.
    terms
        Number of terms of the expanded polynomial.

    Returns
    -------
    :class:`numpy.ndarray`
        Colour corrected *RGB* colourspace array.

    References
    ----------
    :cite:`Cheung2004`, :cite:`Westland2004`

    Examples
    --------
    >>> RGB = np.array([0.17224810, 0.09170660, 0.06416938])
    >>> CCM = np.array(
    ...     [
    ...         [1.05263767, 0.13780789, -0.22763399],
    ...         [0.07395843, 1.02939945, -0.1060115],
    ...         [0.05725508, -0.20526336, 1.10151945],
    ...     ]
    ... )
    >>> apply_matrix_colour_correction_Cheung2004(RGB, CCM)  # doctest: +ELLIPSIS
    array([0.1793456..., 0.1003392..., 0.0617218...])
    """

    RGB = as_float_array(RGB)
    shape = RGB.shape

    RGB = np.reshape(RGB, (-1, 3))

    RGB_e = matrix_augmented_Cheung2004(RGB, terms)

    return np.reshape(np.transpose(np.dot(CCM, np.transpose(RGB_e))), shape)


def colour_correction_Cheung2004(
    RGB: ArrayLike,
    M_T: ArrayLike,
    M_R: ArrayLike,
    terms: Literal[3, 4, 5, 7, 8, 10, 11, 14, 16, 17, 19, 20, 22, 35] | int = 3,
) -> NDArrayFloat:
    """
    Perform colour correction of the specified *RGB* colourspace array using
    the colour correction matrix derived from test array :math:`M_T` to
    reference array :math:`M_R` using the *Cheung et al. (2004)* method.

    Parameters
    ----------
    RGB
        *RGB* colourspace array to colour correct.
    M_T
        Test array :math:`M_T` to fit onto reference array :math:`M_R`.
    M_R
        Reference array that the test array :math:`M_T` will be colour
        fitted against.
    terms
        Number of terms of the expanded polynomial.

    Returns
    -------
    :class:`numpy.ndarray`
        Colour corrected *RGB* colourspace array.

    References
    ----------
    :cite:`Cheung2004`, :cite:`Westland2004`

    Examples
    --------
    >>> RGB = np.array([0.17224810, 0.09170660, 0.06416938])
    >>> prng = np.random.RandomState(2)
    >>> M_T = prng.random_sample((24, 3))
    >>> M_R = M_T + (prng.random_sample((24, 3)) - 0.5) * 0.5
    >>> colour_correction_Cheung2004(RGB, M_T, M_R)  # doctest: +ELLIPSIS
    array([0.1793456..., 0.1003392..., 0.0617218...])
    """

    return apply_matrix_colour_correction_Cheung2004(
        RGB, matrix_colour_correction_Cheung2004(M_T, M_R, terms), terms
    )
