"""
Finlayson et al. (2015) Colour Correction
=========================================

Define the *Finlayson et al. (2015)* colour correction objects:

-   :func:`colour.characterisation.polynomial_expansion_Finlayson2015`
-   :func:`colour.characterisation.matrix_colour_correction_Finlayson2015`
-   :func:`colour.characterisation.apply_matrix_colour_correction_Finlayson2015`
-   :func:`colour.characterisation.colour_correction_Finlayson2015`

References
----------
-   :cite:`Finlayson2015` : Finlayson, G. D., MacKiewicz, M., & Hurlbert, A.
    (2015). Color Correction Using Root-Polynomial Regression. IEEE
    Transactions on Image Processing, 24(5), 1460-1470.
    doi:10.1109/TIP.2015.2405336
"""

from __future__ import annotations

import typing

import numpy as np

from colour.algebra import least_square_mapping_MoorePenrose, spow

if typing.TYPE_CHECKING:
    from colour.hints import ArrayLike, Literal, NDArrayFloat

from colour.utilities import (
    as_float,
    as_float_array,
    as_int,
    closest,
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
    "polynomial_expansion_Finlayson2015",
    "matrix_colour_correction_Finlayson2015",
    "apply_matrix_colour_correction_Finlayson2015",
    "colour_correction_Finlayson2015",
]


def polynomial_expansion_Finlayson2015(
    RGB: ArrayLike,
    degree: Literal[1, 2, 3, 4] | int = 1,
    root_polynomial_expansion: bool = True,
) -> NDArrayFloat:
    """
    Perform polynomial expansion of the specified *RGB* colourspace
    array using the *Finlayson et al. (2015)* method.

    Parameters
    ----------
    RGB
        *RGB* colourspace array to expand using polynomial expansion.
    degree
        Expanded polynomial degree.
    root_polynomial_expansion
        Whether to use the root-polynomials set for the expansion.

    Returns
    -------
    :class:`numpy.ndarray`
        Polynomial-expanded *RGB* colourspace array.

    References
    ----------
    :cite:`Finlayson2015`

    Examples
    --------
    >>> RGB = np.array([0.17224810, 0.09170660, 0.06416938])
    >>> polynomial_expansion_Finlayson2015(RGB, degree=2)  # doctest: +ELLIPSIS
    array([0.1722481..., 0.0917066..., 0.0641693..., 0.1256832..., 0.0767121...,
           0.1051335...])
    """

    RGB = as_float_array(RGB)

    R, G, B = tsplit(RGB)

    # TODO: Generalise polynomial expansion.
    existing_degrees = np.array([1, 2, 3, 4])
    closest_degree = as_int(closest(existing_degrees, degree))
    if closest_degree != degree:
        error = (
            f'"Finlayson et al. (2015)" method does not define a polynomial '
            f"expansion for {degree} degree, closest polynomial expansion is "
            f"{closest_degree} degree!"
        )

        raise ValueError(error)

    if degree == 1:
        expansion = RGB
    elif degree == 2:
        if root_polynomial_expansion:
            expansion = tstack(
                [
                    as_float(R),
                    as_float(G),
                    as_float(B),
                    spow(R * G, 1 / 2),
                    spow(G * B, 1 / 2),
                    spow(R * B, 1 / 2),
                ]
            )

        else:
            expansion = tstack(
                [
                    R,
                    G,
                    B,
                    R**2,
                    G**2,
                    B**2,
                    R * G,
                    G * B,
                    R * B,
                ]
            )
    elif degree == 3:
        if root_polynomial_expansion:
            expansion = tstack(
                [
                    as_float(R),
                    as_float(G),
                    as_float(B),
                    spow(R * G, 1 / 2),
                    spow(G * B, 1 / 2),
                    spow(R * B, 1 / 2),
                    spow(R * G**2, 1 / 3),
                    spow(G * B**2, 1 / 3),
                    spow(R * B**2, 1 / 3),
                    spow(G * R**2, 1 / 3),
                    spow(B * G**2, 1 / 3),
                    spow(B * R**2, 1 / 3),
                    spow(R * G * B, 1 / 3),
                ]
            )
        else:
            expansion = tstack(
                [
                    R,
                    G,
                    B,
                    R**2,
                    G**2,
                    B**2,
                    R * G,
                    G * B,
                    R * B,
                    R**3,
                    G**3,
                    B**3,
                    R * G**2,
                    G * B**2,
                    R * B**2,
                    G * R**2,
                    B * G**2,
                    B * R**2,
                    R * G * B,
                ]
            )
    elif degree == 4:
        if root_polynomial_expansion:
            expansion = tstack(
                [
                    as_float(R),
                    as_float(G),
                    as_float(B),
                    spow(R * G, 1 / 2),
                    spow(G * B, 1 / 2),
                    spow(R * B, 1 / 2),
                    spow(R * G**2, 1 / 3),
                    spow(G * B**2, 1 / 3),
                    spow(R * B**2, 1 / 3),
                    spow(G * R**2, 1 / 3),
                    spow(B * G**2, 1 / 3),
                    spow(B * R**2, 1 / 3),
                    spow(R * G * B, 1 / 3),
                    spow(R**3 * G, 1 / 4),
                    spow(R**3 * B, 1 / 4),
                    spow(G**3 * R, 1 / 4),
                    spow(G**3 * B, 1 / 4),
                    spow(B**3 * R, 1 / 4),
                    spow(B**3 * G, 1 / 4),
                    spow(R**2 * G * B, 1 / 4),
                    spow(G**2 * R * B, 1 / 4),
                    spow(B**2 * R * G, 1 / 4),
                ]
            )
        else:
            expansion = tstack(
                [
                    R,
                    G,
                    B,
                    R**2,
                    G**2,
                    B**2,
                    R * G,
                    G * B,
                    R * B,
                    R**3,
                    G**3,
                    B**3,
                    R * G**2,
                    G * B**2,
                    R * B**2,
                    G * R**2,
                    B * G**2,
                    B * R**2,
                    R * G * B,
                    R**4,
                    G**4,
                    B**4,
                    R**3 * G,
                    R**3 * B,
                    G**3 * R,
                    G**3 * B,
                    B**3 * R,
                    B**3 * G,
                    R**2 * G**2,
                    G**2 * B**2,
                    R**2 * B**2,
                    R**2 * G * B,
                    G**2 * R * B,
                    B**2 * R * G,
                ]
            )

    return expansion


def matrix_colour_correction_Finlayson2015(
    M_T: ArrayLike,
    M_R: ArrayLike,
    degree: Literal[1, 2, 3, 4] | int = 1,
    root_polynomial_expansion: bool = True,
) -> NDArrayFloat:
    """
    Compute a colour correction matrix from test colour array :math:`M_T` to
    reference colour array :math:`M_R` using *Finlayson et al. (2015)*
    root-polynomial colour correction method.

    Parameters
    ----------
    M_T
        Test array :math:`M_T` to fit onto reference array :math:`M_R`.
    M_R
        Reference array the test array :math:`M_T` will be colour fitted
        against.
    degree
        Polynomial expansion degree for the root-polynomial basis. The value
        must be one of the degrees: 1, 2, 3, 4.
    root_polynomial_expansion
        Whether to use the root-polynomial basis set for the expansion. If
        *False*, uses standard polynomial expansion.

    Returns
    -------
    :class:`numpy.ndarray`
        Colour correction matrix mapping expanded test colours to reference
        colours.

    References
    ----------
    :cite:`Finlayson2015`

    Examples
    --------
    >>> prng = np.random.RandomState(2)
    >>> M_T = prng.random_sample((24, 3))
    >>> M_R = M_T + (prng.random_sample((24, 3)) - 0.5) * 0.5
    >>> matrix_colour_correction_Finlayson2015(M_T, M_R)  # doctest: +ELLIPSIS
    array([[ 1.0526376...,  0.1378078..., -0.2276339...],
           [ 0.0739584...,  1.0293994..., -0.1060115...],
           [ 0.0572550..., -0.2052633...,  1.1015194...]])
    """

    return least_square_mapping_MoorePenrose(
        polynomial_expansion_Finlayson2015(M_T, degree, root_polynomial_expansion),
        M_R,
    )


def apply_matrix_colour_correction_Finlayson2015(
    RGB: ArrayLike,
    CCM: ArrayLike,
    degree: Literal[1, 2, 3, 4] | int = 1,
    root_polynomial_expansion: bool = True,
) -> NDArrayFloat:
    """
    Apply colour correction matrix :math:`CCM` computed using
    *Finlayson et al. (2015)* method to the specified *RGB* colourspace array.

    Parameters
    ----------
    RGB
        *RGB* colourspace array to which the colour correction matrix
        :math:`CCM` is applied.
    CCM
        Colour correction matrix :math:`CCM`.
    degree
        Expanded polynomial degree.
    root_polynomial_expansion
        Whether to use the root-polynomials set for the expansion.

    Returns
    -------
    :class:`numpy.ndarray`
        Colour corrected *RGB* colourspace array.

    References
    ----------
    :cite:`Finlayson2015`

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
    >>> apply_matrix_colour_correction_Finlayson2015(RGB, CCM)  # doctest: +ELLIPSIS
    array([0.1793456..., 0.1003392..., 0.0617218...])
    """

    RGB = as_float_array(RGB)
    shape = RGB.shape

    RGB = np.reshape(RGB, (-1, 3))

    RGB_e = polynomial_expansion_Finlayson2015(RGB, degree, root_polynomial_expansion)

    return np.reshape(np.transpose(np.dot(CCM, np.transpose(RGB_e))), shape)


def colour_correction_Finlayson2015(
    RGB: ArrayLike,
    M_T: ArrayLike,
    M_R: ArrayLike,
    degree: Literal[1, 2, 3, 4] | int = 1,
    root_polynomial_expansion: bool = True,
) -> NDArrayFloat:
    """
    Perform colour correction of *RGB* colourspace array using the colour
    correction matrix from test array :math:`M_T` to reference array
    :math:`M_R` using the *Finlayson et al. (2015)* method.

    Parameters
    ----------
    RGB
        *RGB* colourspace array to colour correct.
    M_T
        Test array :math:`M_T` to fit onto reference array :math:`M_R`.
    M_R
        Reference array that the test array :math:`M_T` will be fitted
        against.
    degree
        Polynomial expansion degree.
    root_polynomial_expansion
        Whether to use the root-polynomial set for the expansion.

    Returns
    -------
    :class:`numpy.ndarray`
        Colour corrected *RGB* colourspace array.

    References
    ----------
    :cite:`Finlayson2015`

    Examples
    --------
    >>> RGB = np.array([0.17224810, 0.09170660, 0.06416938])
    >>> prng = np.random.RandomState(2)
    >>> M_T = prng.random_sample((24, 3))
    >>> M_R = M_T + (prng.random_sample((24, 3)) - 0.5) * 0.5
    >>> colour_correction_Finlayson2015(RGB, M_T, M_R)  # doctest: +ELLIPSIS
    array([0.1793456..., 0.1003392..., 0.0617218...])
    """

    return apply_matrix_colour_correction_Finlayson2015(
        RGB,
        matrix_colour_correction_Finlayson2015(
            M_T, M_R, degree, root_polynomial_expansion
        ),
        degree,
        root_polynomial_expansion,
    )
