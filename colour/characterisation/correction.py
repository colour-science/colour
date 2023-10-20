"""
Colour Correction
=================

Defines various objects for colour correction, like colour matching two images:

-   :func:`colour.characterisation.matrix_augmented_Cheung2004` : Polynomial
    expansion using *Cheung, Westland, Connah and Ripamonti (2004)* method.
-   :func:`colour.characterisation.polynomial_expansion_Finlayson2015` :
    Polynomial expansion using *Finlayson, MacKiewicz and Hurlbert (2015)*
    method.
-   :func:`colour.characterisation.polynomial_expansion_Vandermonde` :
    Polynomial expansion using *Vandermonde* method.
-   :attr:`colour.POLYNOMIAL_EXPANSION_METHODS` : Supported polynomial
    expansion methods.
-   :func:`colour.polynomial_expansion`: Polynomial expansion of given
    :math:`a` array.
-   :func:`colour.characterisation.matrix_colour_correction_Cheung2004` :
    Colour correction matrix computation using *Cheung et al. (2004)* method.
-   :func:`colour.characterisation.matrix_colour_correction_Finlayson2015` :
    Colour correction matrix computation using *Finlayson et al. (2015)*
    method.
-   :func:`colour.characterisation.matrix_colour_correction_Vandermonde` :
    Colour correction matrix computation using *Vandermonde* method.
-   :attr:`colour.MATRIX_COLOUR_CORRECTION_METHODS` : Supported colour
    correction matrix methods.
-   :func:`colour.matrix_colour_correction` : Colour correction matrix
    computation from given :math:`M_T` colour array to :math:`M_R` colour
    array.
-   :func:`colour.apply_matrix_colour_correction_Cheung2004` : Apply a colour
    correction matrix computed using *Cheung et al. (2004)* method.
-   :func:`colour.apply_matrix_colour_correction_Finlayson2015 `: Apply a
    colour correction matrix computed using *Finlayson et al. (2015)* method.
-   :func:`colour.apply_matrix_colour_correction_Vandermonde` : Apply a colour
    correction matrix computed using *Vandermonde* method.
-   :attr:`colour.APPLY_MATRIX_COLOUR_CORRECTION_METHODS` : Supported methods
    to apply a colour correction matrix .
-   :func:`colour.apply_matrix_colour_correction` : Apply a colour correction
    matrix.
-   :func:`colour.characterisation.colour_correction_Cheung2004` :
    Colour correction using *Cheung et al. (2004)* method.
-   :func:`colour.characterisation.colour_correction_Finlayson2015` :
    Colour correction using *Finlayson et al. (2015)* method.
-   :func:`colour.characterisation.colour_correction_Vandermonde` :
    Colour correction using *Vandermonde* method.
-   :attr:`colour.COLOUR_CORRECTION_METHODS` : Supported colour correction
    methods.
-   :func:`colour.colour_correction` : Colour correction of given *RGB*
    colourspace array using the colour correction matrix from given
    :math:`M_T` colour array to :math:`M_R` colour array.

References
----------
-   :cite:`Cheung2004` : Cheung, V., Westland, S., Connah, D., & Ripamonti, C.
    (2004). A comparative study of the characterisation of colour cameras by
    means of neural networks and polynomial transforms. Coloration Technology,
    120(1), 19-25. doi:10.1111/j.1478-4408.2004.tb00201.x
-   :cite:`Finlayson2015` : Finlayson, G. D., MacKiewicz, M., & Hurlbert, A.
    (2015). Color Correction Using Root-Polynomial Regression. IEEE
    Transactions on Image Processing, 24(5), 1460-1470.
    doi:10.1109/TIP.2015.2405336
-   :cite:`Westland2004` : Westland, S., & Ripamonti, C. (2004). Table 8.2. In
    Computational Colour Science Using MATLAB (1st ed., p. 137). John Wiley &
    Sons, Ltd. doi:10.1002/0470020326
-   :cite:`Wikipedia2003e` : Wikipedia. (2003). Vandermonde matrix. Retrieved
    May 2, 2018, from https://en.wikipedia.org/wiki/Vandermonde_matrix
"""

from __future__ import annotations

import numpy as np

from colour.algebra import least_square_mapping_MoorePenrose, spow
from colour.hints import Any, ArrayLike, Literal, NDArrayFloat
from colour.utilities import (
    CanonicalMapping,
    as_float,
    as_float_array,
    as_int,
    closest,
    filter_kwargs,
    ones,
    tsplit,
    tstack,
    validate_method,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "matrix_augmented_Cheung2004",
    "polynomial_expansion_Finlayson2015",
    "polynomial_expansion_Vandermonde",
    "POLYNOMIAL_EXPANSION_METHODS",
    "polynomial_expansion",
    "matrix_colour_correction_Cheung2004",
    "matrix_colour_correction_Finlayson2015",
    "matrix_colour_correction_Vandermonde",
    "MATRIX_COLOUR_CORRECTION_METHODS",
    "matrix_colour_correction",
    "apply_matrix_colour_correction_Cheung2004",
    "apply_matrix_colour_correction_Finlayson2015",
    "apply_matrix_colour_correction_Vandermonde",
    "APPLY_MATRIX_COLOUR_CORRECTION_METHODS",
    "apply_matrix_colour_correction",
    "colour_correction_Cheung2004",
    "colour_correction_Finlayson2015",
    "colour_correction_Vandermonde",
    "COLOUR_CORRECTION_METHODS",
    "colour_correction",
]


def matrix_augmented_Cheung2004(
    RGB: ArrayLike,
    terms: Literal[3, 4, 5, 7, 8, 10, 11, 14, 16, 17, 19, 20, 22, 35] = 3,
) -> NDArrayFloat:
    """
    Perform polynomial expansion of given *RGB* colourspace array using
    *Cheung et al. (2004)* method.

    Parameters
    ----------
    RGB
        *RGB* colourspace array to expand.
    terms
        Number of terms of the expanded polynomial.

    Returns
    -------
    :class:`numpy.ndarray`
        Expanded *RGB* colourspace array.

    Notes
    -----
    -   This definition combines the augmented matrices given in
        :cite:`Cheung2004` and :cite:`Westland2004`.

    References
    ----------
    :cite:`Cheung2004`, :cite:`Westland2004`

    Examples
    --------
    >>> RGB = np.array([0.17224810, 0.09170660, 0.06416938])
    >>> matrix_augmented_Cheung2004(RGB, terms=5)  # doctest: +ELLIPSIS
    array([ 0.1722481...,  0.0917066...,  0.0641693...,  0.0010136...,  1...])
    """

    RGB = as_float_array(RGB)

    R, G, B = tsplit(RGB)
    tail = ones(R.shape)

    existing_terms = np.array(
        [3, 4, 5, 7, 8, 10, 11, 14, 16, 17, 19, 20, 22, 35]
    )
    closest_terms = as_int(closest(existing_terms, terms))
    if closest_terms != terms:
        raise ValueError(
            f'"Cheung et al. (2004)" method does not define an augmented '
            f"matrix with {terms} terms, closest augmented matrix has "
            f"{closest_terms} terms!"
        )

    if terms == 3:  # noqa: RET503
        return RGB
    elif terms == 4:
        return tstack([R, G, B, tail])
    elif terms == 5:
        return tstack(
            [
                R,
                G,
                B,
                R * G * B,
                tail,
            ]
        )
    elif terms == 7:
        return tstack(
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
        return tstack(
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
        return tstack(
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
        return tstack(
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
        return tstack(
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
        return tstack(
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
        return tstack(
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
        return tstack(
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
        return tstack(
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
        return tstack(
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
        return tstack(
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


def polynomial_expansion_Finlayson2015(
    RGB: ArrayLike,
    degree: Literal[1, 2, 3, 4] = 1,
    root_polynomial_expansion: bool = True,
) -> NDArrayFloat:
    """
    Perform polynomial expansion of given *RGB* colourspace array using
    *Finlayson et al. (2015)* method.

    Parameters
    ----------
    RGB
        *RGB* colourspace array to expand.
    degree
        Expanded polynomial degree.
    root_polynomial_expansion
        Whether to use the root-polynomials set for the expansion.

    Returns
    -------
    :class:`numpy.ndarray`
        Expanded *RGB* colourspace array.

    References
    ----------
    :cite:`Finlayson2015`

    Examples
    --------
    >>> RGB = np.array([0.17224810, 0.09170660, 0.06416938])
    >>> polynomial_expansion_Finlayson2015(RGB, degree=2)  # doctest: +ELLIPSIS
    array([ 0.1722481...,  0.0917066...,  0.0641693...,  0.1256832...,  \
0.0767121...,
            0.1051335...])
    """

    RGB = as_float_array(RGB)

    R, G, B = tsplit(RGB)

    # TODO: Generalise polynomial expansion.
    existing_degrees = np.array([1, 2, 3, 4])
    closest_degree = as_int(closest(existing_degrees, degree))
    if closest_degree != degree:
        raise ValueError(
            f'"Finlayson et al. (2015)" method does not define a polynomial '
            f"expansion for {degree} degree, closest polynomial expansion is "
            f"{closest_degree} degree!"
        )

    if degree == 1:  # noqa: RET503
        return RGB
    elif degree == 2:
        if root_polynomial_expansion:
            return tstack(
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
            return tstack(
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
            return tstack(
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
            return tstack(
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
            return tstack(
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
            return tstack(
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


def polynomial_expansion_Vandermonde(
    a: ArrayLike, degree: int = 1
) -> NDArrayFloat:
    """
    Perform polynomial expansion of given :math:`a` array using *Vandermonde*
    method.

    Parameters
    ----------
    a
        :math:`a` array to expand.
    degree
        Expanded polynomial degree.

    Returns
    -------
    :class:`numpy.ndarray`
        Expanded :math:`a` array.

    References
    ----------
    :cite:`Wikipedia2003e`

    Examples
    --------
    >>> RGB = np.array([0.17224810, 0.09170660, 0.06416938])
    >>> polynomial_expansion_Vandermonde(RGB)  # doctest: +ELLIPSIS
    array([ 0.1722481 ,  0.0917066 ,  0.06416938,  1.        ])
    """

    a = as_float_array(a)

    a_e = np.transpose(np.vander(np.ravel(a), int(degree) + 1))
    a_e = np.hstack(list(np.reshape(a_e, (a_e.shape[0], -1, 3))))

    return np.squeeze(a_e[:, 0 : a_e.shape[-1] - a.shape[-1] + 1])


POLYNOMIAL_EXPANSION_METHODS: CanonicalMapping = CanonicalMapping(
    {
        "Cheung 2004": matrix_augmented_Cheung2004,
        "Finlayson 2015": polynomial_expansion_Finlayson2015,
        "Vandermonde": polynomial_expansion_Vandermonde,
    }
)
POLYNOMIAL_EXPANSION_METHODS.__doc__ = """
Supported polynomial expansion methods.

References
----------
:cite:`Cheung2004`, :cite:`Finlayson2015`, :cite:`Westland2004`,
:cite:`Wikipedia2003e`
"""


def polynomial_expansion(
    a: ArrayLike,
    method: Literal["Cheung 2004", "Finlayson 2015", "Vandermonde"]
    | str = "Cheung 2004",
    **kwargs: Any,
) -> NDArrayFloat:
    """
    Perform polynomial expansion of given :math:`a` array.

    Parameters
    ----------
    a
        :math:`a` array to expand.
    method
        Computation method.

    Other Parameters
    ----------------
    degree
        {:func:`colour.characterisation.polynomial_expansion_Finlayson2015`,
        :func:`colour.characterisation.polynomial_expansion_Vandermonde`},
        Expanded polynomial degree, must be one of *[1, 2, 3, 4]* for
        :func:`colour.characterisation.polynomial_expansion_Finlayson2015`
        definition.
    root_polynomial_expansion
        {:func:`colour.characterisation.polynomial_expansion_Finlayson2015`},
        Whether to use the root-polynomials set for the expansion.
    terms
        {:func:`colour.characterisation.matrix_augmented_Cheung2004`},
        Number of terms of the expanded polynomial.

    Returns
    -------
    :class:`numpy.ndarray`
        Expanded :math:`a` array.

    References
    ----------
    :cite:`Cheung2004`, :cite:`Finlayson2015`, :cite:`Westland2004`,
    :cite:`Wikipedia2003e`

    Examples
    --------
    >>> RGB = np.array([0.17224810, 0.09170660, 0.06416938])
    >>> polynomial_expansion(RGB)  # doctest: +ELLIPSIS
    array([ 0.1722481...,  0.0917066...,  0.0641693...])
    >>> polynomial_expansion(RGB, "Cheung 2004", terms=5)  # doctest: +ELLIPSIS
    array([ 0.1722481...,  0.0917066...,  0.0641693...,  0.0010136...,  1...])
    """

    method = validate_method(method, tuple(POLYNOMIAL_EXPANSION_METHODS))

    function = POLYNOMIAL_EXPANSION_METHODS[method]

    return function(a, **filter_kwargs(function, **kwargs))


def matrix_colour_correction_Cheung2004(
    M_T: ArrayLike,
    M_R: ArrayLike,
    terms: Literal[3, 4, 5, 7, 8, 10, 11, 14, 16, 17, 19, 20, 22, 35] = 3,
) -> NDArrayFloat:
    """
    Compute a colour correction matrix from given :math:`M_T` colour array to
    :math:`M_R` colour array using *Cheung et al. (2004)* method.

    Parameters
    ----------
    M_T
        Test array :math:`M_T` to fit onto array :math:`M_R`.
    M_R
        Reference array the array :math:`M_T` will be colour fitted against.
    terms
        Number of terms of the expanded polynomial.

    Returns
    -------
    :class:`numpy.ndarray`
        Colour correction matrix.

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


def matrix_colour_correction_Finlayson2015(
    M_T: ArrayLike,
    M_R: ArrayLike,
    degree: Literal[1, 2, 3, 4] = 1,
    root_polynomial_expansion: bool = True,
) -> NDArrayFloat:
    """
    Compute a colour correction matrix from given :math:`M_T` colour array to
    :math:`M_R` colour array using *Finlayson et al. (2015)* method.

    Parameters
    ----------
    M_T
        Test array :math:`M_T` to fit onto array :math:`M_R`.
    M_R
        Reference array the array :math:`M_T` will be colour fitted against.
    degree
        Expanded polynomial degree.
    root_polynomial_expansion
        Whether to use the root-polynomials set for the expansion.

    Returns
    -------
    :class:`numpy.ndarray`
        Colour correction matrix.

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
        polynomial_expansion_Finlayson2015(
            M_T, degree, root_polynomial_expansion
        ),
        M_R,
    )


def matrix_colour_correction_Vandermonde(
    M_T: ArrayLike, M_R: ArrayLike, degree: int = 1
) -> NDArrayFloat:
    """
    Compute a colour correction matrix from given :math:`M_T` colour array to
    :math:`M_R` colour array using *Vandermonde* method.

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
        Colour correction matrix.

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


MATRIX_COLOUR_CORRECTION_METHODS: CanonicalMapping = CanonicalMapping(
    {
        "Cheung 2004": matrix_colour_correction_Cheung2004,
        "Finlayson 2015": matrix_colour_correction_Finlayson2015,
        "Vandermonde": matrix_colour_correction_Vandermonde,
    }
)
MATRIX_COLOUR_CORRECTION_METHODS.__doc__ = """
Supported colour correction matrix methods.

References
----------
:cite:`Cheung2004`, :cite:`Finlayson2015`, :cite:`Westland2004`,
:cite:`Wikipedia2003e`
"""


def matrix_colour_correction(
    M_T: ArrayLike,
    M_R: ArrayLike,
    method: Literal["Cheung 2004", "Finlayson 2015", "Vandermonde"]
    | str = "Cheung 2004",
    **kwargs: Any,
) -> NDArrayFloat:
    """
    Compute a colour correction matrix from given :math:`M_T` colour array to
    :math:`M_R` colour array.

    The resulting colour correction matrix is computed using multiple linear or
    polynomial regression using given method. The purpose of that object
    is for example the matching of two *ColorChecker* colour rendition charts
    together.

    Parameters
    ----------
    M_T
        Test array :math:`M_T` to fit onto array :math:`M_R`.
    M_R
        Reference array the array :math:`M_T` will be colour fitted against.
    method
        Computation method.

    Other Parameters
    ----------------
    degree
        {:func:`colour.characterisation.polynomial_expansion_Finlayson2015`,
        :func:`colour.characterisation.polynomial_expansion_Vandermonde`},
        Expanded polynomial degree, must be one of *[1, 2, 3, 4]* for
        :func:`colour.characterisation.polynomial_expansion_Finlayson2015`
        definition.
    root_polynomial_expansion
        {:func:`colour.characterisation.polynomial_expansion_Finlayson2015`},
        Whether to use the root-polynomials set for the expansion.
    terms
        {:func:`colour.characterisation.matrix_augmented_Cheung2004`},
        Number of terms of the expanded polynomial.

    Returns
    -------
    :class:`numpy.ndarray`
        Colour correction matrix.

    References
    ----------
    :cite:`Cheung2004`, :cite:`Finlayson2015`, :cite:`Westland2004`,
    :cite:`Wikipedia2003e`

    Examples
    --------
    >>> M_T = np.array(
    ...     [
    ...         [0.17224810, 0.09170660, 0.06416938],
    ...         [0.49189645, 0.27802050, 0.21923399],
    ...         [0.10999751, 0.18658946, 0.29938611],
    ...         [0.11666120, 0.14327905, 0.05713804],
    ...         [0.18988879, 0.18227649, 0.36056247],
    ...         [0.12501329, 0.42223442, 0.37027445],
    ...         [0.64785606, 0.22396782, 0.03365194],
    ...         [0.06761093, 0.11076896, 0.39779139],
    ...         [0.49101797, 0.09448929, 0.11623839],
    ...         [0.11622386, 0.04425753, 0.14469986],
    ...         [0.36867946, 0.44545230, 0.06028681],
    ...         [0.61632937, 0.32323906, 0.02437089],
    ...         [0.03016472, 0.06153243, 0.29014596],
    ...         [0.11103655, 0.30553067, 0.08149137],
    ...         [0.41162190, 0.05816656, 0.04845934],
    ...         [0.73339206, 0.53075188, 0.02475212],
    ...         [0.47347718, 0.08834792, 0.30310315],
    ...         [0.00000000, 0.25187016, 0.35062450],
    ...         [0.76809639, 0.78486240, 0.77808297],
    ...         [0.53822392, 0.54307997, 0.54710883],
    ...         [0.35458526, 0.35318419, 0.35524431],
    ...         [0.17976704, 0.18000531, 0.17991488],
    ...         [0.09351417, 0.09510603, 0.09675027],
    ...         [0.03405071, 0.03295077, 0.03702047],
    ...     ]
    ... )
    >>> M_R = np.array(
    ...     [
    ...         [0.15579559, 0.09715755, 0.07514556],
    ...         [0.39113140, 0.25943419, 0.21266708],
    ...         [0.12824821, 0.18463570, 0.31508023],
    ...         [0.12028974, 0.13455659, 0.07408400],
    ...         [0.19368988, 0.21158946, 0.37955964],
    ...         [0.19957425, 0.36085439, 0.40678123],
    ...         [0.48896605, 0.20691688, 0.05816533],
    ...         [0.09775522, 0.16710693, 0.47147724],
    ...         [0.39358649, 0.12233400, 0.10526425],
    ...         [0.10780332, 0.07258529, 0.16151473],
    ...         [0.27502671, 0.34705454, 0.09728099],
    ...         [0.43980441, 0.26880559, 0.05430533],
    ...         [0.05887212, 0.11126272, 0.38552469],
    ...         [0.12705825, 0.25787860, 0.13566464],
    ...         [0.35612929, 0.07933258, 0.05118732],
    ...         [0.48131976, 0.42082843, 0.07120612],
    ...         [0.34665585, 0.15170714, 0.24969804],
    ...         [0.08261116, 0.24588716, 0.48707733],
    ...         [0.66054904, 0.65941137, 0.66376412],
    ...         [0.48051509, 0.47870296, 0.48230082],
    ...         [0.33045354, 0.32904184, 0.33228886],
    ...         [0.18001305, 0.17978567, 0.18004416],
    ...         [0.10283975, 0.10424680, 0.10384975],
    ...         [0.04742204, 0.04772203, 0.04914226],
    ...     ]
    ... )
    >>> matrix_colour_correction(M_T, M_R)  # doctest: +ELLIPSIS
    array([[ 0.6982266...,  0.0307162...,  0.1621042...],
           [ 0.0689349...,  0.6757961...,  0.1643038...],
           [-0.0631495...,  0.0921247...,  0.9713415...]])
    """

    method = validate_method(method, tuple(MATRIX_COLOUR_CORRECTION_METHODS))

    function = MATRIX_COLOUR_CORRECTION_METHODS[method]

    return function(M_T, M_R, **filter_kwargs(function, **kwargs))


def apply_matrix_colour_correction_Cheung2004(
    RGB: ArrayLike,
    CCM: ArrayLike,
    terms: Literal[3, 4, 5, 7, 8, 10, 11, 14, 16, 17, 19, 20, 22, 35] = 3,
) -> NDArrayFloat:
    """
    Apply given colour correction matrix :math:`CCM` computed using
    *Cheung et al. (2004)* method to given *RGB* colourspace array.

    Parameters
    ----------
    RGB
        *RGB* colourspace array to apply the colour correction matrix
        :math:`CCM` to.
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
    >>> apply_matrix_colour_correction_Cheung2004(
    ...     RGB, CCM
    ... )  # doctest: +ELLIPSIS
    array([ 0.1793456...,  0.1003392...,  0.0617218...])
    """

    RGB = as_float_array(RGB)
    shape = RGB.shape

    RGB = np.reshape(RGB, (-1, 3))

    RGB_e = matrix_augmented_Cheung2004(RGB, terms)

    return np.reshape(np.transpose(np.dot(CCM, np.transpose(RGB_e))), shape)


def apply_matrix_colour_correction_Finlayson2015(
    RGB: ArrayLike,
    CCM: ArrayLike,
    degree: Literal[1, 2, 3, 4] = 1,
    root_polynomial_expansion: bool = True,
) -> NDArrayFloat:
    """
    Apply given colour correction matrix :math:`CCM` computed using
    *Finlayson et al. (2015)* method to given *RGB* colourspace array.

    Parameters
    ----------
    RGB
        *RGB* colourspace array to apply the colour correction matrix
        :math:`CCM` to.
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
    >>> apply_matrix_colour_correction_Finlayson2015(
    ...     RGB, CCM
    ... )  # doctest: +ELLIPSIS
    array([ 0.1793456...,  0.1003392...,  0.0617218...])
    """

    RGB = as_float_array(RGB)
    shape = RGB.shape

    RGB = np.reshape(RGB, (-1, 3))

    RGB_e = polynomial_expansion_Finlayson2015(
        RGB, degree, root_polynomial_expansion
    )

    return np.reshape(np.transpose(np.dot(CCM, np.transpose(RGB_e))), shape)


def apply_matrix_colour_correction_Vandermonde(
    RGB: ArrayLike, CCM: ArrayLike, degree: int = 1
) -> NDArrayFloat:
    """
    Apply given colour correction matrix :math:`CCM` computed using
    *Vandermonde* method to given *RGB* colourspace array.

    Parameters
    ----------
    RGB
        *RGB* colourspace array to apply the colour correction matrix
        :math:`CCM` to.
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
    >>> apply_matrix_colour_correction_Vandermonde(
    ...     RGB, CCM
    ... )  # doctest: +ELLIPSIS
    array([ 0.2128689...,  0.1106242...,  0.0362129...])
    """

    RGB = as_float_array(RGB)
    shape = RGB.shape

    RGB = np.reshape(RGB, (-1, 3))

    RGB_e = polynomial_expansion_Vandermonde(RGB, degree)

    return np.reshape(np.transpose(np.dot(CCM, np.transpose(RGB_e))), shape)


APPLY_MATRIX_COLOUR_CORRECTION_METHODS = CanonicalMapping(
    {
        "Cheung 2004": apply_matrix_colour_correction_Cheung2004,
        "Finlayson 2015": apply_matrix_colour_correction_Finlayson2015,
        "Vandermonde": apply_matrix_colour_correction_Vandermonde,
    }
)
APPLY_MATRIX_COLOUR_CORRECTION_METHODS.__doc__ = """
Supported methods to apply a colour correction matrix.

References
----------
:cite:`Cheung2004`, :cite:`Finlayson2015`, :cite:`Westland2004`,
:cite:`Wikipedia2003e`
"""


def apply_matrix_colour_correction(
    RGB: ArrayLike,
    CCM: ArrayLike,
    method: Literal["Cheung 2004", "Finlayson 2015", "Vandermonde"]
    | str = "Cheung 2004",
    **kwargs: Any,
) -> NDArrayFloat:
    """
    Apply given colour correction matrix :math:`CCM` to given *RGB* colourspace
    array.

    Parameters
    ----------
    RGB
        *RGB* colourspace array to apply the colour correction matrix
        :math:`CCM` to.
    CCM
        Colour correction matrix :math:`CCM`.
    method
        Computation method.

    Other Parameters
    ----------------
    degree
        {:func:`colour.characterisation.polynomial_expansion_Finlayson2015`,
        :func:`colour.characterisation.polynomial_expansion_Vandermonde`},
        Expanded polynomial degree, must be one of *[1, 2, 3, 4]* for
        :func:`colour.characterisation.polynomial_expansion_Finlayson2015`
        definition.
    root_polynomial_expansion
        {:func:`colour.characterisation.polynomial_expansion_Finlayson2015`},
        Whether to use the root-polynomials set for the expansion.
    terms
        {:func:`colour.characterisation.matrix_augmented_Cheung2004`},
        Number of terms of the expanded polynomial.

    Returns
    -------
    :class:`numpy.ndarray`
        Colour corrected *RGB* colourspace array.

    References
    ----------
    :cite:`Cheung2004`, :cite:`Finlayson2015`, :cite:`Westland2004`,
    :cite:`Wikipedia2003e`

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
    >>> apply_matrix_colour_correction(RGB, CCM)  # doctest: +ELLIPSIS
    array([ 0.1793456...,  0.1003392...,  0.0617218...])
    """

    method = validate_method(
        method, tuple(APPLY_MATRIX_COLOUR_CORRECTION_METHODS)
    )

    function = APPLY_MATRIX_COLOUR_CORRECTION_METHODS[method]

    return function(RGB, CCM, **filter_kwargs(function, **kwargs))


def colour_correction_Cheung2004(
    RGB: ArrayLike,
    M_T: ArrayLike,
    M_R: ArrayLike,
    terms: Literal[3, 4, 5, 7, 8, 10, 11, 14, 16, 17, 19, 20, 22, 35] = 3,
) -> NDArrayFloat:
    """
    Perform colour correction of given *RGB* colourspace array using the
    colour correction matrix from given :math:`M_T` colour array to
    :math:`M_R` colour array using *Cheung et al. (2004)* method.

    Parameters
    ----------
    RGB
        *RGB* colourspace array to colour correct.
    M_T
        Test array :math:`M_T` to fit onto array :math:`M_R`.
    M_R
        Reference array the array :math:`M_T` will be colour fitted against.
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
    array([ 0.1793456...,  0.1003392...,  0.0617218...])
    """

    return apply_matrix_colour_correction_Cheung2004(
        RGB, matrix_colour_correction_Cheung2004(M_T, M_R, terms), terms
    )


def colour_correction_Finlayson2015(
    RGB: ArrayLike,
    M_T: ArrayLike,
    M_R: ArrayLike,
    degree: Literal[1, 2, 3, 4] = 1,
    root_polynomial_expansion: bool = True,
) -> NDArrayFloat:
    """
    Perform colour correction of given *RGB* colourspace array using the
    colour correction matrix from given :math:`M_T` colour array to
    :math:`M_R` colour array using *Finlayson et al. (2015)* method.

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
    >>> prng = np.random.RandomState(2)
    >>> M_T = prng.random_sample((24, 3))
    >>> M_R = M_T + (prng.random_sample((24, 3)) - 0.5) * 0.5
    >>> colour_correction_Finlayson2015(RGB, M_T, M_R)  # doctest: +ELLIPSIS
    array([ 0.1793456...,  0.1003392...,  0.0617218...])
    """

    return apply_matrix_colour_correction_Finlayson2015(
        RGB,
        matrix_colour_correction_Finlayson2015(
            M_T, M_R, degree, root_polynomial_expansion
        ),
        degree,
        root_polynomial_expansion,
    )


def colour_correction_Vandermonde(
    RGB: ArrayLike, M_T: ArrayLike, M_R: ArrayLike, degree: int = 1
) -> NDArrayFloat:
    """
    Perform colour correction of given *RGB* colourspace array using the
    colour correction matrix from given :math:`M_T` colour array to
    :math:`M_R` colour array using *Vandermonde* method.

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
    array([ 0.2128689...,  0.1106242...,  0.036213 ...])
    """

    return apply_matrix_colour_correction_Vandermonde(
        RGB, matrix_colour_correction_Vandermonde(M_T, M_R, degree), degree
    )


COLOUR_CORRECTION_METHODS = CanonicalMapping(
    {
        "Cheung 2004": colour_correction_Cheung2004,
        "Finlayson 2015": colour_correction_Finlayson2015,
        "Vandermonde": colour_correction_Vandermonde,
    }
)
COLOUR_CORRECTION_METHODS.__doc__ = """
Supported colour correction methods.

References
----------
:cite:`Cheung2004`, :cite:`Finlayson2015`, :cite:`Westland2004`,
:cite:`Wikipedia2003e`
"""


def colour_correction(
    RGB: ArrayLike,
    M_T: ArrayLike,
    M_R: ArrayLike,
    method: Literal["Cheung 2004", "Finlayson 2015", "Vandermonde"]
    | str = "Cheung 2004",
    **kwargs: Any,
) -> NDArrayFloat:
    """
    Perform colour correction of given *RGB* colourspace array using the
    colour correction matrix from given :math:`M_T` colour array to
    :math:`M_R` colour array.

    Parameters
    ----------
    RGB
        *RGB* colourspace array to colour correct.
    M_T
        Test array :math:`M_T` to fit onto array :math:`M_R`.
    M_R
        Reference array the array :math:`M_T` will be colour fitted against.
    method
        Computation method.

    Other Parameters
    ----------------
    degree
        {:func:`colour.characterisation.polynomial_expansion_Finlayson2015`,
        :func:`colour.characterisation.polynomial_expansion_Vandermonde`},
        Expanded polynomial degree, must be one of *[1, 2, 3, 4]* for
        :func:`colour.characterisation.polynomial_expansion_Finlayson2015`
        definition.
    root_polynomial_expansion
        {:func:`colour.characterisation.polynomial_expansion_Finlayson2015`},
        Whether to use the root-polynomials set for the expansion.
    terms
        {:func:`colour.characterisation.matrix_augmented_Cheung2004`},
        Number of terms of the expanded polynomial.

    Returns
    -------
    :class:`numpy.ndarray`
        Colour corrected *RGB* colourspace array.

    References
    ----------
    :cite:`Cheung2004`, :cite:`Finlayson2015`, :cite:`Westland2004`,
    :cite:`Wikipedia2003e`

    Examples
    --------
    >>> RGB = np.array([0.17224810, 0.09170660, 0.06416938])
    >>> M_T = np.array(
    ...     [
    ...         [0.17224810, 0.09170660, 0.06416938],
    ...         [0.49189645, 0.27802050, 0.21923399],
    ...         [0.10999751, 0.18658946, 0.29938611],
    ...         [0.11666120, 0.14327905, 0.05713804],
    ...         [0.18988879, 0.18227649, 0.36056247],
    ...         [0.12501329, 0.42223442, 0.37027445],
    ...         [0.64785606, 0.22396782, 0.03365194],
    ...         [0.06761093, 0.11076896, 0.39779139],
    ...         [0.49101797, 0.09448929, 0.11623839],
    ...         [0.11622386, 0.04425753, 0.14469986],
    ...         [0.36867946, 0.44545230, 0.06028681],
    ...         [0.61632937, 0.32323906, 0.02437089],
    ...         [0.03016472, 0.06153243, 0.29014596],
    ...         [0.11103655, 0.30553067, 0.08149137],
    ...         [0.41162190, 0.05816656, 0.04845934],
    ...         [0.73339206, 0.53075188, 0.02475212],
    ...         [0.47347718, 0.08834792, 0.30310315],
    ...         [0.00000000, 0.25187016, 0.35062450],
    ...         [0.76809639, 0.78486240, 0.77808297],
    ...         [0.53822392, 0.54307997, 0.54710883],
    ...         [0.35458526, 0.35318419, 0.35524431],
    ...         [0.17976704, 0.18000531, 0.17991488],
    ...         [0.09351417, 0.09510603, 0.09675027],
    ...         [0.03405071, 0.03295077, 0.03702047],
    ...     ]
    ... )
    >>> M_R = np.array(
    ...     [
    ...         [0.15579559, 0.09715755, 0.07514556],
    ...         [0.39113140, 0.25943419, 0.21266708],
    ...         [0.12824821, 0.18463570, 0.31508023],
    ...         [0.12028974, 0.13455659, 0.07408400],
    ...         [0.19368988, 0.21158946, 0.37955964],
    ...         [0.19957425, 0.36085439, 0.40678123],
    ...         [0.48896605, 0.20691688, 0.05816533],
    ...         [0.09775522, 0.16710693, 0.47147724],
    ...         [0.39358649, 0.12233400, 0.10526425],
    ...         [0.10780332, 0.07258529, 0.16151473],
    ...         [0.27502671, 0.34705454, 0.09728099],
    ...         [0.43980441, 0.26880559, 0.05430533],
    ...         [0.05887212, 0.11126272, 0.38552469],
    ...         [0.12705825, 0.25787860, 0.13566464],
    ...         [0.35612929, 0.07933258, 0.05118732],
    ...         [0.48131976, 0.42082843, 0.07120612],
    ...         [0.34665585, 0.15170714, 0.24969804],
    ...         [0.08261116, 0.24588716, 0.48707733],
    ...         [0.66054904, 0.65941137, 0.66376412],
    ...         [0.48051509, 0.47870296, 0.48230082],
    ...         [0.33045354, 0.32904184, 0.33228886],
    ...         [0.18001305, 0.17978567, 0.18004416],
    ...         [0.10283975, 0.10424680, 0.10384975],
    ...         [0.04742204, 0.04772203, 0.04914226],
    ...     ]
    ... )
    >>> colour_correction(RGB, M_T, M_R)  # doctest: +ELLIPSIS
    array([ 0.1334872...,  0.0843921...,  0.0599014...])
    """

    method = validate_method(method, tuple(COLOUR_CORRECTION_METHODS))

    function = COLOUR_CORRECTION_METHODS[method]

    return function(RGB, M_T, M_R, **filter_kwargs(function, **kwargs))
