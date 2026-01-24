"""
Colour Correction
=================

Define objects for colour correction, including methods for colour matching
between images:

-   :func:`colour.characterisation.matrix_augmented_Cheung2004`: Perform
    polynomial expansion using *Cheung, Westland, Connah and Ripamonti (2004)*
    method.
-   :func:`colour.characterisation.polynomial_expansion_Finlayson2015`:
    Perform polynomial expansion using *Finlayson, MacKiewicz and Hurlbert
    (2015)* method.
-   :func:`colour.characterisation.polynomial_expansion_Vandermonde`: Perform
    polynomial expansion using *Vandermonde* method.
-   :attr:`colour.POLYNOMIAL_EXPANSION_METHODS`: Supported polynomial
    expansion methods.
-   :func:`colour.polynomial_expansion`: Perform polynomial expansion of
    :math:`a` array.
-   :func:`colour.characterisation.matrix_colour_correction_Cheung2004`:
    Compute colour correction matrix using *Cheung et al. (2004)* method.
-   :func:`colour.characterisation.matrix_colour_correction_Finlayson2015`:
    Compute colour correction matrix using *Finlayson et al. (2015)* method.
-   :func:`colour.characterisation.matrix_colour_correction_Vandermonde`:
    Compute colour correction matrix using *Vandermonde* method.
-   :attr:`colour.MATRIX_COLOUR_CORRECTION_METHODS`: Supported colour
    correction matrix methods.
-   :func:`colour.matrix_colour_correction`: Compute colour correction matrix
    from :math:`M_T` colour array to :math:`M_R` colour array.
-   :func:`colour.apply_matrix_colour_correction_Cheung2004`: Apply colour
    correction matrix computed using *Cheung et al. (2004)* method.
-   :func:`colour.apply_matrix_colour_correction_Finlayson2015`: Apply colour
    correction matrix computed using *Finlayson et al. (2015)* method.
-   :func:`colour.apply_matrix_colour_correction_Vandermonde`: Apply colour
    correction matrix computed using *Vandermonde* method.
-   :attr:`colour.APPLY_MATRIX_COLOUR_CORRECTION_METHODS`: Supported methods
    to apply colour correction matrices.
-   :func:`colour.apply_matrix_colour_correction`: Apply colour correction
    matrix.
-   :func:`colour.characterisation.colour_correction_Cheung2004`: Perform
    colour correction using *Cheung et al. (2004)* method.
-   :func:`colour.characterisation.colour_correction_Finlayson2015`: Perform
    colour correction using *Finlayson et al. (2015)* method.
-   :func:`colour.characterisation.colour_correction_Vandermonde`: Perform
    colour correction using *Vandermonde* method.
-   :func:`colour.characterisation.apply_tps3d`: Apply pre-fitted *TPS-3D*
    colour correction to an *RGB* array.
-   :func:`colour.characterisation.colour_correction_TPS3D`: Perform colour
    correction using *TPS-3D* (Thin-Plate Spline) warping method.
-   :attr:`colour.COLOUR_CORRECTION_METHODS`: Supported colour correction
    methods.
-   :func:`colour.colour_correction`: Perform colour correction of *RGB*
    colourspace array using colour correction matrix from :math:`M_T` colour
    array to :math:`M_R` colour array.

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
-   :cite:`Menesatti2012` : Menesatti, P., Angelini, C., Pallottino, F.,
    Antonucci, F., Aguzzi, J., & Costa, C. (2012). RGB Color Calibration for
    Quantitative Image Analysis: The “3D Thin-Plate Spline” Warping Approach.
    Sensors, 12(6), 7063-7079. doi:10.3390/s120607063
-   :cite:`Westland2004` : Westland, S., & Ripamonti, C. (2004). Table 8.2. In
    Computational Colour Science Using MATLAB (1st ed., p. 137). John Wiley &
    Sons, Ltd. doi:10.1002/0470020326
-   :cite:`Wikipedia2003e` : Wikipedia. (2003). Vandermonde matrix. Retrieved
    May 2, 2018, from https://en.wikipedia.org/wiki/Vandermonde_matrix
"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import Any, ArrayLike, Literal, NDArrayFloat

from colour.utilities import CanonicalMapping, filter_kwargs, validate_method

from .cheung2004 import (
    apply_matrix_colour_correction_Cheung2004,
    colour_correction_Cheung2004,
    matrix_augmented_Cheung2004,
    matrix_colour_correction_Cheung2004,
)
from .finlayson2015 import (
    apply_matrix_colour_correction_Finlayson2015,
    colour_correction_Finlayson2015,
    matrix_colour_correction_Finlayson2015,
    polynomial_expansion_Finlayson2015,
)
from .tps3d import (
    apply_tps3d,
    colour_correction_TPS3D,
)
from .vandermonde import (
    apply_matrix_colour_correction_Vandermonde,
    colour_correction_Vandermonde,
    matrix_colour_correction_Vandermonde,
    polynomial_expansion_Vandermonde,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "apply_matrix_colour_correction_Cheung2004",
    "colour_correction_Cheung2004",
    "matrix_augmented_Cheung2004",
    "matrix_colour_correction_Cheung2004",
]
__all__ += [
    "apply_matrix_colour_correction_Finlayson2015",
    "colour_correction_Finlayson2015",
    "matrix_colour_correction_Finlayson2015",
    "polynomial_expansion_Finlayson2015",
]
__all__ += [
    "apply_tps3d",
    "colour_correction_TPS3D",
]
__all__ += [
    "apply_matrix_colour_correction_Vandermonde",
    "colour_correction_Vandermonde",
    "matrix_colour_correction_Vandermonde",
    "polynomial_expansion_Vandermonde",
]


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
    method: (
        Literal["Cheung 2004", "Finlayson 2015", "Vandermonde"] | str
    ) = "Cheung 2004",
    **kwargs: Any,
) -> NDArrayFloat:
    """
    Perform polynomial expansion of the :math:`a` array.

    Parameters
    ----------
    a
        Array to expand using polynomial expansion.
    method
        Computation method for the polynomial expansion.

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
        Polynomial-expanded :math:`a` array.

    References
    ----------
    :cite:`Cheung2004`, :cite:`Finlayson2015`, :cite:`Westland2004`,
    :cite:`Wikipedia2003e`

    Examples
    --------
    >>> import numpy as np
    >>> RGB = np.array([0.17224810, 0.09170660, 0.06416938])
    >>> polynomial_expansion(RGB)  # doctest: +ELLIPSIS
    array([0.1722481..., 0.0917066..., 0.0641693...])
    >>> polynomial_expansion(RGB, "Cheung 2004", terms=5)  # doctest: +ELLIPSIS
    array([0.1722481..., 0.0917066..., 0.0641693..., 0.0010136..., 1...])
    """

    method = validate_method(method, tuple(POLYNOMIAL_EXPANSION_METHODS))

    function = POLYNOMIAL_EXPANSION_METHODS[method]

    return function(a, **filter_kwargs(function, **kwargs))


MATRIX_COLOUR_CORRECTION_METHODS: CanonicalMapping = CanonicalMapping(
    {
        "Cheung 2004": matrix_colour_correction_Cheung2004,
        "Finlayson 2015": matrix_colour_correction_Finlayson2015,
        "Vandermonde": matrix_colour_correction_Vandermonde,
    }
)
MATRIX_COLOUR_CORRECTION_METHODS.__doc__ = """
Supported colour correction matrix computation methods.

References
----------
:cite:`Cheung2004`, :cite:`Finlayson2015`, :cite:`Westland2004`,
:cite:`Wikipedia2003e`
"""


def matrix_colour_correction(
    M_T: ArrayLike,
    M_R: ArrayLike,
    method: (
        Literal["Cheung 2004", "Finlayson 2015", "Vandermonde"] | str
    ) = "Cheung 2004",
    **kwargs: Any,
) -> NDArrayFloat:
    """
    Compute a colour correction matrix from :math:`M_T` colour array to
    :math:`M_R` colour array.

    Compute the colour correction matrix using multiple linear or polynomial
    regression with the specified method. The resulting matrix enables colour
    matching between two arrays, such as matching two *ColorChecker* colour
    rendition charts together.

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
        {:func:`colour.characterisation.matrix_colour_correction_Finlayson2015`,
        :func:`colour.characterisation.matrix_colour_correction_Vandermonde`},
        Expanded polynomial degree, must be one of *[1, 2, 3, 4]* for
        :func:`colour.characterisation.matrix_colour_correction_Finlayson2015`
        definition.
    root_polynomial_expansion
        {:func:`colour.characterisation.matrix_colour_correction_Finlayson2015`},
        Whether to use the root-polynomials set for the expansion.
    terms
        {:func:`colour.characterisation.matrix_colour_correction_Cheung2004`},
        Number of terms of the expanded polynomial.

    Returns
    -------
    :class:`numpy.ndarray`
        Colour correction matrix mapping expanded test colours to reference
        colours.

    References
    ----------
    :cite:`Cheung2004`, :cite:`Finlayson2015`, :cite:`Westland2004`,
    :cite:`Wikipedia2003e`

    Examples
    --------
    >>> import numpy as np
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
    method: (
        Literal["Cheung 2004", "Finlayson 2015", "Vandermonde"] | str
    ) = "Cheung 2004",
    **kwargs: Any,
) -> NDArrayFloat:
    """
    Apply colour correction matrix :math:`CCM` to the specified *RGB*
    colourspace array.

    The colour correction matrix transforms the input *RGB* values through
    polynomial expansion and matrix multiplication to produce colour
    corrected output values. The computation method determines the
    polynomial expansion approach used before applying the matrix.

    Parameters
    ----------
    RGB
        *RGB* colourspace array to which the colour correction matrix
        :math:`CCM` is applied.
    CCM
        Colour correction matrix :math:`CCM`.
    method
        Computation method.

    Other Parameters
    ----------------
    degree
        {:func:`colour.characterisation.apply_matrix_colour_correction_Finlayson2015`,
        :func:`colour.characterisation.apply_matrix_colour_correction_Vandermonde`},
        Expanded polynomial degree, must be one of *[1, 2, 3, 4]* for
        :func:`colour.characterisation.apply_matrix_colour_correction_Finlayson2015`
        definition.
    root_polynomial_expansion
        {:func:`colour.characterisation.apply_matrix_colour_correction_Finlayson2015`},
        Whether to use the root-polynomials set for the expansion.
    terms
        {:func:`colour.characterisation.apply_matrix_colour_correction_Cheung2004`},
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
    >>> import numpy as np
    >>> RGB = np.array([0.17224810, 0.09170660, 0.06416938])
    >>> CCM = np.array(
    ...     [
    ...         [1.05263767, 0.13780789, -0.22763399],
    ...         [0.07395843, 1.02939945, -0.1060115],
    ...         [0.05725508, -0.20526336, 1.10151945],
    ...     ]
    ... )
    >>> apply_matrix_colour_correction(RGB, CCM)  # doctest: +ELLIPSIS
    array([0.1793456..., 0.1003392..., 0.0617218...])
    """

    method = validate_method(method, tuple(APPLY_MATRIX_COLOUR_CORRECTION_METHODS))

    function = APPLY_MATRIX_COLOUR_CORRECTION_METHODS[method]

    return function(RGB, CCM, **filter_kwargs(function, **kwargs))


COLOUR_CORRECTION_METHODS = CanonicalMapping(
    {
        "Cheung 2004": colour_correction_Cheung2004,
        "Finlayson 2015": colour_correction_Finlayson2015,
        "Vandermonde": colour_correction_Vandermonde,
        "TPS-3D": colour_correction_TPS3D,
    }
)
COLOUR_CORRECTION_METHODS.__doc__ = """
Define the supported colour correction methods.

References
----------
:cite:`Cheung2004`, :cite:`Finlayson2015`, :cite:`Menesatti2012`,
:cite:`Westland2004`, :cite:`Wikipedia2003e`
"""


def colour_correction(
    RGB: ArrayLike,
    M_T: ArrayLike,
    M_R: ArrayLike,
    method: (
        Literal["Cheung 2004", "Finlayson 2015", "Vandermonde", "TPS-3D"] | str
    ) = "Cheung 2004",
    **kwargs: Any,
) -> NDArrayFloat:
    """
    Perform colour correction of *RGB* colourspace array using the colour
    correction matrix from :math:`M_T` colour array to :math:`M_R` colour
    array.

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
        {:func:`colour.characterisation.colour_correction_Finlayson2015`,
        :func:`colour.characterisation.colour_correction_Vandermonde`},
        Expanded polynomial degree, must be one of *[1, 2, 3, 4]* for
        :func:`colour.characterisation.colour_correction_Finlayson2015`
        definition.
    root_polynomial_expansion
        {:func:`colour.characterisation.colour_correction_Finlayson2015`},
        Whether to use the root-polynomials set for the expansion.
    terms
        {:func:`colour.characterisation.colour_correction_Cheung2004`},
        Number of terms of the expanded polynomial.

    Returns
    -------
    :class:`numpy.ndarray`
        Colour corrected *RGB* colourspace array.

    References
    ----------
    :cite:`Cheung2004`, :cite:`Finlayson2015`, :cite:`Menesatti2012`,
    :cite:`Westland2004`, :cite:`Wikipedia2003e`

    Examples
    --------
    >>> import numpy as np
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
    array([0.1334872..., 0.0843921..., 0.0599014...])
    """

    method = validate_method(method, tuple(COLOUR_CORRECTION_METHODS))

    function = COLOUR_CORRECTION_METHODS[method]

    return function(RGB, M_T, M_R, **filter_kwargs(function, **kwargs))


__all__ += [
    "POLYNOMIAL_EXPANSION_METHODS",
    "polynomial_expansion",
    "MATRIX_COLOUR_CORRECTION_METHODS",
    "matrix_colour_correction",
    "APPLY_MATRIX_COLOUR_CORRECTION_METHODS",
    "apply_matrix_colour_correction",
    "COLOUR_CORRECTION_METHODS",
    "colour_correction",
]
