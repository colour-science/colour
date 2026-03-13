from __future__ import annotations

from .centore2014 import (
    CCS_ILLUMINANT_MUNSELL,
    ILLUMINANT_NAME_MUNSELL,
    MUNSELL_COLOUR_EXTENDED_FORMAT,
    MUNSELL_COLOUR_FORMAT,
    MUNSELL_COLOUR_PATTERN,
    MUNSELL_GRAY_EXTENDED_FORMAT,
    MUNSELL_GRAY_FORMAT,
    MUNSELL_GRAY_PATTERN,
    MUNSELL_HUE_LETTER_CODES,
    LCHab_to_munsell_specification,
    bounding_hues_from_renotation,
    hue_angle_to_hue,
    hue_to_ASTM_hue,
    hue_to_hue_angle,
    interpolation_method_from_renotation_ovoid,
    is_grey_munsell_colour,
    is_specification_in_renotation,
    maximum_chroma_from_renotation,
    munsell_colour_to_munsell_specification,
    munsell_colour_to_xyY_Centore2014,
    munsell_specification_to_munsell_colour,
    munsell_specification_to_xy,
    munsell_specification_to_xyY_Centore2014,
    normalise_munsell_specification,
    parse_munsell_colour,
    xy_from_renotation_ovoid,
    xyY_from_renotation,
    xyY_to_munsell_colour_Centore2014,
    xyY_to_munsell_specification_Centore2014,
)
from .onnx import (
    ONNX_MODELS_FROM_XYY,
    ONNX_MODELS_TO_XYY,
    munsell_colour_to_xyY_Onnx,
    munsell_specification_to_xyY_Onnx,
    xyY_to_munsell_colour_Onnx,
    xyY_to_munsell_specification_Onnx,
)
from .value import (
    MUNSELL_VALUE_METHODS,
    munsell_value,
    munsell_value_ASTMD1535,
    munsell_value_Ladd1955,
    munsell_value_McCamy1987,
    munsell_value_Moon1943,
    munsell_value_Munsell1933,
    munsell_value_Priest1920,
    munsell_value_Saunderson1944,
)

# isort: split

import typing

if typing.TYPE_CHECKING:
    from colour.hints import (
        ArrayLike,
        Domain1,
        Literal,
        NDArrayFloat,
        NDArrayStr,
        Range1,
    )

from colour.utilities import (
    CanonicalMapping,
    filter_kwargs,
    validate_method,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "MUNSELL_GRAY_PATTERN",
    "MUNSELL_COLOUR_PATTERN",
    "MUNSELL_GRAY_FORMAT",
    "MUNSELL_COLOUR_FORMAT",
    "MUNSELL_GRAY_EXTENDED_FORMAT",
    "MUNSELL_COLOUR_EXTENDED_FORMAT",
    "MUNSELL_HUE_LETTER_CODES",
    "ILLUMINANT_NAME_MUNSELL",
    "CCS_ILLUMINANT_MUNSELL",
]
__all__ += [
    "munsell_value_Priest1920",
    "munsell_value_Munsell1933",
    "munsell_value_Moon1943",
    "munsell_value_Saunderson1944",
    "munsell_value_Ladd1955",
    "munsell_value_McCamy1987",
    "munsell_value_ASTMD1535",
    "MUNSELL_VALUE_METHODS",
    "munsell_value",
]
__all__ += [
    "munsell_specification_to_xyY_Centore2014",
    "munsell_colour_to_xyY_Centore2014",
    "xyY_to_munsell_specification_Centore2014",
    "xyY_to_munsell_colour_Centore2014",
]
__all__ += [
    "ONNX_MODELS_TO_XYY",
    "ONNX_MODELS_FROM_XYY",
    "munsell_specification_to_xyY_Onnx",
    "munsell_colour_to_xyY_Onnx",
    "xyY_to_munsell_specification_Onnx",
    "xyY_to_munsell_colour_Onnx",
]
__all__ += [
    "MUNSELL_SPECIFICATION_TO_XYY_METHODS",
    "MUNSELL_COLOUR_TO_XYY_METHODS",
    "XYY_TO_MUNSELL_SPECIFICATION_METHODS",
    "XYY_TO_MUNSELL_COLOUR_METHODS",
    "munsell_specification_to_xyY",
    "munsell_colour_to_xyY",
    "xyY_to_munsell_specification",
    "xyY_to_munsell_colour",
]
__all__ += [
    "parse_munsell_colour",
    "is_grey_munsell_colour",
    "normalise_munsell_specification",
    "munsell_colour_to_munsell_specification",
    "munsell_specification_to_munsell_colour",
    "xyY_from_renotation",
    "is_specification_in_renotation",
    "bounding_hues_from_renotation",
    "hue_to_hue_angle",
    "hue_angle_to_hue",
    "hue_to_ASTM_hue",
    "interpolation_method_from_renotation_ovoid",
    "xy_from_renotation_ovoid",
    "LCHab_to_munsell_specification",
    "maximum_chroma_from_renotation",
    "munsell_specification_to_xy",
]

MUNSELL_SPECIFICATION_TO_XYY_METHODS: CanonicalMapping = CanonicalMapping(
    {
        "Centore 2014": munsell_specification_to_xyY_Centore2014,
        "ONNX": munsell_specification_to_xyY_Onnx,
    }
)
"""Supported *Munsell* specification to *CIE xyY* conversion methods."""

MUNSELL_COLOUR_TO_XYY_METHODS: CanonicalMapping = CanonicalMapping(
    {
        "Centore 2014": munsell_colour_to_xyY_Centore2014,
        "ONNX": munsell_colour_to_xyY_Onnx,
    }
)
"""Supported *Munsell* colour to *CIE xyY* conversion methods."""

XYY_TO_MUNSELL_SPECIFICATION_METHODS: CanonicalMapping = CanonicalMapping(
    {
        "Centore 2014": xyY_to_munsell_specification_Centore2014,
        "ONNX": xyY_to_munsell_specification_Onnx,
    }
)
"""Supported *CIE xyY* to *Munsell* specification conversion methods."""

XYY_TO_MUNSELL_COLOUR_METHODS: CanonicalMapping = CanonicalMapping(
    {
        "Centore 2014": xyY_to_munsell_colour_Centore2014,
        "ONNX": xyY_to_munsell_colour_Onnx,
    }
)
"""Supported *CIE xyY* to *Munsell* colour conversion methods."""


def munsell_specification_to_xyY(
    specification: ArrayLike,
    method: Literal["Centore 2014", "ONNX"] | str = "Centore 2014",
) -> NDArrayFloat:
    """
    Convert specified *Munsell* *Colorlab* specification to *CIE xyY*
    colourspace.

    Parameters
    ----------
    specification
        *Munsell* *Colorlab* specification.
    method
        Computation method.

    Returns
    -------
    :class:`numpy.NDArrayFloat`
        *CIE xyY* colourspace array.

    Notes
    -----
    +-------------------+-----------------------+---------------+
    | **Domain**        | **Scale - Reference** | **Scale - 1** |
    +===================+=======================+===============+
    | ``specification`` | ``hue``    : 10       | 1             |
    |                   |                       |               |
    |                   | ``value``  : 10       | 1             |
    |                   |                       |               |
    |                   | ``chroma`` : 50       | 1             |
    |                   |                       |               |
    |                   | ``code``   : 10       | 1             |
    +-------------------+-----------------------+---------------+

    +-------------------+-----------------------+---------------+
    | **Range**         | **Scale - Reference** | **Scale - 1** |
    +===================+=======================+===============+
    | ``xyY``           | 1                     | 1             |
    +-------------------+-----------------------+---------------+

    References
    ----------
    :cite:`Centore2014m`

    Examples
    --------
    >>> import numpy as np
    >>> munsell_specification_to_xyY(np.array([2.1, 8.0, 17.9, 4]))
    ... # doctest: +ELLIPSIS
    array([0.4400632..., 0.5522428..., 0.5761962...])
    >>> munsell_specification_to_xyY(np.array([np.nan, 8.9, np.nan, np.nan]))
    ... # doctest: +ELLIPSIS
    array([0.31006  , 0.31616  , 0.7461345...])
    """

    method = validate_method(method, tuple(MUNSELL_SPECIFICATION_TO_XYY_METHODS))

    return MUNSELL_SPECIFICATION_TO_XYY_METHODS[method](specification)


def munsell_colour_to_xyY(
    munsell_colour: ArrayLike,
    method: Literal["Centore 2014", "ONNX"] | str = "Centore 2014",
) -> Range1:
    """
    Convert the specified *Munsell* colour to *CIE xyY* colourspace.

    Parameters
    ----------
    munsell_colour
        *Munsell* colour notation formatted as "H V/C" where H is hue,
        V is value, and C is chroma.
    method
        Computation method.

    Returns
    -------
    :class:`numpy.NDArrayFloat`
        *CIE xyY* colourspace array.

    Notes
    -----
    +-----------+-----------------------+---------------+
    | **Range** | **Scale - Reference** | **Scale - 1** |
    +===========+=======================+===============+
    | ``xyY``   | 1                     | 1             |
    +-----------+-----------------------+---------------+

    References
    ----------
    :cite:`Centorea`, :cite:`Centore2012a`

    Examples
    --------
    >>> munsell_colour_to_xyY("4.2YR 8.1/5.3")  # doctest: +ELLIPSIS
    array([0.3873694..., 0.3575165..., 0.59362   ])
    >>> munsell_colour_to_xyY("N8.9")  # doctest: +ELLIPSIS
    array([0.31006  , 0.31616  , 0.7461345...])
    """

    method = validate_method(method, tuple(MUNSELL_COLOUR_TO_XYY_METHODS))

    return MUNSELL_COLOUR_TO_XYY_METHODS[method](munsell_colour)


def xyY_to_munsell_specification(
    xyY: ArrayLike,
    method: Literal["Centore 2014", "ONNX"] | str = "Centore 2014",
) -> NDArrayFloat:
    """
    Convert from *CIE xyY* colourspace to *Munsell* *Colorlab*
    specification.

    Parameters
    ----------
    xyY
        *CIE xyY* colourspace array.
    method
        Computation method.

    Returns
    -------
    :class:`numpy.NDArrayFloat`
        *Munsell* *Colorlab* specification.

    Raises
    ------
    ValueError
        If the specified *CIE xyY* colourspace array is not within
        MacAdam limits.
    RuntimeError
        If the maximum iterations count has been reached without
        converging to a result.

    Notes
    -----
    +-------------------+-----------------------+---------------+
    | **Domain**        | **Scale - Reference** | **Scale - 1** |
    +===================+=======================+===============+
    | ``xyY``           | 1                     | 1             |
    +-------------------+-----------------------+---------------+

    +-------------------+-----------------------+---------------+
    | **Range**         | **Scale - Reference** | **Scale - 1** |
    +===================+=======================+===============+
    | ``specification`` | ``hue``    : 10       | 1             |
    |                   |                       |               |
    |                   | ``value``  : 10       | 1             |
    |                   |                       |               |
    |                   | ``chroma`` : 50       | 1             |
    |                   |                       |               |
    |                   | ``code``   : 10       | 1             |
    +-------------------+-----------------------+---------------+

    References
    ----------
    :cite:`Centore2014p`

    Examples
    --------
    >>> import numpy as np
    >>> xyY = np.array([0.38736945, 0.35751656, 0.59362000])
    >>> xyY_to_munsell_specification(xyY)  # doctest: +ELLIPSIS
    array([4.2000019..., 8.0999999..., 5.2999996..., 6.        ])
    """

    method = validate_method(method, tuple(XYY_TO_MUNSELL_SPECIFICATION_METHODS))

    return XYY_TO_MUNSELL_SPECIFICATION_METHODS[method](xyY)


def xyY_to_munsell_colour(
    xyY: Domain1,
    hue_decimals: int = 1,
    value_decimals: int = 1,
    chroma_decimals: int = 1,
    method: Literal["Centore 2014", "ONNX"] | str = "Centore 2014",
) -> str | NDArrayStr:
    """
    Convert from *CIE xyY* colourspace to *Munsell* colour notation.

    Parameters
    ----------
    xyY
        *CIE xyY* colourspace array representing chromaticity coordinates
        and luminance.
    hue_decimals
        Number of decimal places for formatting the hue component.
    value_decimals
        Number of decimal places for formatting the value component.
    chroma_decimals
        Number of decimal places for formatting the chroma component.
    method
        Computation method.

    Returns
    -------
    :class:`str` or :class:`numpy.NDArrayFloat`
        *Munsell* colour notation formatted as "H V/C" where H is hue,
        V is value, and C is chroma.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``xyY``    | 1                     | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Centorea`, :cite:`Centore2012a`

    Examples
    --------
    >>> import numpy as np
    >>> xyY = np.array([0.38736945, 0.35751656, 0.59362000])
    >>> xyY_to_munsell_colour(xyY)
    '4.2YR 8.1/5.3'
    """

    method = validate_method(method, tuple(XYY_TO_MUNSELL_COLOUR_METHODS))

    return XYY_TO_MUNSELL_COLOUR_METHODS[method](
        xyY,
        **filter_kwargs(
            XYY_TO_MUNSELL_COLOUR_METHODS[method],
            hue_decimals=hue_decimals,
            value_decimals=value_decimals,
            chroma_decimals=chroma_decimals,
        ),
    )
