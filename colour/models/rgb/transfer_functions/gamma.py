"""
Gamma Colour Component Transfer Function
========================================

Define the gamma encoding / decoding colour component transfer function
related objects:

- :func:`colour.gamma_function`
"""

from __future__ import annotations

import numpy as np

from colour.algebra import spow
from colour.hints import ArrayLike, Literal, NDArrayFloat
from colour.utilities import as_float, as_float_array, validate_method

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "gamma_function",
    "GammaFunction",
]

NegativeNumberHandlingType = (
    Literal["Clamp", "Indeterminate", "Mirror", "Preserve"] | str
)


class GammaFunction:
    """Provides an object oriented interface to contain optional parameters for
    an underlying :func:gamma_function call. Useful for providing both a simpler
    and constructed api for gamma_function as well as allowing for control flow.
    """

    def __init__(
        self,
        exponent: float = 1,
        negative_number_handling: NegativeNumberHandlingType = "Indeterminate",
    ):
        """
        Construct an object oriented interface to contain optional parameters for
        an underlying :func:gamma_function call. Useful for providing both a simpler
        and constructed api for gamma_function as well as allowing for control flow.

        Parameters
        ----------
        exponent : float, optional
            The exponent value in a^b, by default 1
        negative_number_handling : NegativeNumberHandlingType, optional
            Defines the behavior for negative number handling, by default
            "Indeterminate"

        See Also
        --------
        :func:gamma_function
        """
        self._exponent = exponent
        self._negative_number_handling = negative_number_handling

    @property
    def exponent(self) -> float:
        """The exponent, b, in the function a^b

        Returns
        -------
        float
        """
        return self._exponent

    @property
    def negative_number_handling(self) -> NegativeNumberHandlingType:
        """How to treat negative numbers. See also :func:gamma_function

        Returns
        -------
        NegativeNumberHandlingType
            See also :func:gamma_function
        """
        return self._negative_number_handling

    def __call__(self, a: ArrayLike):
        """Calculate a typical encoding / decoding function on `a`. Representative
        of the function a ^ b where b is determined by the instance value of
        `exponent` and negative handling behavior is defined by the instance
        value `negative_number_handling`. See also :func:gamma_function

        Parameters
        ----------
        a : ArrayLike
        """
        return gamma_function(
            a,
            exponent=self.exponent,
            negative_number_handling=self.negative_number_handling,
        )


def gamma_function(
    a: ArrayLike,
    exponent: ArrayLike = 1,
    negative_number_handling: NegativeNumberHandlingType = "Indeterminate",
) -> NDArrayFloat:
    """
    Define a typical gamma encoding / decoding function.

    Parameters
    ----------
    a
        Array to encode / decode.
    exponent
        Encoding / decoding exponent.
    negative_number_handling
        Defines the behaviour for ``a`` negative numbers and / or the
        definition return value:

        -   *Indeterminate*: The behaviour will be indeterminate and
            definition return value might contain *nans*.
        -   *Mirror*: The definition return value will be mirrored around
            abscissa and ordinate axis, i.e. Blackmagic Design: Davinci Resolve
            behaviour.
        -   *Preserve*: The definition will preserve any negative number in
            ``a``, i.e. The Foundry Nuke behaviour.
        -   *Clamp*: The definition will clamp any negative number in ``a`` to
            0.

    Returns
    -------
    :class:`numpy.ndarray`
        Encoded / decoded array.

    Examples
    --------
    >>> gamma_function(0.18, 2.2)  # doctest: +ELLIPSIS
    0.0229932...
    >>> gamma_function(-0.18, 2.0)  # doctest: +ELLIPSIS
    0.0323999...
    >>> gamma_function(-0.18, 2.2)
    nan
    >>> gamma_function(-0.18, 2.2, "Mirror")  # doctest: +ELLIPSIS
    -0.0229932...
    >>> gamma_function(-0.18, 2.2, "Preserve")  # doctest: +ELLIPSIS
    -0.1...
    >>> gamma_function(-0.18, 2.2, "Clamp")  # doctest: +ELLIPSIS
    0.0
    """

    a = as_float_array(a)
    exponent = as_float_array(exponent)
    negative_number_handling = validate_method(
        negative_number_handling,
        ("Indeterminate", "Mirror", "Preserve", "Clamp"),
        '"{0}" negative number handling is invalid, it must be one of {1}!',
    )

    if negative_number_handling == "indeterminate":
        return as_float(a**exponent)
    elif negative_number_handling == "mirror":
        return spow(a, exponent)
    elif negative_number_handling == "preserve":
        return as_float(np.where(a <= 0, a, a**exponent))
    else:  # negative_number_handling == 'clamp':
        return as_float(np.where(a <= 0, 0, a**exponent))
