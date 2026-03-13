"""
Munsell Value
=============

Define *Munsell* value computation objects:

-   :func:`colour.notation.munsell_value_Priest1920`: Compute *Munsell* value
    :math:`V` from the specified *luminance* :math:`Y` using
    *Priest, Gibson and MacNicholas (1920)* method.
-   :func:`colour.notation.munsell_value_Munsell1933`: Compute *Munsell* value
    :math:`V` from the specified *luminance* :math:`Y` using
    *Munsell, Sloan and Godlove (1933)* method.
-   :func:`colour.notation.munsell_value_Moon1943`: Compute *Munsell* value
    :math:`V` from the specified *luminance* :math:`Y` using
    *Moon and Spencer (1943)* method.
-   :func:`colour.notation.munsell_value_Saunderson1944`: Compute *Munsell*
    value :math:`V` from the specified *luminance* :math:`Y` using
    *Saunderson and Milner (1944)* method.
-   :func:`colour.notation.munsell_value_Ladd1955`: Compute *Munsell* value
    :math:`V` from the specified *luminance* :math:`Y` using
    *Ladd and Pinney (1955)* method.
-   :func:`colour.notation.munsell_value_McCamy1987`: Compute *Munsell* value
    :math:`V` from the specified *luminance* :math:`Y` using *McCamy (1987)*
    method.
-   :func:`colour.notation.munsell_value_ASTMD1535`: Compute *Munsell* value
    :math:`V` from the specified *luminance* :math:`Y` using *ASTM D1535-08e1*
    method.
-   :attr:`colour.MUNSELL_VALUE_METHODS`: Supported *Munsell* value
    computation methods.
-   :func:`colour.munsell_value`: Compute *Munsell* value :math:`V` from
    specified *luminance* :math:`Y` using the specified method.

References
----------
-   :cite:`ASTMInternational1989a` : ASTM International. (1989). ASTM D1535-89
    - Standard Practice for Specifying Color by the Munsell System (pp. 1-29).
    Retrieved September 25, 2014, from
    http://www.astm.org/DATABASE.CART/HISTORICAL/D1535-89.htm
-   :cite:`Wikipedia2007c` : Nayatani, Y., Sobagaki, H., & Yano, K. H. T.
    (1995). Lightness dependency of chroma scales of a nonlinear
    color-appearance model and its latest formulation. Color Research &
    Application, 20(3), 156-167. doi:10.1002/col.5080200305
"""

from __future__ import annotations

import typing

import numpy as np

from colour.algebra import (
    Extrapolator,
    LinearInterpolator,
    sdiv,
    sdiv_mode,
    spow,
)
from colour.colorimetry import luminance_ASTMD1535

if typing.TYPE_CHECKING:
    from colour.hints import (
        Domain100,
        Literal,
        Range10,
    )

from colour.utilities import (
    CACHE_REGISTRY,
    CanonicalMapping,
    as_float,
    from_range_10,
    to_domain_100,
    validate_method,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
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

_CACHE_MUNSELL_VALUE_ASTM_D1535_08_INTERPOLATOR: dict = CACHE_REGISTRY.register_cache(
    f"{__name__}._CACHE_MUNSELL_VALUE_ASTM_D1535_08_INTERPOLATOR"
)


def _munsell_value_ASTMD1535_interpolator() -> Extrapolator:
    """
    Return the *Munsell* value interpolator for the *ASTM D1535-08e1*
    method, caching it if not existing.

    Returns
    -------
    :class:`colour.Extrapolator`
        *Munsell* value interpolator for the *ASTM D1535-08e1* method.
    """

    global _CACHE_MUNSELL_VALUE_ASTM_D1535_08_INTERPOLATOR  # noqa: PLW0602

    if "ASTM D1535-08 Interpolator" in (
        _CACHE_MUNSELL_VALUE_ASTM_D1535_08_INTERPOLATOR
    ):
        return _CACHE_MUNSELL_VALUE_ASTM_D1535_08_INTERPOLATOR[
            "ASTM D1535-08 Interpolator"
        ]

    munsell_values = np.arange(0, 10, 0.001)
    interpolator = LinearInterpolator(
        luminance_ASTMD1535(munsell_values), munsell_values
    )
    extrapolator = Extrapolator(interpolator)

    _CACHE_MUNSELL_VALUE_ASTM_D1535_08_INTERPOLATOR["ASTM D1535-08 Interpolator"] = (
        extrapolator
    )

    return extrapolator


def munsell_value_Priest1920(
    Y: Domain100,
) -> Range10:
    """
    Compute the *Munsell* value :math:`V` from the specified *luminance*
    :math:`Y` using *Priest et al. (1920)* method.

    Parameters
    ----------
    Y
        *Luminance* :math:`Y`.

    Returns
    -------
    :class:`np.float` or :class:`numpy.NDArrayFloat`
        *Munsell* value :math:`V`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``Y``      | 100                   | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``V``      | 10                    | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Wikipedia2007c`

    Examples
    --------
    >>> munsell_value_Priest1920(12.23634268)  # doctest: +ELLIPSIS
    np.float64(3.4980484...)
    """

    Y = to_domain_100(Y)

    V = 10 * np.sqrt(Y / 100)

    return as_float(from_range_10(V))


def munsell_value_Munsell1933(
    Y: Domain100,
) -> Range10:
    """
    Compute *Munsell* value :math:`V` from the specified *luminance* :math:`Y`
    using *Munsell et al. (1933)* method.

    Parameters
    ----------
    Y
        *Luminance* :math:`Y`.

    Returns
    -------
    :class:`np.float` or :class:`numpy.NDArrayFloat`
        *Munsell* value :math:`V`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``Y``      | 100                   | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``V``      | 10                    | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Wikipedia2007c`

    Examples
    --------
    >>> munsell_value_Munsell1933(12.23634268)  # doctest: +ELLIPSIS
    np.float64(4.1627702...)
    """

    Y = to_domain_100(Y)

    V = np.sqrt(1.4742 * Y - 0.004743 * (Y * Y))

    return as_float(from_range_10(V))


def munsell_value_Moon1943(Y: Domain100) -> Range10:
    """
    Compute *Munsell* value :math:`V` from the specified *luminance* :math:`Y`
    using *Moon and Spencer (1943)* method.

    Parameters
    ----------
    Y
        *Luminance* :math:`Y`.

    Returns
    -------
    :class:`np.float` or :class:`numpy.NDArrayFloat`
        *Munsell* value :math:`V`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``Y``      | 100                   | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``V``      | 10                    | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Wikipedia2007c`

    Examples
    --------
    >>> munsell_value_Moon1943(12.23634268)  # doctest: +ELLIPSIS
    np.float64(4.0688120...)
    """

    Y = to_domain_100(Y)

    V = 1.4 * spow(Y, 0.426)

    return as_float(from_range_10(V))


def munsell_value_Saunderson1944(
    Y: Domain100,
) -> Range10:
    """
    Compute the *Munsell* value :math:`V` from the specified *luminance* :math:`Y`
    using *Saunderson and Milner (1944)* method.

    Parameters
    ----------
    Y
        *Luminance* :math:`Y`.

    Returns
    -------
    :class:`np.float` or :class:`numpy.NDArrayFloat`
        *Munsell* value :math:`V`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``Y``      | 100                   | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``V``      | 10                    | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Wikipedia2007c`

    Examples
    --------
    >>> munsell_value_Saunderson1944(12.23634268)  # doctest: +ELLIPSIS
    np.float64(4.0444736...)
    """

    Y = to_domain_100(Y)

    V = 2.357 * spow(Y, 0.343) - 1.52

    return as_float(from_range_10(V))


def munsell_value_Ladd1955(Y: Domain100) -> Range10:
    """
    Compute *Munsell* value :math:`V` from the specified *luminance* :math:`Y`
    using *Ladd and Pinney (1955)* method.

    Parameters
    ----------
    Y
        *Luminance* :math:`Y`.

    Returns
    -------
    :class:`np.float` or :class:`numpy.NDArrayFloat`
        *Munsell* value :math:`V`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``Y``      | 100                   | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``V``      | 10                    | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Wikipedia2007c`

    Examples
    --------
    >>> munsell_value_Ladd1955(12.23634268)  # doctest: +ELLIPSIS
    np.float64(4.0511633...)
    """

    Y = to_domain_100(Y)

    V = 2.468 * spow(Y, 1 / 3) - 1.636

    return as_float(from_range_10(V))


def munsell_value_McCamy1987(
    Y: Domain100,
) -> Range10:
    """
    Compute *Munsell* value :math:`V` from the specified *luminance* :math:`Y`
    using *McCamy (1987)* method.

    Parameters
    ----------
    Y
        *Luminance* :math:`Y`.

    Returns
    -------
    :class:`np.float` or :class:`numpy.NDArrayFloat`
        *Munsell* value :math:`V`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``Y``      | 100                   | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``V``      | 10                    | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`ASTMInternational1989a`

    Examples
    --------
    >>> munsell_value_McCamy1987(12.23634268)  # doctest: +ELLIPSIS
    np.float64(4.0814348...)
    """

    Y = to_domain_100(Y)

    with sdiv_mode():
        V = np.where(
            Y <= 0.9,
            0.87445 * spow(Y, 0.9967),
            2.49268 * spow(Y, 1 / 3)
            - 1.5614
            - (0.985 / (((0.1073 * Y - 3.084) ** 2) + 7.54))
            + sdiv(0.0133, spow(Y, 2.3))
            + 0.0084 * np.sin(4.1 * spow(Y, 1 / 3) + 1)
            + sdiv(0.0221, Y) * np.sin(0.39 * (Y - 2))
            - (sdiv(0.0037, 0.44 * Y)) * np.sin(1.28 * (Y - 0.53)),
        )

    return as_float(from_range_10(V))


def munsell_value_ASTMD1535(
    Y: Domain100,
) -> Range10:
    """
    Compute the *Munsell* value :math:`V` from the specified *luminance*
    :math:`Y` using an inverse lookup table from *ASTM D1535-08e1* method.

    Parameters
    ----------
    Y
        *Luminance* :math:`Y`.

    Returns
    -------
    :class:`np.float` or :class:`numpy.NDArrayFloat`
        *Munsell* value :math:`V`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``Y``      | 100                   | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``V``      | 10                    | 1             |
    +------------+-----------------------+---------------+

    -   The *Munsell* value computation with *ASTM D1535-08e1* method is
        only defined for domain [0, 100].

    References
    ----------
    :cite:`ASTMInternational1989a`

    Examples
    --------
    >>> munsell_value_ASTMD1535(12.23634268)  # doctest: +ELLIPSIS
    np.float64(4.0824437...)
    """

    Y = to_domain_100(Y)

    V = _munsell_value_ASTMD1535_interpolator()(Y)

    return as_float(from_range_10(V))


MUNSELL_VALUE_METHODS: CanonicalMapping = CanonicalMapping(
    {
        "Priest 1920": munsell_value_Priest1920,
        "Munsell 1933": munsell_value_Munsell1933,
        "Moon 1943": munsell_value_Moon1943,
        "Saunderson 1944": munsell_value_Saunderson1944,
        "Ladd 1955": munsell_value_Ladd1955,
        "McCamy 1987": munsell_value_McCamy1987,
        "ASTM D1535": munsell_value_ASTMD1535,
    }
)
MUNSELL_VALUE_METHODS.__doc__ = """
Supported *Munsell* value computation methods.

References
----------
:cite:`ASTMInternational1989a`, :cite:`Wikipedia2007c`

Aliases:

-   'astm2008': 'ASTM D1535'
"""
MUNSELL_VALUE_METHODS["astm2008"] = MUNSELL_VALUE_METHODS["ASTM D1535"]


def munsell_value(
    Y: Domain100,
    method: (
        Literal[
            "ASTM D1535",
            "Ladd 1955",
            "McCamy 1987",
            "Moon 1943",
            "Munsell 1933",
            "Priest 1920",
            "Saunderson 1944",
        ]
        | str
    ) = "ASTM D1535",
) -> Range10:
    """
    Compute the *Munsell* value :math:`V` from the specified *luminance*
    :math:`Y` using the specified computational method.

    Parameters
    ----------
    Y
        *Luminance* :math:`Y`.
    method
        Computation method.

    Returns
    -------
    :class:`np.float` or :class:`numpy.NDArrayFloat`
        *Munsell* value :math:`V`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``Y``      | 100                   | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``V``      | 10                    | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`ASTMInternational1989a`, :cite:`Wikipedia2007c`

    Examples
    --------
    >>> munsell_value(12.23634268)  # doctest: +ELLIPSIS
    np.float64(4.0824437...)
    >>> munsell_value(12.23634268, method="Priest 1920")  # doctest: +ELLIPSIS
    np.float64(3.4980484...)
    >>> munsell_value(12.23634268, method="Munsell 1933")  # doctest: +ELLIPSIS
    np.float64(4.1627702...)
    >>> munsell_value(12.23634268, method="Moon 1943")  # doctest: +ELLIPSIS
    np.float64(4.0688120...)
    >>> munsell_value(12.23634268, method="Saunderson 1944")
    ... # doctest: +ELLIPSIS
    np.float64(4.0444736...)
    >>> munsell_value(12.23634268, method="Ladd 1955")  # doctest: +ELLIPSIS
    np.float64(4.0511633...)
    >>> munsell_value(12.23634268, method="McCamy 1987")  # doctest: +ELLIPSIS
    np.float64(4.0814348...)
    """

    method = validate_method(method, tuple(MUNSELL_VALUE_METHODS))

    return MUNSELL_VALUE_METHODS[method](Y)
