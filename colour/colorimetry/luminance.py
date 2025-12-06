"""
Luminance :math:`Y`
===================

Define *luminance* :math:`Y` computation methods.

-   :func:`colour.colorimetry.luminance_Newhall1943`: Compute *luminance*
    :math:`Y` from *Munsell* value :math:`V` using *Newhall, Nickerson and
    Judd (1943)* polynomial approximation.
-   :func:`colour.colorimetry.luminance_ASTMD1535`: Compute *luminance*
    :math:`Y` from *Munsell* value :math:`V` using *ASTM D1535-08e1*
    standard polynomial.
-   :func:`colour.colorimetry.luminance_CIE1976`: Compute *luminance*
    :math:`Y` from *CIE 1976* *Lightness* :math:`L^*` using the inverse
    of the standard lightness function.
-   :func:`colour.colorimetry.luminance_Fairchild2010`: Compute *luminance*
    :math:`Y` from *lightness* :math:`L_{hdr}` using *Fairchild and
    Wyble (2010)* method according to *Michaelis-Menten* kinetics.
-   :func:`colour.colorimetry.luminance_Fairchild2011`: Compute *luminance*
    :math:`Y` from *lightness* :math:`L_{hdr}` using *Fairchild and
    Chen (2011)* method according to *Michaelis-Menten* kinetics.
-   :func:`colour.colorimetry.luminance_Abebe2017`: Compute *luminance*
    :math:`Y` from *lightness* :math:`L` using *Abebe, Pouli, Larabi and
    Reinhard (2017)* adaptive method for high-dynamic-range imaging.
-   :attr:`colour.LUMINANCE_METHODS`: Supported *luminance* :math:`Y`
    computation methods registry.
-   :func:`colour.luminance`: Compute *luminance* :math:`Y` from
    *Lightness* :math:`L^*` or *Munsell* value :math:`V` using the specified
    method.

References
----------
-   :cite:`Abebe2017` : Abebe, M. A., Pouli, T., Larabi, M.-C., & Reinhard,
    E. (2017). Perceptual Lightness Modeling for High-Dynamic-Range Imaging.
    ACM Transactions on Applied Perception, 15(1), 1-19. doi:10.1145/3086577
-   :cite:`ASTMInternational2008a` : ASTM International. (2008). ASTM
    D1535-08e1 - Standard Practice for Specifying Color by the Munsell System.
    doi:10.1520/D1535-08E01
-   :cite:`CIETC1-482004m` : CIE TC 1-48. (2004). CIE 1976 uniform colour
    spaces. In CIE 015:2004 Colorimetry, 3rd Edition (p. 24).
    ISBN:978-3-901906-33-6
-   :cite:`Fairchild2010` : Fairchild, M. D., & Wyble, D. R. (2010). hdr-CIELAB
    and hdr-IPT: Simple Models for Describing the Color of High-Dynamic-Range
    and Wide-Color-Gamut Images. Proc. of Color and Imaging Conference,
    322-326. ISBN:978-1-62993-215-6
-   :cite:`Fairchild2011` : Fairchild, M. D., & Chen, P. (2011). Brightness,
    lightness, and specifying color in high-dynamic-range scenes and images. In
    S. P. Farnand & F. Gaykema (Eds.), Proc. SPIE 7867, Image Quality and
    System Performance VIII (p. 78670O). doi:10.1117/12.872075
-   :cite:`Newhall1943a` : Newhall, S. M., Nickerson, D., & Judd, D. B. (1943).
    Final Report of the OSA Subcommittee on the Spacing of the Munsell Colors.
    Journal of the Optical Society of America, 33(7), 385.
    doi:10.1364/JOSA.33.000385
-   :cite:`Wikipedia2001b` : Wikipedia. (2001). Luminance. Retrieved February
    10, 2018, from https://en.wikipedia.org/wiki/Luminance
-   :cite:`Wyszecki2000bd` : Wyszecki, Günther, & Stiles, W. S. (2000). CIE
    1976 (L*u*v*)-Space and Color-Difference Formula. In Color Science:
    Concepts and Methods, Quantitative Data and Formulae (p. 167). Wiley.
    ISBN:978-0-471-39918-6
"""

from __future__ import annotations

import typing

import numpy as np

from colour.algebra import spow
from colour.biochemistry import (
    substrate_concentration_MichaelisMenten_Abebe2017,
    substrate_concentration_MichaelisMenten_Michaelis1913,
)

if typing.TYPE_CHECKING:
    from colour.hints import Any, Literal

from colour.hints import (  # noqa: TC001
    ArrayLike,
    Domain10,
    Domain100,
    NDArrayFloat,
    Range1,
    Range100,
)
from colour.utilities import (
    CanonicalMapping,
    as_float,
    as_float_array,
    filter_kwargs,
    from_range_1,
    from_range_100,
    get_domain_range_scale,
    optional,
    to_domain_10,
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
    "luminance_Newhall1943",
    "luminance_ASTMD1535",
    "intermediate_luminance_function_CIE1976",
    "luminance_CIE1976",
    "luminance_Fairchild2010",
    "luminance_Fairchild2011",
    "luminance_Abebe2017",
    "LUMINANCE_METHODS",
    "luminance",
]


def luminance_Newhall1943(V: Domain10) -> Range100:
    """
    Compute the *luminance* :math:`R_Y` from the specified *Munsell* value
    :math:`V` using *Newhall et al. (1943)* method.

    Parameters
    ----------
    V
        *Munsell* value :math:`V`.

    Returns
    -------
    :class:`numpy.ndarray`
        *Luminance* :math:`R_Y`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``V``      | 10                    | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``R_Y``    | 100                   | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Newhall1943a`

    Examples
    --------
    >>> luminance_Newhall1943(4.08244375)  # doctest: +ELLIPSIS
    12.5500788...
    """

    V = to_domain_10(V)

    R_Y = (
        1.2219 * V
        - 0.23111 * (V * V)
        + 0.23951 * (V**3)
        - 0.021009 * (V**4)
        + 0.0008404 * (V**5)
    )

    return as_float(from_range_100(R_Y))


def luminance_ASTMD1535(V: Domain10) -> Range100:
    """
    Compute *luminance* :math:`Y` from the specified *Munsell* value :math:`V`
    using *ASTM D1535-08e1* method.

    Parameters
    ----------
    V
        *Munsell* value :math:`V`.

    Returns
    -------
    :class:`numpy.ndarray`
        *Luminance* :math:`Y`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``V``      | 10                    | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``Y``      | 100                   | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`ASTMInternational2008a`

    Examples
    --------
    >>> luminance_ASTMD1535(4.08244375)  # doctest: +ELLIPSIS
    12.2363426...
    """

    V = to_domain_10(V)

    Y = (
        1.1914 * V
        - 0.22533 * (V**2)
        + 0.23352 * (V**3)
        - 0.020484 * (V**4)
        + 0.00081939 * (V**5)
    )

    return as_float(from_range_100(Y))


def intermediate_luminance_function_CIE1976(
    f_Y_Y_n: ArrayLike, Y_n: ArrayLike = 100
) -> NDArrayFloat:
    """
    Compute *luminance* :math:`Y` from the specified intermediate value
    :math:`f(Y/Y_n)` using the specified reference white *luminance* :math:`Y_n`
    as per *CIE 1976* recommendation.

    Parameters
    ----------
    f_Y_Y_n
        Intermediate value :math:`f(Y/Y_n)`.
    Y_n
        White reference *luminance* :math:`Y_n`.

    Returns
    -------
    :class:`numpy.ndarray`
        *Luminance* :math:`Y`.

    Notes
    -----
    +-------------+-----------------------+---------------+
    | **Domain**  | **Scale - Reference** | **Scale - 1** |
    +=============+=======================+===============+
    | ``f_Y_Y_n`` | 1                     | 1             |
    +-------------+-----------------------+---------------+

    +-------------+-----------------------+---------------+
    | **Range**   | **Scale - Reference** | **Scale - 1** |
    +=============+=======================+===============+
    | ``Y``       | 100                   | 100           |
    +-------------+-----------------------+---------------+

    References
    ----------
    :cite:`CIETC1-482004m`, :cite:`Wyszecki2000bd`

    Examples
    --------
    >>> intermediate_luminance_function_CIE1976(0.495929964178047)
    ... # doctest: +ELLIPSIS
    12.1972253...
    >>> intermediate_luminance_function_CIE1976(0.504482161449319, 95)
    ... # doctest: +ELLIPSIS
    12.1972253...
    """

    f_Y_Y_n = as_float_array(f_Y_Y_n)
    Y_n = as_float_array(Y_n)

    Y = np.where(
        f_Y_Y_n > 24 / 116,
        Y_n * f_Y_Y_n**3,
        Y_n * (f_Y_Y_n - 16 / 116) * (108 / 841),
    )

    return as_float(Y)


def luminance_CIE1976(L_star: Domain100, Y_n: ArrayLike | None = None) -> Range100:
    """
    Compute the *luminance* :math:`Y` from the specified *lightness* :math:`L^*`
    with the specified reference white *luminance* :math:`Y_n`.

    Parameters
    ----------
    L_star
        *Lightness* :math:`L^*`.
    Y_n
        White reference *luminance* :math:`Y_n`.

    Returns
    -------
    :class:`numpy.ndarray`
        *Luminance* :math:`Y`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``L_star`` | 100                   | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``Y``      | 100                   | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`CIETC1-482004m`, :cite:`Wyszecki2000bd`

    Examples
    --------
    >>> luminance_CIE1976(41.527875844653451)  # doctest: +ELLIPSIS
    12.1972253...
    >>> luminance_CIE1976(41.527875844653451, 95)  # doctest: +ELLIPSIS
    11.5873640...
    """

    L_star = to_domain_100(L_star)
    Y_n = to_domain_100(
        optional(Y_n, 100 if get_domain_range_scale() == "reference" else 1)
    )

    f_Y_Y_n = (L_star + 16) / 116

    Y = intermediate_luminance_function_CIE1976(f_Y_Y_n, Y_n)

    return as_float(from_range_100(Y))


def luminance_Fairchild2010(L_hdr: Domain100, epsilon: ArrayLike = 1.836) -> Range1:
    """
    Compute *luminance* :math:`Y` from the specified *lightness* :math:`L_{hdr}`
    using *Fairchild and Wyble (2010)* method according to *Michaelis-Menten*
    kinetics.

    Parameters
    ----------
    L_hdr
        *Lightness* :math:`L_{hdr}`.
    epsilon
        :math:`\\epsilon` exponent.

    Returns
    -------
    :class:`numpy.ndarray`
        *Luminance* :math:`Y`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``L_hdr``  | 100                   | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``Y``      | 1                     | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Fairchild2010`

    Examples
    --------
    >>> luminance_Fairchild2010(31.996390226262736, 1.836)
    ... # doctest: +ELLIPSIS
    0.1219722...
    """

    L_hdr = to_domain_100(L_hdr)

    Y = np.exp(
        np.log(
            substrate_concentration_MichaelisMenten_Michaelis1913(
                L_hdr - 0.02, 100, spow(0.184, epsilon)
            )
        )
        / epsilon
    )

    return as_float(from_range_1(Y))


def luminance_Fairchild2011(
    L_hdr: Domain100,
    epsilon: ArrayLike = 0.474,
    method: Literal["hdr-CIELAB", "hdr-IPT"] | str = "hdr-CIELAB",
) -> Range1:
    """
    Compute *luminance* :math:`Y` from the specified *lightness* :math:`L_{hdr}`
    using *Fairchild and Chen (2011)* method according to *Michaelis-Menten*
    kinetics.

    Parameters
    ----------
    L_hdr
        *Lightness* :math:`L_{hdr}`.
    epsilon
        :math:`\\epsilon` exponent.
    method
        *Lightness* :math:`L_{hdr}` computation method.

    Returns
    -------
    :class:`numpy.ndarray`
        *Luminance* :math:`Y`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``L_hdr``  | 100                   | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``Y``      | 1                     | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Fairchild2011`

    Examples
    --------
    >>> luminance_Fairchild2011(51.852958445912506)  # doctest: +ELLIPSIS
    0.1219722...
    >>> luminance_Fairchild2011(51.643108411718522, method="hdr-IPT")
    ... # doctest: +ELLIPSIS
    0.1219722...
    """

    L_hdr = to_domain_100(L_hdr)
    method = validate_method(method, ("hdr-CIELAB", "hdr-IPT"))

    maximum_perception = 247 if method == "hdr-cielab" else 246

    Y = np.exp(
        np.log(
            substrate_concentration_MichaelisMenten_Michaelis1913(
                L_hdr - 0.02, maximum_perception, spow(2, epsilon)
            )
        )
        / epsilon
    )

    return as_float(from_range_1(Y))


def luminance_Abebe2017(
    L: ArrayLike,
    Y_n: ArrayLike | None = None,
    method: Literal["Michaelis-Menten", "Stevens"] | str = "Michaelis-Menten",
) -> NDArrayFloat:
    """
    Compute *luminance* :math:`Y` from *lightness* :math:`L` using
    *Abebe, Pouli, Larabi and Reinhard (2017)* adaptive method for
    high-dynamic-range imaging according to *Michaelis-Menten* kinetics or
    *Stevens's Power Law*.

    Parameters
    ----------
    L
        *Lightness* :math:`L`.
    Y_n
        Adapting luminance :math:`Y_n` in :math:`cd/m^2`.
    method
        *Luminance* :math:`Y` computation method.

    Returns
    -------
    :class:`numpy.ndarray`
        *Luminance* :math:`Y` in :math:`cd/m^2`.

    Notes
    -----
    -   *Abebe, Pouli, Larabi and Reinhard (2017)* method uses absolute
        luminance levels, thus the domain and range values for the
        *Reference* and *1* scales are only indicative that the data is not
        affected by scale transformations.

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``L``      | ``UN``                | ``UN``        |
    +------------+-----------------------+---------------+
    | ``Y_n``    | ``UN``                | ``UN``        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``Y``      | ``UN``                | ``UN``        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Abebe2017`

    Examples
    --------
    >>> luminance_Abebe2017(0.486955571109229)  # doctest: +ELLIPSIS
    12.1972253...
    >>> luminance_Abebe2017(0.474544792145434, method="Stevens")
    ... # doctest: +ELLIPSIS
    12.1972253...
    """

    L = as_float_array(L)
    Y_n = as_float_array(optional(Y_n, 100))
    method = validate_method(method, ("Michaelis-Menten", "Stevens"))

    if method == "stevens":
        Y = np.where(
            Y_n <= 100,
            spow((L + 0.226) / 1.226, 1 / 0.266),
            spow((L + 0.127) / 1.127, 1 / 0.230),
        )
    else:
        Y = np.where(
            Y_n <= 100,
            spow(
                substrate_concentration_MichaelisMenten_Abebe2017(
                    L, 1.448, 0.635, 0.813
                ),
                1 / 0.582,
            ),
            spow(
                substrate_concentration_MichaelisMenten_Abebe2017(
                    L, 1.680, 1.584, 0.096
                ),
                1 / 0.293,
            ),
        )
    Y = Y * Y_n

    return as_float(Y)


LUMINANCE_METHODS: CanonicalMapping = CanonicalMapping(
    {
        "Newhall 1943": luminance_Newhall1943,
        "ASTM D1535": luminance_ASTMD1535,
        "CIE 1976": luminance_CIE1976,
        "Fairchild 2010": luminance_Fairchild2010,
        "Fairchild 2011": luminance_Fairchild2011,
        "Abebe 2017": luminance_Abebe2017,
    }
)
LUMINANCE_METHODS.__doc__ = """
Supported *luminance* computation methods.

References
----------
:cite:`ASTMInternational2008a`, :cite:`CIETC1-482004m`,
:cite:`Fairchild2010`, :cite:`Fairchild2011`, :cite:`Newhall1943a`,
:cite:`Wyszecki2000bd`

Aliases:

-   'astm2008': 'ASTM D1535'
-   'cie1976': 'CIE 1976'
"""
LUMINANCE_METHODS["astm2008"] = LUMINANCE_METHODS["ASTM D1535"]
LUMINANCE_METHODS["cie1976"] = LUMINANCE_METHODS["CIE 1976"]


def luminance(
    LV: Domain100,
    method: (
        Literal[
            "Abebe 2017",
            "CIE 1976",
            "Glasser 1958",
            "Fairchild 2010",
            "Fairchild 2011",
            "Wyszecki 1963",
        ]
        | str
    ) = "CIE 1976",
    **kwargs: Any,
) -> Range100:
    """
    Compute the *luminance* :math:`Y` from the specified *lightness*
    :math:`L^*` or *Munsell* value :math:`V`.

    Parameters
    ----------
    LV
        *Lightness* :math:`L^*` or *Munsell* value :math:`V`.
    method
        Computation method.

    Other Parameters
    ----------------
    Y_n
        {:func:`colour.colorimetry.luminance_Abebe2017`,
        :func:`colour.colorimetry.luminance_CIE1976`},
        White reference *luminance* :math:`Y_n`.
    epsilon
        {:func:`colour.colorimetry.luminance_Fairchild2010`,
        :func:`colour.colorimetry.luminance_Fairchild2011`},
        :math:`\\epsilon` exponent.

    Returns
    -------
    :class:`numpy.ndarray`
        *Luminance* :math:`Y`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``LV``     | 100                   | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``Y``      | 100                   | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Abebe2017`, :cite:`ASTMInternational2008a`,
    :cite:`CIETC1-482004m`, :cite:`Fairchild2010`, :cite:`Fairchild2011`,
    :cite:`Newhall1943a`, :cite:`Wikipedia2001b`, :cite:`Wyszecki2000bd`

    Examples
    --------
    >>> luminance(41.527875844653451)  # doctest: +ELLIPSIS
    12.1972253...
    >>> luminance(41.527875844653451, Y_n=100)  # doctest: +ELLIPSIS
    12.1972253...
    >>> luminance(42.51993072812094, Y_n=95)  # doctest: +ELLIPSIS
    12.1972253...
    >>> luminance(4.08244375 * 10, method="Newhall 1943")
    ... # doctest: +ELLIPSIS
    12.5500788...
    >>> luminance(4.08244375 * 10, method="ASTM D1535")
    ... # doctest: +ELLIPSIS
    12.2363426...
    >>> luminance(29.829510892279330, epsilon=0.710, method="Fairchild 2011")
    ... # doctest: +ELLIPSIS
    12.1972253...
    >>> luminance(48.695557110922894, method="Abebe 2017")
    ... # doctest: +ELLIPSIS
    12.1972253...
    """

    LV = as_float_array(LV)
    method = validate_method(method, tuple(LUMINANCE_METHODS))

    function = LUMINANCE_METHODS[method]

    domain_range_reference = get_domain_range_scale() == "reference"
    domain_range_1 = get_domain_range_scale() == "1"
    domain_range_100 = get_domain_range_scale() == "100"

    # Newhall/ASTM methods expect V in [0, 10].
    if (
        function in (luminance_Newhall1943, luminance_ASTMD1535)
        and domain_range_reference
    ):
        LV = LV / 10

    # Abebe expects L in [0, 1] and Y_n in cd/m².
    if function in (luminance_Abebe2017,):
        if domain_range_reference or domain_range_100:
            LV = LV / 100
        if domain_range_1 and "Y_n" in kwargs:
            kwargs["Y_n"] = kwargs["Y_n"] * 100

    Y_V = function(LV, **filter_kwargs(function, **kwargs))

    # Fairchild methods output Y in [0, 1], scale to [0, 100] in reference.
    if (
        function in (luminance_Fairchild2010, luminance_Fairchild2011)
        and domain_range_reference
    ):
        Y_V = Y_V * 100

    # Abebe outputs absolute cd/m², scale to [0, 1] in scale 1.
    if function in (luminance_Abebe2017,) and domain_range_1:
        Y_V = Y_V / 100

    return Y_V
