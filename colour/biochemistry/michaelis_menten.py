"""
Michaelis-Menten Kinetics
=========================

Implements support for *Michaelis-Menten* kinetics, a model of enzyme kinetics:

-   :func:`colour.biochemistry.reaction_rate_MichaelisMenten_Michaelis1913`
-   :func:`colour.biochemistry.reaction_rate_MichaelisMenten_Abebe2017`
-   :func:`colour.biochemistry.REACTION_RATE_MICHAELISMENTEN_METHODS`
-   :func:`colour.biochemistry.reaction_rate_MichaelisMenten`
-   :func:`colour.biochemistry.\
substrate_concentration_MichaelisMenten_Michaelis1913`
-   :func:`colour.biochemistry.\
substrate_concentration_MichaelisMenten_Abebe2017`
-   :func:`colour.biochemistry.SUBSTRATE_CONCENTRATION_MICHAELISMENTEN_METHODS`
-   :func:`colour.biochemistry.substrate_concentration_MichaelisMenten`

References
----------
-   :cite:`Abebe2017` : Abebe, M. A., Pouli, T., Larabi, M.-C., & Reinhard,
    E. (2017). Perceptual Lightness Modeling for High-Dynamic-Range Imaging.
    ACM Transactions on Applied Perception, 15(1), 1-19. doi:10.1145/3086577
-   :cite:`Wikipedia2003d` : Wikipedia. (2003). Michaelis-Menten kinetics.
    Retrieved April 29, 2017, from
    https://en.wikipedia.org/wiki/Michaelis%E2%80%93Menten_kinetics
"""

from __future__ import annotations

from colour.hints import Any, ArrayLike, Literal, NDArrayFloat
from colour.utilities import (
    CanonicalMapping,
    as_float,
    as_float_array,
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
    "reaction_rate_MichaelisMenten_Michaelis1913",
    "reaction_rate_MichaelisMenten_Abebe2017",
    "REACTION_RATE_MICHAELISMENTEN_METHODS",
    "reaction_rate_MichaelisMenten",
    "substrate_concentration_MichaelisMenten_Michaelis1913",
    "substrate_concentration_MichaelisMenten_Abebe2017",
    "SUBSTRATE_CONCENTRATION_MICHAELISMENTEN_METHODS",
    "substrate_concentration_MichaelisMenten",
]


def reaction_rate_MichaelisMenten_Michaelis1913(
    S: ArrayLike,
    V_max: ArrayLike,
    K_m: ArrayLike,
) -> NDArrayFloat:
    """
    Describe the rate of enzymatic reactions, by relating reaction rate
    :math:`v` to concentration of a substrate :math:`S`.

    Parameters
    ----------
    S
        Concentration of a substrate :math:`S`.
    V_max
        Maximum rate :math:`V_{max}` achieved by the system, at saturating
        substrate concentration.
    K_m
        Substrate concentration :math:`K_m` at which the reaction rate is
        half of :math:`V_{max}`.

    Returns
    -------
    :class:`numpy.ndarray`
        Reaction rate :math:`v`.

    References
    ----------
    :cite:`Wikipedia2003d`

    Examples
    --------
    >>> reaction_rate_MichaelisMenten(0.5, 2.5, 0.8)  # doctest: +ELLIPSIS
    0.9615384...
    """

    S = as_float_array(S)
    V_max = as_float_array(V_max)
    K_m = as_float_array(K_m)

    v = (V_max * S) / (K_m + S)

    return as_float(v)


def reaction_rate_MichaelisMenten_Abebe2017(
    S: ArrayLike,
    V_max: ArrayLike,
    K_m: ArrayLike,
    b_m: ArrayLike,
) -> NDArrayFloat:
    """
    Describe the rate of enzymatic reactions, by relating reaction rate
    :math:`v` to concentration of a substrate :math:`S` according to the
    modified *Michaelis-Menten* kinetics equation as given by
    *Abebe, Pouli, Larabi and Reinhard (2017)*.

    Parameters
    ----------
    S
        Concentration of a substrate :math:`S` (or
        :math:`(\\cfrac{Y}{Y_n})^{\\epsilon}`).
    V_max
        Maximum rate :math:`V_{max}` (or :math:`a_m`) achieved by the system,
        at saturating substrate concentration.
    K_m
        Substrate concentration :math:`K_m` (or :math:`c_m`) at which the
        reaction rate is half of :math:`V_{max}`.
    b_m
        Bias factor :math:`b_m`.

    Returns
    -------
    :class:`numpy.ndarray`
        Reaction rate :math:`v`.

    References
    ----------
    :cite:`Abebe2017`

    Examples
    --------
    >>> reaction_rate_MichaelisMenten_Abebe2017(0.5, 1.448, 0.635, 0.813)
    ... # doctest: +ELLIPSIS
    0.6951512...
    """

    S = as_float_array(S)
    V_max = as_float_array(V_max)
    K_m = as_float_array(K_m)
    b_m = as_float_array(b_m)

    v = (V_max * S) / (b_m * S + K_m)

    return as_float(v)


REACTION_RATE_MICHAELISMENTEN_METHODS: CanonicalMapping = CanonicalMapping(
    {
        "Michaelis 1913": reaction_rate_MichaelisMenten_Michaelis1913,
        "Abebe 2017": reaction_rate_MichaelisMenten_Abebe2017,
    }
)
REACTION_RATE_MICHAELISMENTEN_METHODS.__doc__ = """
Supported *Michaelis-Menten* kinetics reaction rate equation computation
methods.

References
----------
:cite:`Wikipedia2003d`, :cite:`Abebe2017`
"""


def reaction_rate_MichaelisMenten(
    S: ArrayLike,
    V_max: ArrayLike,
    K_m: ArrayLike,
    method: Literal["Michaelis 1913", "Abebe 2017"] | str = "Michaelis 1913",
    **kwargs: Any,
) -> NDArrayFloat:
    """
    Describe the rate of enzymatic reactions, by relating reaction rate
    :math:`v` to concentration of a substrate :math:`S` according to given
    method.

    Parameters
    ----------
    S
        Concentration of a substrate :math:`S`.
    V_max
        Maximum rate :math:`V_{max}` achieved by the system, at saturating
        substrate concentration.
    K_m
        Substrate concentration :math:`K_m` at which the reaction rate is
        half of :math:`V_{max}`.
    method
        Computation method.

    Other Parameters
    ----------------
    b_m
        {:func:`colour.biochemistry.reaction_rate_MichaelisMenten_Abebe2017`},
        Bias factor :math:`b_m`.

    Returns
    -------
    :class:`numpy.ndarray`
        Reaction rate :math:`v`.

    References
    ----------
    :cite:`Wikipedia2003d`, :cite:`Abebe2017`

    Examples
    --------
    >>> reaction_rate_MichaelisMenten(0.5, 2.5, 0.8)  # doctest: +ELLIPSIS
    0.9615384...
    >>> reaction_rate_MichaelisMenten(
    ...     0.5, 2.5, 0.8, method="Abebe 2017", b_m=0.813
    ... )  # doctest: +ELLIPSIS
    1.0360547...
    """

    method = validate_method(
        method, tuple(REACTION_RATE_MICHAELISMENTEN_METHODS)
    )

    function = REACTION_RATE_MICHAELISMENTEN_METHODS[method]

    return function(S, V_max, K_m, **filter_kwargs(function, **kwargs))


def substrate_concentration_MichaelisMenten_Michaelis1913(
    v: ArrayLike,
    V_max: ArrayLike,
    K_m: ArrayLike,
) -> NDArrayFloat:
    """
    Describe the rate of enzymatic reactions, by relating concentration of a
    substrate :math:`S` to reaction rate :math:`v`.

    Parameters
    ----------
    v
        Reaction rate :math:`v`.
    V_max
        Maximum rate :math:`V_{max}` achieved by the system, at saturating
        substrate concentration.
    K_m
        Substrate concentration :math:`K_m` at which the reaction rate is
        half of :math:`V_{max}`.

    Returns
    -------
    :class:`numpy.ndarray`
        Concentration of a substrate :math:`S`.

    References
    ----------
    :cite:`Wikipedia2003d`

    Examples
    --------
    >>> substrate_concentration_MichaelisMenten(0.961538461538461, 2.5, 0.8)
    ... # doctest: +ELLIPSIS
    0.4999999...
    """

    v = as_float_array(v)
    V_max = as_float_array(V_max)
    K_m = as_float_array(K_m)

    S = (v * K_m) / (V_max - v)

    return as_float(S)


def substrate_concentration_MichaelisMenten_Abebe2017(
    v: ArrayLike,
    V_max: ArrayLike,
    K_m: ArrayLike,
    b_m: ArrayLike,
) -> NDArrayFloat:
    """
    Describe the rate of enzymatic reactions, by relating concentration of a
    substrate :math:`S` to reaction rate :math:`v` according to the modified
    *Michaelis-Menten* kinetics equation as given by
    *Abebe, Pouli, Larabi and Reinhard (2017)*.

    Parameters
    ----------
    v
        Reaction rate :math:`v`.
    V_max
        Maximum rate :math:`V_{max}` (or :math:`a_m`) achieved by the system,
        at saturating substrate concentration.
    K_m
        Substrate concentration :math:`K_m` (or :math:`c_m`) at which the
        reaction rate is half of :math:`V_{max}`.
    b_m
        Bias factor :math:`b_m`.

    Returns
    -------
    :class:`numpy.ndarray`
        Concentration of a substrate :math:`S`.

    References
    ----------
    :cite:`Abebe2017`

    Examples
    --------
    >>> substrate_concentration_MichaelisMenten_Abebe2017(
    ...     0.695151224195871, 1.448, 0.635, 0.813
    ... )  # doctest: +ELLIPSIS
    0.4999999...
    """

    v = as_float_array(v)
    V_max = as_float_array(V_max)
    K_m = as_float_array(K_m)
    b_m = as_float_array(b_m)

    S = (v * K_m) / (V_max - b_m * v)

    return as_float(S)


SUBSTRATE_CONCENTRATION_MICHAELISMENTEN_METHODS: (
    CanonicalMapping
) = CanonicalMapping(
    {
        "Michaelis 1913": substrate_concentration_MichaelisMenten_Michaelis1913,
        "Abebe 2017": substrate_concentration_MichaelisMenten_Abebe2017,
    }
)
SUBSTRATE_CONCENTRATION_MICHAELISMENTEN_METHODS.__doc__ = """
Supported *Michaelis-Menten* kinetics substrate concentration equation
computation methods.

References
----------
:cite:`Wikipedia2003d`, :cite:`Abebe2017`
"""


def substrate_concentration_MichaelisMenten(
    v: ArrayLike,
    V_max: ArrayLike,
    K_m: ArrayLike,
    method: Literal["Michaelis 1913", "Abebe 2017"] | str = "Michaelis 1913",
    **kwargs: Any,
) -> NDArrayFloat:
    """
    Describe the rate of enzymatic reactions, by relating concentration of a
    substrate :math:`S` to reaction rate :math:`v` according to given method.

    Parameters
    ----------
    v
        Reaction rate :math:`v`.
    V_max
        Maximum rate :math:`V_{max}` achieved by the system, at saturating
        substrate concentration.
    K_m
        Substrate concentration :math:`K_m` at which the reaction rate is
        half of :math:`V_{max}`.
    method
        Computation method.

    Other Parameters
    ----------------
    b_m
        {:func:`colour.biochemistry.\
substrate_concentration_MichaelisMenten_Abebe2017`},
        Bias factor :math:`b_m`.

    Returns
    -------
    :class:`numpy.ndarray`
        Concentration of a substrate :math:`S`.

    References
    ----------
    :cite:`Wikipedia2003d`, :cite:`Abebe2017`

    Examples
    --------
    >>> substrate_concentration_MichaelisMenten(0.961538461538461, 2.5, 0.8)
    ... # doctest: +ELLIPSIS
    0.4999999...
    >>> substrate_concentration_MichaelisMenten(
    ...     1.036054703688355, 2.5, 0.8, method="Abebe 2017", b_m=0.813
    ... )
    ... # doctest: +ELLIPSIS
    0.5000000...
    """

    method = validate_method(
        method, tuple(SUBSTRATE_CONCENTRATION_MICHAELISMENTEN_METHODS)
    )

    function = SUBSTRATE_CONCENTRATION_MICHAELISMENTEN_METHODS[method]

    return function(v, V_max, K_m, **filter_kwargs(function, **kwargs))
