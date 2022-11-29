"""
Blackbody - Planckian Radiator
==============================

Defines the objects to compute the spectral radiance of a planckian radiator
and its spectral distribution.

References
----------
-   :cite:`CIETC1-482004i` : CIE TC 1-48. (2004). APPENDIX E. INFORMATION ON
    THE USE OF PLANCK'S EQUATION FOR STANDARD AIR. In CIE 015:2004 Colorimetry,
    3rd Edition (pp. 77-82). ISBN:978-3-901906-33-6
-   :cite:`Wikipedia2003f` : Wikipedia. (2003). Rayleigh-Jeans law. Retrieved
    February 12, 2022, from https://en.wikipedia.org/wiki/Rayleigh-Jeans_law
"""

from __future__ import annotations

import numpy as np

from colour.algebra import sdiv, sdiv_mode
from colour.colorimetry import (
    SPECTRAL_SHAPE_DEFAULT,
    SpectralDistribution,
    SpectralShape,
)
from colour.constants import CONSTANT_BOLTZMANN, CONSTANT_LIGHT_SPEED
from colour.hints import (
    Floating,
    FloatingOrArrayLike,
    FloatingOrNDArray,
    NDArray,
    cast,
)
from colour.utilities import as_float, as_float_array

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "CONSTANT_C1",
    "CONSTANT_C2",
    "CONSTANT_N",
    "planck_law",
    "blackbody_spectral_radiance",
    "sd_blackbody",
    "rayleigh_jeans_law",
    "sd_rayleigh_jeans",
]

# 2 * math.pi * CONSTANT_PLANCK * CONSTANT_LIGHT_SPEED ** 2
CONSTANT_C1: float = 3.741771e-16

# CONSTANT_PLANCK * CONSTANT_LIGHT_SPEED / CONSTANT_BOLTZMANN
CONSTANT_C2: float = 1.4388e-2

CONSTANT_N: float = 1


def planck_law(
    wavelength: FloatingOrArrayLike,
    temperature: FloatingOrArrayLike,
    c1: Floating = CONSTANT_C1,
    c2: Floating = CONSTANT_C2,
    n: Floating = CONSTANT_N,
) -> FloatingOrNDArray:
    """
    Return the spectral radiance of a blackbody as a function of wavelength at
    thermodynamic temperature :math:`T[K]` in a medium having index of
    refraction :math:`n`.

    Parameters
    ----------
    wavelength
        Wavelength in meters.
    temperature
        Temperature :math:`T[K]` in kelvin degrees.
    c1
        The official value of :math:`c1` is provided by the Committee on Data
        for Science and Technology (CODATA) and is
        :math:`c1=3,741771x10.16\\ W/m_2` *(Mohr and Taylor, 2000)*.
    c2
        Since :math:`T` is measured on the International Temperature Scale,
        the value of :math:`c2` used in colorimetry should follow that adopted
        in the current International Temperature Scale (ITS-90)
        *(Preston-Thomas, 1990; Mielenz et aI., 1991)*, namely
        :math:`c2=1,4388x10.2\\ m/K`.
    n
        Medium index of refraction. For dry air at 15C and 101 325 Pa,
        containing 0,03 percent by volume of carbon dioxide, it is
        approximately 1,00028 throughout the visible region although
        *CIE 15:2004* recommends using :math:`n=1`.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Radiance in *watts per steradian per square metre* (:math:`W/sr/m^2`).

    Warnings
    --------
    The :func:`colour.colorimetry.planck_law` definition behaviour with
    n-dimensional arrays is unusual: The ``wavelength`` and ``temperature``
    parameters are first raveled using :func:`numpy.ravel`. Then, they are
    *broadcasted* together by transposing the ``temperature`` parameter.
    Finally, and for convenience, the return value is squeezed using
    :func:`numpy.squeeze`.

    Notes
    -----
    -   The following implementation is expressed in terms of wavelength.
    -   The SI unit of radiance is *watts per steradian per square metre*
        (:math:`W/sr/m^2`).

    References
    ----------
    :cite:`CIETC1-482004i`

    Examples
    --------
    >>> planck_law(500 * 1e-9, 5500)  # doctest: +ELLIPSIS
    20472701909806.5...
    >>> planck_law(500 * 1e-9, [5000, 5500, 6000])  # doctest: +ELLIPSIS
    array([  1.2106064...e+13,   2.0472701...e+13,   3.1754431...e+13])
    """

    l = as_float_array(wavelength)  # noqa
    t = as_float_array(temperature)

    l = np.ravel(l)[..., None]  # noqa
    t = np.ravel(t)[None, ...]

    with sdiv_mode():
        d = cast(NDArray, sdiv(c2, (n * l * t)))

    d[d != 0] = np.expm1(d[d != 0]) ** -1
    p = ((c1 * n**-2 * l**-5) / np.pi) * d

    return as_float(np.squeeze(p))


blackbody_spectral_radiance = planck_law


def sd_blackbody(
    temperature: Floating,
    shape: SpectralShape = SPECTRAL_SHAPE_DEFAULT,
    c1: Floating = CONSTANT_C1,
    c2: Floating = CONSTANT_C2,
    n: Floating = CONSTANT_N,
) -> SpectralDistribution:
    """
    Return the spectral distribution of the planckian radiator for given
    temperature :math:`T[K]` with values in
    *watts per steradian per square metre per nanometer* (:math:`W/sr/m^2/nm`).

    Parameters
    ----------
    temperature
        Temperature :math:`T[K]` in kelvin degrees.
    shape
        Spectral shape used to create the spectral distribution of the
        planckian radiator.
    c1
        The official value of :math:`c1` is provided by the Committee on Data
        for Science and Technology (CODATA) and is
        :math:`c1=3,741771x10.16\\ W/m_2` *(Mohr and Taylor, 2000)*.
    c2
        Since :math:`T` is measured on the International Temperature Scale,
        the value of :math:`c2` used in colorimetry should follow that adopted
        in the current International Temperature Scale (ITS-90)
        *(Preston-Thomas, 1990; Mielenz et aI., 1991)*, namely
        :math:`c2=1,4388x10.2\\ m/K`.
    n
        Medium index of refraction. For dry air at 15C and 101 325 Pa,
        containing 0,03 percent by volume of carbon dioxide, it is
        approximately 1,00028 throughout the visible region although
        *CIE 15:2004* recommends using :math:`n=1`.

    Returns
    -------
    :class:`colour.SpectralDistribution`
        Blackbody spectral distribution with values in
        *watts per steradian per square metre per nanometer*
        (:math:`W/sr/m^2/nm`).

    Examples
    --------
    >>> from colour.utilities import numpy_print_options
    >>> with numpy_print_options(suppress=True):
    ...     sd_blackbody(5000, shape=SpectralShape(400, 700, 20))
    ...     # doctest: +ELLIPSIS
    ...
    SpectralDistribution([[   400.        ,   8742.5713329...],
                          [   420.        ,   9651.6810212...],
                          [   440.        ,  10447.3423137...],
                          [   460.        ,  11121.8597759...],
                          [   480.        ,  11673.7121534...],
                          [   500.        ,  12106.0645344...],
                          [   520.        ,  12425.4166118...],
                          [   540.        ,  12640.4550541...],
                          [   560.        ,  12761.1284859...],
                          [   580.        ,  12797.9345572...],
                          [   600.        ,  12761.3938171...],
                          [   620.        ,  12661.6795247...],
                          [   640.        ,  12508.3723863...],
                          [   660.        ,  12310.3119640...],
                          [   680.        ,  12075.5205176...],
                          [   700.        ,  11811.1793602...]],
                         SpragueInterpolator,
                         {},
                         Extrapolator,
                         {'method': 'Constant', 'left': None, 'right': None})
    """

    return SpectralDistribution(
        planck_law(shape.wavelengths * 1e-9, temperature, c1, c2, n) * 1e-9,
        shape.wavelengths,
        name=f"{temperature}K Blackbody",
    )


def rayleigh_jeans_law(
    wavelength: FloatingOrArrayLike, temperature: FloatingOrArrayLike
) -> FloatingOrNDArray:
    """
    Return the approximation of the spectral radiance of a blackbody as a
    function of wavelength at thermodynamic temperature :math:`T[K]` according
    to *Rayleigh-Jeans* law.

    Parameters
    ----------
    wavelength
        Wavelength in meters.
    temperature
        Temperature :math:`T[K]` in kelvin degrees.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Radiance in *watts per steradian per square metre* (:math:`W/sr/m^2`).

    Warnings
    --------
    The :func:`colour.colorimetry.rayleigh_jeans_law` definition behaviour with
    n-dimensional arrays is unusual: The ``wavelength`` and ``temperature``
    parameters are first raveled using :func:`numpy.ravel`. Then, they are
    *broadcasted* together by transposing the ``temperature`` parameter.
    Finally, and for convenience, the return value is squeezed using
    :func:`numpy.squeeze`.

    Notes
    -----
    -   The *Rayleigh-Jeans* law agrees with experimental results at large
        wavelengths (low frequencies) but strongly disagrees at short
        wavelengths (high frequencies). This inconsistency between observations
        and the predictions of classical physics is commonly known as the
        *ultraviolet catastrophe*.
    -   The following implementation is expressed in terms of wavelength.
    -   The SI unit of radiance is *watts per steradian per square metre*
        (:math:`W/sr/m^2`).

    References
    ----------
    :cite:`Wikipedia2003f`

    Examples
    --------
    >>> rayleigh_jeans_law(500 * 1e-9, 5500)  # doctest: +ELLIPSIS
    728478884562351.5...
    >>> rayleigh_jeans_law(500 * 1e-9, [5000, 5500, 6000])
    ... # doctest: +ELLIPSIS
    array([  6.6225353...e+14,   7.2847888...e+14,   7.9470423...e+14])
    """

    l = as_float_array(wavelength)  # noqa
    t = as_float_array(temperature)

    l = np.ravel(l)[..., None]  # noqa
    t = np.ravel(t)[None, ...]

    c = CONSTANT_LIGHT_SPEED
    k_B = CONSTANT_BOLTZMANN

    B = (2 * c * k_B * t) / (l**4)

    return as_float(np.squeeze(B))


def sd_rayleigh_jeans(
    temperature: Floating,
    shape: SpectralShape = SPECTRAL_SHAPE_DEFAULT,
) -> SpectralDistribution:
    """
    Return the spectral distribution of the planckian radiator for given
    temperature :math:`T[K]` with values in
    *watts per steradian per square metre per nanometer* (:math:`W/sr/m^2/nm`)
    according to *Rayleigh-Jeans* law.

    Parameters
    ----------
    temperature
        Temperature :math:`T[K]` in kelvin degrees.
    shape
        Spectral shape used to create the spectral distribution of the
        planckian radiator.

    Returns
    -------
    :class:`colour.SpectralDistribution`
        Blackbody spectral distribution with values in
        *watts per steradian per square metre per nanometer*
        (:math:`W/sr/m^2/nm`).

    Notes
    -----
    -   The *Rayleigh-Jeans* law agrees with experimental results at large
        wavelengths (low frequencies) but strongly disagrees at short
        wavelengths (high frequencies). This inconsistency between observations
        and the predictions of classical physics is commonly known as the
        *ultraviolet catastrophe*.

    Examples
    --------
    >>> from colour.utilities import numpy_print_options
    >>> with numpy_print_options(suppress=True):
    ...     sd_rayleigh_jeans(5000, shape=SpectralShape(400, 700, 20))
    ...     # doctest: +ELLIPSIS
    ...
    SpectralDistribution([[     400.        ,  1616829.9106941...],
                          [     420.        ,  1330169.9688456...],
                          [     440.        ,  1104316.5840408...],
                          [     460.        ,   924427.7490112...],
                          [     480.        ,   779721.2146480...],
                          [     500.        ,   662253.5314203...],
                          [     520.        ,   566097.0941823...],
                          [     540.        ,   486776.1157138...],
                          [     560.        ,   420874.0917050...],
                          [     580.        ,   365756.7299433...],
                          [     600.        ,   319373.8095198...],
                          [     620.        ,   280115.7588306...],
                          [     640.        ,   246708.6655722...],
                          [     660.        ,   218136.6091932...],
                          [     680.        ,   193583.6389284...],
                          [     700.        ,   172390.0279623...]],
                         SpragueInterpolator,
                         {},
                         Extrapolator,
                         {'method': 'Constant', 'left': None, 'right': None})
    """

    return SpectralDistribution(
        rayleigh_jeans_law(shape.wavelengths * 1e-9, temperature) * 1e-9,
        shape.wavelengths,
        name=f"{temperature}K Rayleigh-Jeans",
    )
