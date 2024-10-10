"""
Rayleigh Optical Depth - Scattering in the Atmosphere
=====================================================

Implement *Rayleigh* scattering / optical depth in the atmosphere computation:

-   :func:`colour.scattering_cross_section`
-   :func:`colour.phenomena.rayleigh_optical_depth`
-   :func:`colour.rayleigh_scattering`

References
----------
-   :cite:`Bodhaine1999a` : Bodhaine, B. A., Wood, N. B., Dutton, E. G., &
    Slusser, J. R. (1999). On Rayleigh Optical Depth Calculations. Journal of
    Atmospheric and Oceanic Technology, 16(11), 1854-1861.
    doi:10.1175/1520-0426(1999)016<1854:ORODC>2.0.CO;2
-   :cite:`Wikipedia2001c` : Wikipedia. (2001). Rayleigh scattering. Retrieved
    September 23, 2014, from http://en.wikipedia.org/wiki/Rayleigh_scattering
"""

from __future__ import annotations

import numpy as np

from colour.algebra import sdiv, sdiv_mode
from colour.colorimetry import (
    SPECTRAL_SHAPE_DEFAULT,
    SpectralDistribution,
    SpectralShape,
)
from colour.constants import CONSTANT_AVOGADRO
from colour.hints import ArrayLike, Callable, NDArrayFloat
from colour.utilities import as_float, as_float_array, filter_kwargs

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "CONSTANT_STANDARD_AIR_TEMPERATURE",
    "CONSTANT_STANDARD_CO2_CONCENTRATION",
    "CONSTANT_AVERAGE_PRESSURE_MEAN_SEA_LEVEL",
    "CONSTANT_DEFAULT_LATITUDE",
    "CONSTANT_DEFAULT_ALTITUDE",
    "air_refraction_index_Penndorf1957",
    "air_refraction_index_Edlen1966",
    "air_refraction_index_Peck1972",
    "air_refraction_index_Bodhaine1999",
    "N2_depolarisation",
    "O2_depolarisation",
    "F_air_Penndorf1957",
    "F_air_Young1981",
    "F_air_Bates1984",
    "F_air_Bodhaine1999",
    "molecular_density",
    "mean_molecular_weights",
    "gravity_List1968",
    "scattering_cross_section",
    "rayleigh_optical_depth",
    "rayleigh_scattering",
    "sd_rayleigh_scattering",
]

CONSTANT_STANDARD_AIR_TEMPERATURE: float = 288.15
"""*Standard air* temperature :math:`T[K]` in kelvin degrees (:math:`15\\circ C`)."""

CONSTANT_STANDARD_CO2_CONCENTRATION: float = 300
"""*Standard air* :math:`CO_2` concentration in parts per million (ppm)."""

CONSTANT_AVERAGE_PRESSURE_MEAN_SEA_LEVEL: float = 101325
"""*Standard air* average pressure :math:`Hg` at mean sea-level in pascal (Pa)."""

CONSTANT_DEFAULT_LATITUDE: float = 0
"""Default latitude in degrees (equator)."""

CONSTANT_DEFAULT_ALTITUDE: float = 0
"""Default altitude in meters (sea level)."""


def air_refraction_index_Penndorf1957(
    wavelength: ArrayLike,
) -> NDArrayFloat:
    """
    Return the air refraction index :math:`n_s` from given wavelength
    :math:`\\lambda` in  micrometers (:math:`\\mu m`) using *Penndorf (1957)*
    method.

    Parameters
    ----------
    wavelength
        Wavelength :math:`\\lambda` in micrometers (:math:`\\mu m`).

    Returns
    -------
    :class:`numpy.ndarray`
        Air refraction index :math:`n_s`.

    Examples
    --------
    >>> air_refraction_index_Penndorf1957(0.555)  # doctest: +ELLIPSIS
    1.0002777...
    """

    wl = as_float_array(wavelength)

    n = 6432.8 + 2949810 / (146 - wl ** (-2)) + 25540 / (41 - wl ** (-2))
    n /= 1.0e8
    n += +1

    return n


def air_refraction_index_Edlen1966(
    wavelength: ArrayLike,
) -> NDArrayFloat:
    """
    Return the air refraction index :math:`n_s` from given wavelength
    :math:`\\lambda` in micrometers (:math:`\\mu m`) using *Edlen (1966)*
    method.

    Parameters
    ----------
    wavelength
        Wavelength :math:`\\lambda` in micrometers (:math:`\\mu m`).

    Returns
    -------
    :class:`numpy.ndarray`
        Air refraction index :math:`n_s`.

    Examples
    --------
    >>> air_refraction_index_Edlen1966(0.555)  # doctest: +ELLIPSIS
    1.0002777...
    """

    wl = as_float_array(wavelength)

    n = 8342.13 + 2406030 / (130 - wl ** (-2)) + 15997 / (38.9 - wl ** (-2))
    n /= 1.0e8
    n += +1

    return n


def air_refraction_index_Peck1972(
    wavelength: ArrayLike,
) -> NDArrayFloat:
    """
    Return the air refraction index :math:`n_s` from given wavelength
    :math:`\\lambda` in micrometers (:math:`\\mu m`) using
    *Peck and Reeder (1972)* method.

    Parameters
    ----------
    wavelength
        Wavelength :math:`\\lambda` in micrometers (:math:`\\mu m`).

    Returns
    -------
    :class:`numpy.ndarray`
        Air refraction index :math:`n_s`.

    Examples
    --------
    >>> air_refraction_index_Peck1972(0.555)  # doctest: +ELLIPSIS
    1.0002777...
    """

    wl = as_float_array(wavelength)

    n = 8060.51 + 2480990 / (132.274 - wl ** (-2)) + 17455.7 / (39.32957 - wl ** (-2))
    n /= 1.0e8
    n += +1

    return n


def air_refraction_index_Bodhaine1999(
    wavelength: ArrayLike,
    CO2_concentration: ArrayLike = CONSTANT_STANDARD_CO2_CONCENTRATION,
) -> NDArrayFloat:
    """
    Return the air refraction index :math:`n_s` from given wavelength
    :math:`\\lambda` in micrometers (:math:`\\mu m`) using
    *Bodhaine, Wood, Dutton and Slusser (1999)* method.

    Parameters
    ----------
    wavelength
        Wavelength :math:`\\lambda` in micrometers (:math:`\\mu m`).
    CO2_concentration
        :math:`CO_2` concentration in parts per million (ppm).

    Returns
    -------
    :class:`numpy.ndarray`
        Air refraction index :math:`n_s`.

    Examples
    --------
    >>> air_refraction_index_Bodhaine1999(0.555)  # doctest: +ELLIPSIS
    1.0002777...
    """

    wl = as_float_array(wavelength)
    CO2_c = as_float_array(CO2_concentration)

    # Converting from parts per million (ppm) to parts per volume (ppv).
    CO2_c = CO2_c * 1e-6

    n = (1 + 0.54 * (CO2_c - 300e-6)) * (air_refraction_index_Peck1972(wl) - 1) + 1

    return as_float(n)


def N2_depolarisation(wavelength: ArrayLike) -> NDArrayFloat:
    """
    Return the depolarisation of nitrogen :math:`N_2` as function of
    wavelength :math:`\\lambda` in micrometers (:math:`\\mu m`).

    Parameters
    ----------
    wavelength
        Wavelength :math:`\\lambda` in micrometers (:math:`\\mu m`).

    Returns
    -------
    :class:`numpy.ndarray`
        Nitrogen :math:`N_2` depolarisation.

    Examples
    --------
    >>> N2_depolarisation(0.555)  # doctest: +ELLIPSIS
    1.0350291...
    """

    wl = as_float_array(wavelength)

    N2 = 1.034 + 3.17 * 1.0e-4 * (1 / wl**2)

    return N2


def O2_depolarisation(wavelength: ArrayLike) -> NDArrayFloat:
    """
    Return the depolarisation of oxygen :math:`O_2` as function of
    wavelength :math:`\\lambda` in micrometers (:math:`\\mu m`).

    Parameters
    ----------
    wavelength
        Wavelength :math:`\\lambda` in micrometers (:math:`\\mu m`).

    Returns
    -------
    :class:`numpy.ndarray`
        Oxygen :math:`O_2` depolarisation.

    Examples
    --------
    >>> O2_depolarisation(0.555)  # doctest: +ELLIPSIS
    1.1020225...
    """

    wl = as_float_array(wavelength)

    O2 = 1.096 + 1.385 * 1.0e-3 * (1 / wl**2) + 1.448 * 1.0e-4 * (1 / wl**4)

    return O2


def F_air_Penndorf1957(wavelength: ArrayLike) -> NDArrayFloat:
    """
    Return :math:`(6+3_p)/(6-7_p)`, the depolarisation term :math:`F(air)` or
    *King Factor* using *Penndorf (1957)* method.

    Parameters
    ----------
    wavelength
        Wavelength :math:`\\lambda` in micrometers (:math:`\\mu m`).

    Returns
    -------
    :class:`numpy.ndarray`
        Air depolarisation.

    Notes
    -----
    -   The argument *wavelength* is only provided for consistency with the
        other air depolarisation methods but is actually not used as this
        definition is essentially a constant in its current implementation.

    Examples
    --------
    >>> F_air_Penndorf1957(0.555)
    1.0608
    """

    wl = as_float_array(wavelength)

    return as_float(np.resize(np.array([1.0608]), wl.shape))


def F_air_Young1981(wavelength: ArrayLike) -> NDArrayFloat:
    """
    Return :math:`(6+3_p)/(6-7_p)`, the depolarisation term :math:`F(air)` or
    *King Factor* using *Young (1981)* method.

    Parameters
    ----------
    wavelength
        Wavelength :math:`\\lambda` in micrometers (:math:`\\mu m`).

    Returns
    -------
    :class:`numpy.ndarray`
        Air depolarisation.

    Notes
    -----
    -   The argument *wavelength* is only provided for consistency with the
        other air depolarisation methods but is actually not used as this
        definition is essentially a constant in its current implementation.

    Examples
    --------
    >>> F_air_Young1981(0.555)
    1.048
    """

    wl = as_float_array(wavelength)

    return as_float(np.resize(np.array([1.0480]), wl.shape))


def F_air_Bates1984(wavelength: ArrayLike) -> NDArrayFloat:
    """
    Return :math:`(6+3_p)/(6-7_p)`, the depolarisation term :math:`F(air)` or
    *King Factor* as function of wavelength :math:`\\lambda` in micrometers
    (:math:`\\mu m`) using *Bates (1984)* method.

    Parameters
    ----------
    wavelength
        Wavelength :math:`\\lambda` in micrometers (:math:`\\mu m`).

    Returns
    -------
    :class:`numpy.ndarray`
        Air depolarisation.

    Examples
    --------
    >>> F_air_Bates1984(0.555)  # doctest: +ELLIPSIS
    1.0481535...
    """

    O2 = O2_depolarisation(wavelength)
    N2 = N2_depolarisation(wavelength)
    Ar = 1.00
    CO2 = 1.15

    F_air = (78.084 * N2 + 20.946 * O2 + CO2 + Ar) / (78.084 + 20.946 + Ar + CO2)

    return F_air


def F_air_Bodhaine1999(
    wavelength: ArrayLike,
    CO2_concentration: ArrayLike = CONSTANT_STANDARD_CO2_CONCENTRATION,
) -> NDArrayFloat:
    """
    Return :math:`(6+3_p)/(6-7_p)`, the depolarisation term :math:`F(air)` or
    *King Factor* as function of wavelength :math:`\\lambda` in micrometers
    (:math:`\\mu m`) and :math:`CO_2` concentration in parts per million (ppm)
    using *Bodhaine, Wood, Dutton and Slusser (1999)* method.

    Parameters
    ----------
    wavelength
        Wavelength :math:`\\lambda` in micrometers (:math:`\\mu m`).
    CO2_concentration
        :math:`CO_2` concentration in parts per million (ppm).

    Returns
    -------
    :class:`numpy.ndarray`
        Air depolarisation.

    Examples
    --------
    >>> F_air_Bodhaine1999(0.555)  # doctest: +ELLIPSIS
    1.0487697...
    """

    O2 = O2_depolarisation(wavelength)
    N2 = N2_depolarisation(wavelength)
    CO2_c = as_float_array(CO2_concentration)

    # Converting from parts per million (ppm) to parts per volume per percent.
    CO2_c = CO2_c * 1e-4

    F_air = (78.084 * N2 + 20.946 * O2 + 0.934 * 1 + CO2_c * 1.15) / (
        78.084 + 20.946 + 0.934 + CO2_c
    )

    return F_air


def molecular_density(
    temperature: ArrayLike = CONSTANT_STANDARD_AIR_TEMPERATURE,
    avogadro_constant: ArrayLike = CONSTANT_AVOGADRO,
) -> NDArrayFloat:
    """
    Return the molecular density :math:`N_s` (molecules :math:`cm^{-3}`)
    as function of air temperature :math:`T[K]` in kelvin degrees.

    Parameters
    ----------
    temperature
        Air temperature :math:`T[K]` in kelvin degrees.
    avogadro_constant
        *Avogadro*'s number (molecules :math:`mol^{-1}`).

    Returns
    -------
    :class:`numpy.ndarray`
        Molecular density :math:`N_s` (molecules :math:`cm^{-3}`).

    Notes
    -----
    -   The *Avogadro*'s number used in this implementation is the one given by
        by the Committee on Data for Science and Technology (CODATA):
        :math:`6.02214179x10^{23}`, which is different from the reference
        :cite:`Bodhaine1999a` value :math:`6.0221367x10^{23}`.

    Examples
    --------
    >>> molecular_density(288.15)  # doctest: +ELLIPSIS
    2.5469021...e+19
    >>> molecular_density(288.15, 6.0221367e23)  # doctest: +ELLIPSIS
    2.5468999...e+19
    """

    T = as_float_array(temperature)
    avogadro_constant = as_float_array(avogadro_constant)

    with sdiv_mode():
        N_s = (avogadro_constant / 22.4141) * sdiv(273.15, T) * (1 / 1000)

    return N_s


def mean_molecular_weights(
    CO2_concentration: ArrayLike = CONSTANT_STANDARD_CO2_CONCENTRATION,
) -> NDArrayFloat:
    """
    Return the mean molecular weights :math:`m_a` for dry air as function of
    :math:`CO_2` concentration in parts per million (ppm).

    Parameters
    ----------
    CO2_concentration
        :math:`CO_2` concentration in parts per million (ppm).

    Returns
    -------
    :class:`numpy.ndarray`
        Mean molecular weights :math:`m_a` for dry air.

    Examples
    --------
    >>> mean_molecular_weights()  # doctest: +ELLIPSIS
    28.9640166...
    """

    CO2_concentration = as_float_array(CO2_concentration)

    CO2_c = CO2_concentration * 1.0e-6

    m_a = 15.0556 * CO2_c + 28.9595
    return m_a


def gravity_List1968(
    latitude: ArrayLike = CONSTANT_DEFAULT_LATITUDE,
    altitude: ArrayLike = CONSTANT_DEFAULT_ALTITUDE,
) -> NDArrayFloat:
    """
    Return the gravity :math:`g` in :math:`cm/s_2` (gal) representative of the
    mass-weighted column of air molecules above the site of given latitude and
    altitude using *list (1968)* method.

    Parameters
    ----------
    latitude
        Latitude of the site in degrees.
    altitude
        Altitude of the site in meters.

    Returns
    -------
    :class:`numpy.ndarray`
        Gravity :math:`g` in :math:`cm/s_2` (gal).

    Examples
    --------
    >>> gravity_List1968()  # doctest: +ELLIPSIS
    978.0356070...
    >>> gravity_List1968(0.0, 1500.0)  # doctest: +ELLIPSIS
    977.5726106...

    Gravity :math:`g` for Paris:

    >>> gravity_List1968(48.8567, 35.0)  # doctest: +ELLIPSIS
    980.9524178...
    """

    latitude = as_float_array(latitude)
    altitude = as_float_array(altitude)

    cos2phi = np.cos(2 * np.radians(latitude))

    # Sea level acceleration of gravity.
    g0 = 980.6160 * (1 - 0.0026373 * cos2phi + 0.0000059 * cos2phi**2)

    g = (
        g0
        - (3.085462e-4 + 2.27e-7 * cos2phi) * altitude
        + (7.254e-11 + 1.0e-13 * cos2phi) * altitude**2
        - (1.517e-17 + 6e-20 * cos2phi) * altitude**3
    )

    return g


def scattering_cross_section(
    wavelength: ArrayLike,
    CO2_concentration: ArrayLike = CONSTANT_STANDARD_CO2_CONCENTRATION,
    temperature: ArrayLike = CONSTANT_STANDARD_AIR_TEMPERATURE,
    avogadro_constant: ArrayLike = CONSTANT_AVOGADRO,
    n_s_function: Callable = air_refraction_index_Bodhaine1999,
    F_air_function: Callable = F_air_Bodhaine1999,
) -> NDArrayFloat:
    """
    Return the scattering cross-section per molecule :math:`\\sigma` of dry
    air as function of wavelength :math:`\\lambda` in centimeters (cm) using
    given :math:`CO_2` concentration in parts per million (ppm) and temperature
    :math:`T[K]` in kelvin degrees following *Van de Hulst (1957)* method.

    Parameters
    ----------
    wavelength
        Wavelength :math:`\\lambda` in centimeters (cm).
    CO2_concentration
        :math:`CO_2` concentration in parts per million (ppm).
    temperature
        Air temperature :math:`T[K]` in kelvin degrees.
    avogadro_constant
        *Avogadro*'s number (molecules :math:`mol^{-1}`).
    n_s_function
        Air refraction index :math:`n_s` computation method.
    F_air_function
        :math:`(6+3_p)/(6-7_p)`, the depolarisation term :math:`F(air)` or
        *King Factor* computation method.

    Returns
    -------
    :class:`numpy.ndarray`
        Scattering cross-section per molecule :math:`\\sigma` of dry air.

    Warnings
    --------
    Unlike most objects of :mod:`colour.phenomena.rayleigh` module,
    :func:`colour.scattering_cross_section` expects wavelength :math:`\\lambda`
    to be expressed in centimeters (cm).

    References
    ----------
    :cite:`Bodhaine1999a`, :cite:`Wikipedia2001c`

    Examples
    --------
    >>> scattering_cross_section(555 * 10e-8)  # doctest: +ELLIPSIS
    4.3466692...e-27
    """

    wl = as_float_array(wavelength)
    CO2_c = as_float_array(CO2_concentration)
    temperature = as_float_array(temperature)

    wl_micrometers = wl * 10e3

    N_s = molecular_density(temperature, avogadro_constant)
    n_s = n_s_function(wl_micrometers)
    # n_s = n_s_function(**filter_kwargs(
    #     n_s_function, wavelength=wl_micrometers, CO2_concentration=CO2_c))
    F_air = F_air_function(
        **filter_kwargs(
            F_air_function, wavelength=wl_micrometers, CO2_concentration=CO2_c
        )
    )

    sigma = 24 * np.pi**3 * (n_s**2 - 1) ** 2 / (wl**4 * N_s**2 * (n_s**2 + 2) ** 2)
    sigma *= F_air

    return sigma


def rayleigh_optical_depth(
    wavelength: ArrayLike,
    CO2_concentration: ArrayLike = CONSTANT_STANDARD_CO2_CONCENTRATION,
    temperature: ArrayLike = CONSTANT_STANDARD_AIR_TEMPERATURE,
    pressure: ArrayLike = CONSTANT_AVERAGE_PRESSURE_MEAN_SEA_LEVEL,
    latitude: ArrayLike = CONSTANT_DEFAULT_LATITUDE,
    altitude: ArrayLike = CONSTANT_DEFAULT_ALTITUDE,
    avogadro_constant: ArrayLike = CONSTANT_AVOGADRO,
    n_s_function: Callable = air_refraction_index_Bodhaine1999,
    F_air_function: Callable = F_air_Bodhaine1999,
) -> NDArrayFloat:
    """
    Return the *Rayleigh* optical depth :math:`T_r(\\lambda)` as function of
    wavelength :math:`\\lambda` in centimeters (cm).

    Parameters
    ----------
    wavelength
        Wavelength :math:`\\lambda` in centimeters (cm).
    CO2_concentration
        :math:`CO_2` concentration in parts per million (ppm).
    temperature
        Air temperature :math:`T[K]` in kelvin degrees.
    pressure
        Surface pressure :math:`P` of the measurement site.
    latitude
        Latitude of the site in degrees.
    altitude
        Altitude of the site in meters.
    avogadro_constant
        *Avogadro*'s number (molecules :math:`mol^{-1}`).
    n_s_function
        Air refraction index :math:`n_s` computation method.
    F_air_function
        :math:`(6+3_p)/(6-7_p)`, the depolarisation term :math:`F(air)` or
        *King Factor* computation method.

    Returns
    -------
    :class:`numpy.ndarray`
        *Rayleigh* optical depth :math:`T_r(\\lambda)`.

    Warnings
    --------
    Unlike most objects of :mod:`colour.phenomena.rayleigh` module,
    :func:`colour.phenomena.rayleigh_optical_depth` expects wavelength
    :math:`\\lambda` to be expressed in centimeters (cm).

    References
    ----------
    :cite:`Bodhaine1999a`, :cite:`Wikipedia2001c`

    Examples
    --------
    >>> rayleigh_optical_depth(555 * 10e-8)  # doctest: +ELLIPSIS
    0.0936290...
    """

    wavelength = as_float_array(wavelength)
    CO2_c = as_float_array(CO2_concentration)
    pressure = as_float_array(pressure)
    latitude = as_float_array(latitude)
    altitude = as_float_array(altitude)
    avogadro_constant = as_float_array(avogadro_constant)
    # Conversion from pascal to dyne/cm2.
    P = as_float_array(pressure * 10)

    sigma = scattering_cross_section(
        wavelength,
        CO2_c,
        temperature,
        avogadro_constant,
        n_s_function,
        F_air_function,
    )

    m_a = mean_molecular_weights(CO2_c)
    g = gravity_List1968(latitude, altitude)

    T_R = sigma * (P * avogadro_constant) / (m_a * g)

    return as_float(T_R)


rayleigh_scattering = rayleigh_optical_depth


def sd_rayleigh_scattering(
    shape: SpectralShape = SPECTRAL_SHAPE_DEFAULT,
    CO2_concentration: ArrayLike = CONSTANT_STANDARD_CO2_CONCENTRATION,
    temperature: ArrayLike = CONSTANT_STANDARD_AIR_TEMPERATURE,
    pressure: ArrayLike = CONSTANT_AVERAGE_PRESSURE_MEAN_SEA_LEVEL,
    latitude: ArrayLike = CONSTANT_DEFAULT_LATITUDE,
    altitude: ArrayLike = CONSTANT_DEFAULT_ALTITUDE,
    avogadro_constant: ArrayLike = CONSTANT_AVOGADRO,
    n_s_function: Callable = air_refraction_index_Bodhaine1999,
    F_air_function: Callable = F_air_Bodhaine1999,
) -> SpectralDistribution:
    """
    Return the *Rayleigh* spectral distribution for given spectral shape.

    Parameters
    ----------
    shape
        Spectral shape used to create the *Rayleigh* scattering spectral
        distribution.
    CO2_concentration
        :math:`CO_2` concentration in parts per million (ppm).
    temperature
        Air temperature :math:`T[K]` in kelvin degrees.
    pressure
        Surface pressure :math:`P` of the measurement site.
    latitude
        Latitude of the site in degrees.
    altitude
        Altitude of the site in meters.
    avogadro_constant
        *Avogadro*'s number (molecules :math:`mol^{-1}`).
    n_s_function
        Air refraction index :math:`n_s` computation method.
    F_air_function
        :math:`(6+3_p)/(6-7_p)`, the depolarisation term :math:`F(air)` or
        *King Factor* computation method.

    Returns
    -------
    :class:`colour.SpectralDistribution`
        *Rayleigh* optical depth spectral distribution.

    References
    ----------
    :cite:`Bodhaine1999a`, :cite:`Wikipedia2001c`

    Examples
    --------
    >>> from colour.utilities import numpy_print_options
    >>> with numpy_print_options(suppress=True):
    ...     sd_rayleigh_scattering()  # doctest: +ELLIPSIS
    SpectralDistribution([[ 360.        ,    0.5602465...],
                          [ 361.        ,    0.5537481...],
                          [ 362.        ,    0.5473446...],
                          [ 363.        ,    0.5410345...],
                          [ 364.        ,    0.5348161...],
                          [ 365.        ,    0.5286877...],
                          [ 366.        ,    0.5226477...],
                          [ 367.        ,    0.5166948...],
                          [ 368.        ,    0.5108272...],
                          [ 369.        ,    0.5050436...],
                          [ 370.        ,    0.4993425...],
                          [ 371.        ,    0.4937224...],
                          [ 372.        ,    0.4881820...],
                          [ 373.        ,    0.4827199...],
                          [ 374.        ,    0.4773348...],
                          [ 375.        ,    0.4720253...],
                          [ 376.        ,    0.4667902...],
                          [ 377.        ,    0.4616282...],
                          [ 378.        ,    0.4565380...],
                          [ 379.        ,    0.4515186...],
                          [ 380.        ,    0.4465686...],
                          [ 381.        ,    0.4416869...],
                          [ 382.        ,    0.4368724...],
                          [ 383.        ,    0.4321240...],
                          [ 384.        ,    0.4274405...],
                          [ 385.        ,    0.4228209...],
                          [ 386.        ,    0.4182641...],
                          [ 387.        ,    0.4137692...],
                          [ 388.        ,    0.4093350...],
                          [ 389.        ,    0.4049607...],
                          [ 390.        ,    0.4006451...],
                          [ 391.        ,    0.3963874...],
                          [ 392.        ,    0.3921867...],
                          [ 393.        ,    0.3880419...],
                          [ 394.        ,    0.3839523...],
                          [ 395.        ,    0.3799169...],
                          [ 396.        ,    0.3759348...],
                          [ 397.        ,    0.3720053...],
                          [ 398.        ,    0.3681274...],
                          [ 399.        ,    0.3643003...],
                          [ 400.        ,    0.3605233...],
                          [ 401.        ,    0.3567956...],
                          [ 402.        ,    0.3531163...],
                          [ 403.        ,    0.3494847...],
                          [ 404.        ,    0.3459001...],
                          [ 405.        ,    0.3423617...],
                          [ 406.        ,    0.3388689...],
                          [ 407.        ,    0.3354208...],
                          [ 408.        ,    0.3320169...],
                          [ 409.        ,    0.3286563...],
                          [ 410.        ,    0.3253386...],
                          [ 411.        ,    0.3220629...],
                          [ 412.        ,    0.3188287...],
                          [ 413.        ,    0.3156354...],
                          [ 414.        ,    0.3124822...],
                          [ 415.        ,    0.3093687...],
                          [ 416.        ,    0.3062941...],
                          [ 417.        ,    0.3032579...],
                          [ 418.        ,    0.3002596...],
                          [ 419.        ,    0.2972985...],
                          [ 420.        ,    0.2943741...],
                          [ 421.        ,    0.2914858...],
                          [ 422.        ,    0.2886332...],
                          [ 423.        ,    0.2858157...],
                          [ 424.        ,    0.2830327...],
                          [ 425.        ,    0.2802837...],
                          [ 426.        ,    0.2775683...],
                          [ 427.        ,    0.2748860...],
                          [ 428.        ,    0.2722362...],
                          [ 429.        ,    0.2696185...],
                          [ 430.        ,    0.2670324...],
                          [ 431.        ,    0.2644775...],
                          [ 432.        ,    0.2619533...],
                          [ 433.        ,    0.2594594...],
                          [ 434.        ,    0.2569952...],
                          [ 435.        ,    0.2545605...],
                          [ 436.        ,    0.2521548...],
                          [ 437.        ,    0.2497776...],
                          [ 438.        ,    0.2474285...],
                          [ 439.        ,    0.2451072...],
                          [ 440.        ,    0.2428133...],
                          [ 441.        ,    0.2405463...],
                          [ 442.        ,    0.2383059...],
                          [ 443.        ,    0.2360916...],
                          [ 444.        ,    0.2339033...],
                          [ 445.        ,    0.2317404...],
                          [ 446.        ,    0.2296026...],
                          [ 447.        ,    0.2274895...],
                          [ 448.        ,    0.2254009...],
                          [ 449.        ,    0.2233364...],
                          [ 450.        ,    0.2212956...],
                          [ 451.        ,    0.2192782...],
                          [ 452.        ,    0.2172839...],
                          [ 453.        ,    0.2153124...],
                          [ 454.        ,    0.2133633...],
                          [ 455.        ,    0.2114364...],
                          [ 456.        ,    0.2095313...],
                          [ 457.        ,    0.2076478...],
                          [ 458.        ,    0.2057855...],
                          [ 459.        ,    0.2039442...],
                          [ 460.        ,    0.2021235...],
                          [ 461.        ,    0.2003233...],
                          [ 462.        ,    0.1985432...],
                          [ 463.        ,    0.1967829...],
                          [ 464.        ,    0.1950423...],
                          [ 465.        ,    0.1933209...],
                          [ 466.        ,    0.1916186...],
                          [ 467.        ,    0.1899351...],
                          [ 468.        ,    0.1882702...],
                          [ 469.        ,    0.1866236...],
                          [ 470.        ,    0.1849951...],
                          [ 471.        ,    0.1833844...],
                          [ 472.        ,    0.1817913...],
                          [ 473.        ,    0.1802156...],
                          [ 474.        ,    0.1786570...],
                          [ 475.        ,    0.1771153...],
                          [ 476.        ,    0.1755903...],
                          [ 477.        ,    0.1740818...],
                          [ 478.        ,    0.1725895...],
                          [ 479.        ,    0.1711133...],
                          [ 480.        ,    0.1696529...],
                          [ 481.        ,    0.1682082...],
                          [ 482.        ,    0.1667789...],
                          [ 483.        ,    0.1653648...],
                          [ 484.        ,    0.1639658...],
                          [ 485.        ,    0.1625816...],
                          [ 486.        ,    0.1612121...],
                          [ 487.        ,    0.1598570...],
                          [ 488.        ,    0.1585163...],
                          [ 489.        ,    0.1571896...],
                          [ 490.        ,    0.1558769...],
                          [ 491.        ,    0.1545779...],
                          [ 492.        ,    0.1532925...],
                          [ 493.        ,    0.1520205...],
                          [ 494.        ,    0.1507617...],
                          [ 495.        ,    0.1495160...],
                          [ 496.        ,    0.1482832...],
                          [ 497.        ,    0.1470632...],
                          [ 498.        ,    0.1458558...],
                          [ 499.        ,    0.1446607...],
                          [ 500.        ,    0.1434780...],
                          [ 501.        ,    0.1423074...],
                          [ 502.        ,    0.1411488...],
                          [ 503.        ,    0.140002 ...],
                          [ 504.        ,    0.1388668...],
                          [ 505.        ,    0.1377433...],
                          [ 506.        ,    0.1366311...],
                          [ 507.        ,    0.1355301...],
                          [ 508.        ,    0.1344403...],
                          [ 509.        ,    0.1333615...],
                          [ 510.        ,    0.1322936...],
                          [ 511.        ,    0.1312363...],
                          [ 512.        ,    0.1301897...],
                          [ 513.        ,    0.1291535...],
                          [ 514.        ,    0.1281277...],
                          [ 515.        ,    0.1271121...],
                          [ 516.        ,    0.1261065...],
                          [ 517.        ,    0.1251110...],
                          [ 518.        ,    0.1241253...],
                          [ 519.        ,    0.1231493...],
                          [ 520.        ,    0.1221829...],
                          [ 521.        ,    0.1212261...],
                          [ 522.        ,    0.1202786...],
                          [ 523.        ,    0.1193405...],
                          [ 524.        ,    0.1184115...],
                          [ 525.        ,    0.1174915...],
                          [ 526.        ,    0.1165806...],
                          [ 527.        ,    0.1156784...],
                          [ 528.        ,    0.1147851...],
                          [ 529.        ,    0.1139004...],
                          [ 530.        ,    0.1130242...],
                          [ 531.        ,    0.1121564...],
                          [ 532.        ,    0.1112971...],
                          [ 533.        ,    0.1104459...],
                          [ 534.        ,    0.1096030...],
                          [ 535.        ,    0.1087681...],
                          [ 536.        ,    0.1079411...],
                          [ 537.        ,    0.1071221...],
                          [ 538.        ,    0.1063108...],
                          [ 539.        ,    0.1055072...],
                          [ 540.        ,    0.1047113...],
                          [ 541.        ,    0.1039229...],
                          [ 542.        ,    0.1031419...],
                          [ 543.        ,    0.1023682...],
                          [ 544.        ,    0.1016019...],
                          [ 545.        ,    0.1008427...],
                          [ 546.        ,    0.1000906...],
                          [ 547.        ,    0.0993456...],
                          [ 548.        ,    0.0986075...],
                          [ 549.        ,    0.0978763...],
                          [ 550.        ,    0.0971519...],
                          [ 551.        ,    0.0964342...],
                          [ 552.        ,    0.0957231...],
                          [ 553.        ,    0.0950186...],
                          [ 554.        ,    0.0943206...],
                          [ 555.        ,    0.0936290...],
                          [ 556.        ,    0.0929438...],
                          [ 557.        ,    0.0922649...],
                          [ 558.        ,    0.0915922...],
                          [ 559.        ,    0.0909256...],
                          [ 560.        ,    0.0902651...],
                          [ 561.        ,    0.0896106...],
                          [ 562.        ,    0.0889620...],
                          [ 563.        ,    0.0883194...],
                          [ 564.        ,    0.0876825...],
                          [ 565.        ,    0.0870514...],
                          [ 566.        ,    0.0864260...],
                          [ 567.        ,    0.0858063...],
                          [ 568.        ,    0.0851921...],
                          [ 569.        ,    0.0845834...],
                          [ 570.        ,    0.0839801...],
                          [ 571.        ,    0.0833822...],
                          [ 572.        ,    0.0827897...],
                          [ 573.        ,    0.0822025...],
                          [ 574.        ,    0.0816204...],
                          [ 575.        ,    0.0810436...],
                          [ 576.        ,    0.0804718...],
                          [ 577.        ,    0.0799051...],
                          [ 578.        ,    0.0793434...],
                          [ 579.        ,    0.0787866...],
                          [ 580.        ,    0.0782347...],
                          [ 581.        ,    0.0776877...],
                          [ 582.        ,    0.0771454...],
                          [ 583.        ,    0.0766079...],
                          [ 584.        ,    0.0760751...],
                          [ 585.        ,    0.0755469...],
                          [ 586.        ,    0.0750234...],
                          [ 587.        ,    0.0745043...],
                          [ 588.        ,    0.0739898...],
                          [ 589.        ,    0.0734797...],
                          [ 590.        ,    0.0729740...],
                          [ 591.        ,    0.0724727...],
                          [ 592.        ,    0.0719757...],
                          [ 593.        ,    0.0714830...],
                          [ 594.        ,    0.0709944...],
                          [ 595.        ,    0.0705101...],
                          [ 596.        ,    0.0700299...],
                          [ 597.        ,    0.0695538...],
                          [ 598.        ,    0.0690818...],
                          [ 599.        ,    0.0686137...],
                          [ 600.        ,    0.0681497...],
                          [ 601.        ,    0.0676895...],
                          [ 602.        ,    0.0672333...],
                          [ 603.        ,    0.0667809...],
                          [ 604.        ,    0.0663323...],
                          [ 605.        ,    0.0658875...],
                          [ 606.        ,    0.0654464...],
                          [ 607.        ,    0.0650091...],
                          [ 608.        ,    0.0645753...],
                          [ 609.        ,    0.0641453...],
                          [ 610.        ,    0.0637187...],
                          [ 611.        ,    0.0632958...],
                          [ 612.        ,    0.0628764...],
                          [ 613.        ,    0.0624604...],
                          [ 614.        ,    0.0620479...],
                          [ 615.        ,    0.0616388...],
                          [ 616.        ,    0.0612331...],
                          [ 617.        ,    0.0608307...],
                          [ 618.        ,    0.0604316...],
                          [ 619.        ,    0.0600358...],
                          [ 620.        ,    0.0596433...],
                          [ 621.        ,    0.0592539...],
                          [ 622.        ,    0.0588678...],
                          [ 623.        ,    0.0584848...],
                          [ 624.        ,    0.0581049...],
                          [ 625.        ,    0.0577281...],
                          [ 626.        ,    0.0573544...],
                          [ 627.        ,    0.0569837...],
                          [ 628.        ,    0.0566160...],
                          [ 629.        ,    0.0562513...],
                          [ 630.        ,    0.0558895...],
                          [ 631.        ,    0.0555306...],
                          [ 632.        ,    0.0551746...],
                          [ 633.        ,    0.0548215...],
                          [ 634.        ,    0.0544712...],
                          [ 635.        ,    0.0541237...],
                          [ 636.        ,    0.0537789...],
                          [ 637.        ,    0.0534369...],
                          [ 638.        ,    0.0530977...],
                          [ 639.        ,    0.0527611...],
                          [ 640.        ,    0.0524272...],
                          [ 641.        ,    0.0520960...],
                          [ 642.        ,    0.0517674...],
                          [ 643.        ,    0.0514413...],
                          [ 644.        ,    0.0511179...],
                          [ 645.        ,    0.0507970...],
                          [ 646.        ,    0.0504786...],
                          [ 647.        ,    0.0501627...],
                          [ 648.        ,    0.0498493...],
                          [ 649.        ,    0.0495383...],
                          [ 650.        ,    0.0492298...],
                          [ 651.        ,    0.0489236...],
                          [ 652.        ,    0.0486199...],
                          [ 653.        ,    0.0483185...],
                          [ 654.        ,    0.0480194...],
                          [ 655.        ,    0.0477227...],
                          [ 656.        ,    0.0474283...],
                          [ 657.        ,    0.0471361...],
                          [ 658.        ,    0.0468462...],
                          [ 659.        ,    0.0465585...],
                          [ 660.        ,    0.0462730...],
                          [ 661.        ,    0.0459898...],
                          [ 662.        ,    0.0457087...],
                          [ 663.        ,    0.0454297...],
                          [ 664.        ,    0.0451529...],
                          [ 665.        ,    0.0448782...],
                          [ 666.        ,    0.0446055...],
                          [ 667.        ,    0.0443350...],
                          [ 668.        ,    0.0440665...],
                          [ 669.        ,    0.0438000...],
                          [ 670.        ,    0.0435356...],
                          [ 671.        ,    0.0432731...],
                          [ 672.        ,    0.0430127...],
                          [ 673.        ,    0.0427542...],
                          [ 674.        ,    0.0424976...],
                          [ 675.        ,    0.0422430...],
                          [ 676.        ,    0.0419902...],
                          [ 677.        ,    0.0417394...],
                          [ 678.        ,    0.0414905...],
                          [ 679.        ,    0.0412434...],
                          [ 680.        ,    0.0409981...],
                          [ 681.        ,    0.0407547...],
                          [ 682.        ,    0.0405131...],
                          [ 683.        ,    0.0402732...],
                          [ 684.        ,    0.0400352...],
                          [ 685.        ,    0.0397989...],
                          [ 686.        ,    0.0395643...],
                          [ 687.        ,    0.0393315...],
                          [ 688.        ,    0.0391004...],
                          [ 689.        ,    0.0388710...],
                          [ 690.        ,    0.0386433...],
                          [ 691.        ,    0.0384173...],
                          [ 692.        ,    0.0381929...],
                          [ 693.        ,    0.0379701...],
                          [ 694.        ,    0.0377490...],
                          [ 695.        ,    0.0375295...],
                          [ 696.        ,    0.0373115...],
                          [ 697.        ,    0.0370952...],
                          [ 698.        ,    0.0368804...],
                          [ 699.        ,    0.0366672...],
                          [ 700.        ,    0.0364556...],
                          [ 701.        ,    0.0362454...],
                          [ 702.        ,    0.0360368...],
                          [ 703.        ,    0.0358297...],
                          [ 704.        ,    0.0356241...],
                          [ 705.        ,    0.0354199...],
                          [ 706.        ,    0.0352172...],
                          [ 707.        ,    0.0350160...],
                          [ 708.        ,    0.0348162...],
                          [ 709.        ,    0.0346178...],
                          [ 710.        ,    0.0344208...],
                          [ 711.        ,    0.0342253...],
                          [ 712.        ,    0.0340311...],
                          [ 713.        ,    0.0338383...],
                          [ 714.        ,    0.0336469...],
                          [ 715.        ,    0.0334569...],
                          [ 716.        ,    0.0332681...],
                          [ 717.        ,    0.0330807...],
                          [ 718.        ,    0.0328947...],
                          [ 719.        ,    0.0327099...],
                          [ 720.        ,    0.0325264...],
                          [ 721.        ,    0.0323443...],
                          [ 722.        ,    0.0321634...],
                          [ 723.        ,    0.0319837...],
                          [ 724.        ,    0.0318054...],
                          [ 725.        ,    0.0316282...],
                          [ 726.        ,    0.0314523...],
                          [ 727.        ,    0.0312777...],
                          [ 728.        ,    0.0311042...],
                          [ 729.        ,    0.0309319...],
                          [ 730.        ,    0.0307609...],
                          [ 731.        ,    0.0305910...],
                          [ 732.        ,    0.0304223...],
                          [ 733.        ,    0.0302548...],
                          [ 734.        ,    0.0300884...],
                          [ 735.        ,    0.0299231...],
                          [ 736.        ,    0.0297590...],
                          [ 737.        ,    0.0295960...],
                          [ 738.        ,    0.0294342...],
                          [ 739.        ,    0.0292734...],
                          [ 740.        ,    0.0291138...],
                          [ 741.        ,    0.0289552...],
                          [ 742.        ,    0.0287977...],
                          [ 743.        ,    0.0286413...],
                          [ 744.        ,    0.0284859...],
                          [ 745.        ,    0.0283316...],
                          [ 746.        ,    0.0281784...],
                          [ 747.        ,    0.0280262...],
                          [ 748.        ,    0.0278750...],
                          [ 749.        ,    0.0277248...],
                          [ 750.        ,    0.0275757...],
                          [ 751.        ,    0.0274275...],
                          [ 752.        ,    0.0272804...],
                          [ 753.        ,    0.0271342...],
                          [ 754.        ,    0.0269890...],
                          [ 755.        ,    0.0268448...],
                          [ 756.        ,    0.0267015...],
                          [ 757.        ,    0.0265592...],
                          [ 758.        ,    0.0264179...],
                          [ 759.        ,    0.0262775...],
                          [ 760.        ,    0.0261380...],
                          [ 761.        ,    0.0259995...],
                          [ 762.        ,    0.0258618...],
                          [ 763.        ,    0.0257251...],
                          [ 764.        ,    0.0255893...],
                          [ 765.        ,    0.0254544...],
                          [ 766.        ,    0.0253204...],
                          [ 767.        ,    0.0251872...],
                          [ 768.        ,    0.0250550...],
                          [ 769.        ,    0.0249236...],
                          [ 770.        ,    0.0247930...],
                          [ 771.        ,    0.0246633...],
                          [ 772.        ,    0.0245345...],
                          [ 773.        ,    0.0244065...],
                          [ 774.        ,    0.0242794...],
                          [ 775.        ,    0.0241530...],
                          [ 776.        ,    0.0240275...],
                          [ 777.        ,    0.0239029...],
                          [ 778.        ,    0.0237790...],
                          [ 779.        ,    0.0236559...],
                          [ 780.        ,    0.0235336...]],
                         SpragueInterpolator,
                         {},
                         Extrapolator,
                         {'method': 'Constant', 'left': None, 'right': None})
    """

    return SpectralDistribution(
        rayleigh_optical_depth(
            shape.wavelengths * 10e-8,
            CO2_concentration,
            temperature,
            pressure,
            latitude,
            altitude,
            avogadro_constant,
            n_s_function,
            F_air_function,
        ),
        shape.wavelengths,
        name=(
            "Rayleigh Scattering - "
            f"{CO2_concentration!r} ppm, "
            f"{temperature!r} K, "
            f"{pressure!r} Pa, "
            f"{latitude!r} Degrees, "
            f"{altitude!r} m"
        ),
    )
