# -*- coding: utf-8 -*-
"""
Rayleigh Optical Depth - Scattering in the Atmosphere
=====================================================

Implements *Rayleigh* scattering / optical depth in the atmosphere computation:

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

import numpy as np

from colour.colorimetry import (
    SPECTRAL_SHAPE_DEFAULT,
    SpectralDistribution,
)
from colour.constants import CONSTANT_AVOGADRO
from colour.utilities import as_float, as_float_array, filter_kwargs

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'CONSTANT_STANDARD_AIR_TEMPERATURE',
    'CONSTANT_STANDARD_CO2_CONCENTRATION',
    'CONSTANT_AVERAGE_PRESSURE_MEAN_SEA_LEVEL',
    'CONSTANT_DEFAULT_LATITUDE',
    'CONSTANT_DEFAULT_ALTITUDE',
    'air_refraction_index_Penndorf1957',
    'air_refraction_index_Edlen1966',
    'air_refraction_index_Peck1972',
    'air_refraction_index_Bodhaine1999',
    'N2_depolarisation',
    'O2_depolarisation',
    'F_air_Penndorf1957',
    'F_air_Young1981',
    'F_air_Bates1984',
    'F_air_Bodhaine1999',
    'molecular_density',
    'mean_molecular_weights',
    'gravity_List1968',
    'scattering_cross_section',
    'rayleigh_optical_depth',
    'rayleigh_scattering',
]

CONSTANT_STANDARD_AIR_TEMPERATURE = 288.15
"""
*Standard air* temperature :math:`T[K]` in kelvin degrees (:math:`15\\circ C`).

CONSTANT_STANDARD_AIR_TEMPERATURE : numeric
"""

CONSTANT_STANDARD_CO2_CONCENTRATION = 300
"""
*Standard air* :math:`CO_2` concentration in parts per million (ppm).

CONSTANT_STANDARD_CO2_CONCENTRATION : numeric
"""

CONSTANT_AVERAGE_PRESSURE_MEAN_SEA_LEVEL = 101325
"""
*Standard air* average pressure :math:`Hg` at mean sea-level in pascal (Pa).

CONSTANT_AVERAGE_PRESSURE_MEAN_SEA_LEVEL : numeric
"""

CONSTANT_DEFAULT_LATITUDE = 0
"""
Default latitude in degrees (equator).

CONSTANT_DEFAULT_LATITUDE : numeric
"""

CONSTANT_DEFAULT_ALTITUDE = 0
"""
Default altitude in meters (sea level).

CONSTANT_DEFAULT_ALTITUDE : numeric
"""


def air_refraction_index_Penndorf1957(wavelength):
    """
    Returns the air refraction index :math:`n_s` from given wavelength
    :math:`\\lambda` in  micrometers (:math:`\\mu m`) using *Penndorf (1957)*
    method.

    Parameters
    ----------
    wavelength : numeric or array_like
        Wavelength :math:`\\lambda` in micrometers (:math:`\\mu m`).

    Returns
    -------
    numeric or ndarray
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


def air_refraction_index_Edlen1966(wavelength):
    """
    Returns the air refraction index :math:`n_s` from given wavelength
    :math:`\\lambda` in micrometers (:math:`\\mu m`) using *Edlen (1966)*
    method.

    Parameters
    ----------
    wavelength : numeric or array_like
        Wavelength :math:`\\lambda` in micrometers (:math:`\\mu m`).

    Returns
    -------
    numeric or ndarray
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


def air_refraction_index_Peck1972(wavelength):
    """
    Returns the air refraction index :math:`n_s` from given wavelength
    :math:`\\lambda` in micrometers (:math:`\\mu m`) using
    *Peck and Reeder (1972)* method.

    Parameters
    ----------
    wavelength : numeric or array_like
        Wavelength :math:`\\lambda` in micrometers (:math:`\\mu m`).

    Returns
    -------
    numeric or ndarray
        Air refraction index :math:`n_s`.

    Examples
    --------
    >>> air_refraction_index_Peck1972(0.555)  # doctest: +ELLIPSIS
    1.0002777...
    """

    wl = as_float_array(wavelength)

    n = (8060.51 + 2480990 / (132.274 - wl **
                              (-2)) + 17455.7 / (39.32957 - wl ** (-2)))
    n /= 1.0e8
    n += +1

    return n


def air_refraction_index_Bodhaine1999(
        wavelength, CO2_concentration=CONSTANT_STANDARD_CO2_CONCENTRATION):
    """
    Returns the air refraction index :math:`n_s` from given wavelength
    :math:`\\lambda` in micrometers (:math:`\\mu m`) using
    *Bodhaine, Wood, Dutton and Slusser (1999)* method.

    Parameters
    ----------
    wavelength : numeric or array_like
        Wavelength :math:`\\lambda` in micrometers (:math:`\\mu m`).
    CO2_concentration : numeric or array_like
        :math:`CO_2` concentration in parts per million (ppm).

    Returns
    -------
    numeric or ndarray
        Air refraction index :math:`n_s`.

    Examples
    --------
    >>> air_refraction_index_Bodhaine1999(0.555)  # doctest: +ELLIPSIS
    1.0002777...
    """

    wl = as_float_array(wavelength)
    CO2_c = as_float_array(CO2_concentration)

    n = ((1 + 0.54 * ((CO2_c * 1e-6) - 300e-6)) *
         (air_refraction_index_Peck1972(wl) - 1) + 1)

    return n


def N2_depolarisation(wavelength):
    """
    Returns the depolarisation of nitrogen :math:`N_2` as function of
    wavelength :math:`\\lambda` in micrometers (:math:`\\mu m`).

    Parameters
    ----------
    wavelength : numeric or array_like
        Wavelength :math:`\\lambda` in micrometers (:math:`\\mu m`).

    Returns
    -------
    numeric or ndarray
        Nitrogen :math:`N_2` depolarisation.

    Examples
    --------
    >>> N2_depolarisation(0.555)  # doctest: +ELLIPSIS
    1.0350291...
    """

    wl = as_float_array(wavelength)

    N2 = 1.034 + 3.17 * 1.0e-4 * (1 / wl ** 2)

    return N2


def O2_depolarisation(wavelength):
    """
    Returns the depolarisation of oxygen :math:`O_2` as function of
    wavelength :math:`\\lambda` in micrometers (:math:`\\mu m`).

    Parameters
    ----------
    wavelength : numeric or array_like
        Wavelength :math:`\\lambda` in micrometers (:math:`\\mu m`).

    Returns
    -------
    numeric or ndarray
        Oxygen :math:`O_2` depolarisation.

    Examples
    --------
    >>> O2_depolarisation(0.555)  # doctest: +ELLIPSIS
    1.1020225...
    """

    wl = as_float_array(wavelength)

    O2 = (1.096 + 1.385 * 1.0e-3 * (1 / wl ** 2) +
          1.448 * 1.0e-4 * (1 / wl ** 4))

    return O2


def F_air_Penndorf1957(wavelength):
    """
    Returns :math:`(6+3_p)/(6-7_p)`, the depolarisation term :math:`F(air)` or
    *King Factor* using *Penndorf (1957)* method.

    Parameters
    ----------
    wavelength : numeric or array_like
        Wavelength :math:`\\lambda` in micrometers (:math:`\\mu m`).

    Returns
    -------
    numeric or ndarray
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


def F_air_Young1981(wavelength):
    """
    Returns :math:`(6+3_p)/(6-7_p)`, the depolarisation term :math:`F(air)` or
    *King Factor* using *Young (1981)* method.

    Parameters
    ----------
    wavelength : numeric or array_like
        Wavelength :math:`\\lambda` in micrometers (:math:`\\mu m`).

    Returns
    -------
    numeric or ndarray
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


def F_air_Bates1984(wavelength):
    """
    Returns :math:`(6+3_p)/(6-7_p)`, the depolarisation term :math:`F(air)` or
    *King Factor* as function of wavelength :math:`\\lambda` in micrometers
    (:math:`\\mu m`) using *Bates (1984)* method.

    Parameters
    ----------
    wavelength : numeric or array_like
        Wavelength :math:`\\lambda` in micrometers (:math:`\\mu m`).

    Returns
    -------
    numeric or ndarray
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

    F_air = (
        (78.084 * N2 + 20.946 * O2 + CO2 + Ar) / (78.084 + 20.946 + Ar + CO2))

    return F_air


def F_air_Bodhaine1999(wavelength,
                       CO2_concentration=CONSTANT_STANDARD_CO2_CONCENTRATION):
    """
    Returns :math:`(6+3_p)/(6-7_p)`, the depolarisation term :math:`F(air)` or
    *King Factor* as function of wavelength :math:`\\lambda` in micrometers
    (:math:`\\mu m`) and :math:`CO_2` concentration in parts per million (ppm)
    using *Bodhaine, Wood, Dutton and Slusser (1999)* method.

    Parameters
    ----------
    wavelength : numeric or array_like
        Wavelength :math:`\\lambda` in micrometers (:math:`\\mu m`).
    CO2_concentration : numeric or array_like, optional
        :math:`CO_2` concentration in parts per million (ppm).

    Returns
    -------
    numeric or ndarray
        Air depolarisation.

    Examples
    --------
    >>> F_air_Bodhaine1999(0.555)  # doctest: +ELLIPSIS
    1.1246916...
    """

    O2 = O2_depolarisation(wavelength)
    N2 = N2_depolarisation(wavelength)
    CO2_c = as_float_array(CO2_concentration)

    F_air = ((78.084 * N2 + 20.946 * O2 + 0.934 * 1 + CO2_c * 1.15) /
             (78.084 + 20.946 + 0.934 + CO2_c))

    return F_air


def molecular_density(temperature=CONSTANT_STANDARD_AIR_TEMPERATURE,
                      avogadro_constant=CONSTANT_AVOGADRO):
    """
    Returns the molecular density :math:`N_s` (molecules :math:`cm^{-3}`)
    as function of air temperature :math:`T[K]` in kelvin degrees.

    Parameters
    ----------
    temperature : numeric or array_like, optional
        Air temperature :math:`T[K]` in kelvin degrees.
    avogadro_constant : numeric or array_like, optional
        *Avogadro*'s number (molecules :math:`mol^{-1}`).

    Returns
    -------
    numeric or ndarray
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

    N_s = (avogadro_constant / 22.4141) * (273.15 / T) * (1 / 1000)

    return N_s


def mean_molecular_weights(
        CO2_concentration=CONSTANT_STANDARD_CO2_CONCENTRATION):
    """
    Returns the mean molecular weights :math:`m_a` for dry air as function of
    :math:`CO_2` concentration in parts per million (ppm).

    Parameters
    ----------
    CO2_concentration : numeric or array_like, optional
        :math:`CO_2` concentration in parts per million (ppm).

    Returns
    -------
    numeric or ndarray
        Mean molecular weights :math:`m_a` for dry air.

    Examples
    --------
    >>> mean_molecular_weights()  # doctest: +ELLIPSIS
    28.9640166...
    """

    CO2_c = CO2_concentration * 1.0e-6

    m_a = 15.0556 * CO2_c + 28.9595
    return m_a


def gravity_List1968(latitude=CONSTANT_DEFAULT_LATITUDE,
                     altitude=CONSTANT_DEFAULT_ALTITUDE):
    """
    Returns the gravity :math:`g` in :math:`cm/s_2` (gal) representative of the
    mass-weighted column of air molecules above the site of given latitude and
    altitude using *List (1968)* method.

    Parameters
    ----------
    latitude : numeric or array_like, optional
        Latitude of the site in degrees.
    altitude : numeric or array_like, optional
        Altitude of the site in meters.

    Returns
    -------
    numeric or ndarray
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
    g0 = 980.6160 * (1 - 0.0026373 * cos2phi + 0.0000059 * cos2phi ** 2)

    g = (g0 - (3.085462e-4 + 2.27e-7 * cos2phi) * altitude +
         (7.254e-11 + 1.0e-13 * cos2phi) * altitude ** 2 -
         (1.517e-17 + 6e-20 * cos2phi) * altitude ** 3)

    return g


def scattering_cross_section(
        wavelength,
        CO2_concentration=CONSTANT_STANDARD_CO2_CONCENTRATION,
        temperature=CONSTANT_STANDARD_AIR_TEMPERATURE,
        avogadro_constant=CONSTANT_AVOGADRO,
        n_s=air_refraction_index_Bodhaine1999,
        F_air=F_air_Bodhaine1999):
    """
    Returns the scattering cross section per molecule :math:`\\sigma` of dry
    air as function of wavelength :math:`\\lambda` in centimeters (cm) using
    given :math:`CO_2` concentration in parts per million (ppm) and temperature
    :math:`T[K]` in kelvin degrees following *Van de Hulst (1957)* method.

    Parameters
    ----------
    wavelength : numeric or array_like
        Wavelength :math:`\\lambda` in centimeters (cm).
    CO2_concentration : numeric or array_like, optional
        :math:`CO_2` concentration in parts per million (ppm).
    temperature : numeric or array_like, optional
        Air temperature :math:`T[K]` in kelvin degrees.
    avogadro_constant : numeric or array_like, optional
        *Avogadro*'s number (molecules :math:`mol^{-1}`).
    n_s : object
        Air refraction index :math:`n_s` computation method.
    F_air : object
        :math:`(6+3_p)/(6-7_p)`, the depolarisation term :math:`F(air)` or
        *King Factor* computation method.

    Returns
    -------
    numeric or ndarray
        Scattering cross section per molecule :math:`\\sigma` of dry air.

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
    4.6613309...e-27
    """

    wl = as_float_array(wavelength)
    CO2_c = as_float_array(CO2_concentration)
    temperature = as_float_array(temperature)

    wl_micrometers = wl * 10e3

    n_s = n_s(wl_micrometers)
    # n_s = n_s(**filter_kwargs(
    #     n_s, wavelength=wl_micrometers, CO2_concentration=CO2_c))
    N_s = molecular_density(temperature, avogadro_constant)
    F_air = F_air(**filter_kwargs(
        F_air, wavelength=wl_micrometers, CO2_concentration=CO2_c))

    sigma = (24 * np.pi ** 3 * (n_s ** 2 - 1) ** 2 / (wl ** 4 * N_s ** 2 *
                                                      (n_s ** 2 + 2) ** 2))
    sigma *= F_air

    return sigma


def rayleigh_optical_depth(
        wavelength,
        CO2_concentration=CONSTANT_STANDARD_CO2_CONCENTRATION,
        temperature=CONSTANT_STANDARD_AIR_TEMPERATURE,
        pressure=CONSTANT_AVERAGE_PRESSURE_MEAN_SEA_LEVEL,
        latitude=CONSTANT_DEFAULT_LATITUDE,
        altitude=CONSTANT_DEFAULT_ALTITUDE,
        avogadro_constant=CONSTANT_AVOGADRO,
        n_s=air_refraction_index_Bodhaine1999,
        F_air=F_air_Bodhaine1999):
    """
    Returns the *Rayleigh* optical depth :math:`T_r(\\lambda)` as function of
    wavelength :math:`\\lambda` in centimeters (cm).

    Parameters
    ----------
    wavelength : numeric or array_like
        Wavelength :math:`\\lambda` in centimeters (cm).
    CO2_concentration : numeric or array_like, optional
        :math:`CO_2` concentration in parts per million (ppm).
    temperature : numeric or array_like, optional
        Air temperature :math:`T[K]` in kelvin degrees.
    pressure : numeric or array_like
        Surface pressure :math:`P` of the measurement site.
    latitude : numeric or array_like, optional
        Latitude of the site in degrees.
    altitude : numeric or array_like, optional
        Altitude of the site in meters.
    avogadro_constant : numeric or array_like, optional
        *Avogadro*'s number (molecules :math:`mol^{-1}`).
    n_s : object
        Air refraction index :math:`n_s` computation method.
    F_air : object
        :math:`(6+3_p)/(6-7_p)`, the depolarisation term :math:`F(air)` or
        *King Factor* computation method.

    Returns
    -------
    numeric or ndarray
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
    0.1004070...
    """

    wavelength = as_float_array(wavelength)
    CO2_c = as_float_array(CO2_concentration)
    latitude = as_float_array(latitude)
    altitude = as_float_array(altitude)
    # Conversion from pascal to dyne/cm2.
    P = as_float_array(pressure * 10)

    sigma = scattering_cross_section(wavelength, CO2_c, temperature,
                                     avogadro_constant, n_s, F_air)

    m_a = mean_molecular_weights(CO2_c)
    g = gravity_List1968(latitude, altitude)

    T_R = sigma * (P * avogadro_constant) / (m_a * g)

    return T_R


rayleigh_scattering = rayleigh_optical_depth


def sd_rayleigh_scattering(
        shape=SPECTRAL_SHAPE_DEFAULT,
        CO2_concentration=CONSTANT_STANDARD_CO2_CONCENTRATION,
        temperature=CONSTANT_STANDARD_AIR_TEMPERATURE,
        pressure=CONSTANT_AVERAGE_PRESSURE_MEAN_SEA_LEVEL,
        latitude=CONSTANT_DEFAULT_LATITUDE,
        altitude=CONSTANT_DEFAULT_ALTITUDE,
        avogadro_constant=CONSTANT_AVOGADRO,
        n_s=air_refraction_index_Bodhaine1999,
        F_air=F_air_Bodhaine1999):
    """
    Returns the *Rayleigh* spectral distribution for given spectral shape.

    Parameters
    ----------
    shape : SpectralShape, optional
        Spectral shape used to create the *Rayleigh* scattering spectral
        distribution.
    CO2_concentration : numeric or array_like, optional
        :math:`CO_2` concentration in parts per million (ppm).
    temperature : numeric or array_like, optional
        Air temperature :math:`T[K]` in kelvin degrees.
    pressure : numeric or array_like
        Surface pressure :math:`P` of the measurement site.
    latitude : numeric or array_like, optional
        Latitude of the site in degrees.
    altitude : numeric or array_like, optional
        Altitude of the site in meters.
    avogadro_constant : numeric or array_like, optional
        *Avogadro*'s number (molecules :math:`mol^{-1}`).
    n_s : object
        Air refraction index :math:`n_s` computation method.
    F_air : object
        :math:`(6+3_p)/(6-7_p)`, the depolarisation term :math:`F(air)` or
        *King Factor* computation method.

    Returns
    -------
    SpectralDistribution
        *Rayleigh* optical depth spectral distribution.

    References
    ----------
    :cite:`Bodhaine1999a`, :cite:`Wikipedia2001c`

    Examples
    --------
    >>> from colour.utilities import numpy_print_options
    >>> with numpy_print_options(suppress=True):
    ...     sd_rayleigh_scattering()  # doctest: +ELLIPSIS
    SpectralDistribution([[ 360.        ,    0.5991013...],
                          [ 361.        ,    0.5921706...],
                          [ 362.        ,    0.5853410...],
                          [ 363.        ,    0.5786105...],
                          [ 364.        ,    0.5719774...],
                          [ 365.        ,    0.5654401...],
                          [ 366.        ,    0.5589968...],
                          [ 367.        ,    0.5526460...],
                          [ 368.        ,    0.5463860...],
                          [ 369.        ,    0.5402153...],
                          [ 370.        ,    0.5341322...],
                          [ 371.        ,    0.5281354...],
                          [ 372.        ,    0.5222234...],
                          [ 373.        ,    0.5163946...],
                          [ 374.        ,    0.5106476...],
                          [ 375.        ,    0.5049812...],
                          [ 376.        ,    0.4993939...],
                          [ 377.        ,    0.4938844...],
                          [ 378.        ,    0.4884513...],
                          [ 379.        ,    0.4830934...],
                          [ 380.        ,    0.4778095...],
                          [ 381.        ,    0.4725983...],
                          [ 382.        ,    0.4674585...],
                          [ 383.        ,    0.4623891...],
                          [ 384.        ,    0.4573889...],
                          [ 385.        ,    0.4524566...],
                          [ 386.        ,    0.4475912...],
                          [ 387.        ,    0.4427917...],
                          [ 388.        ,    0.4380568...],
                          [ 389.        ,    0.4333856...],
                          [ 390.        ,    0.4287771...],
                          [ 391.        ,    0.4242302...],
                          [ 392.        ,    0.4197439...],
                          [ 393.        ,    0.4153172...],
                          [ 394.        ,    0.4109493...],
                          [ 395.        ,    0.4066391...],
                          [ 396.        ,    0.4023857...],
                          [ 397.        ,    0.3981882...],
                          [ 398.        ,    0.3940458...],
                          [ 399.        ,    0.3899576...],
                          [ 400.        ,    0.3859227...],
                          [ 401.        ,    0.3819402...],
                          [ 402.        ,    0.3780094...],
                          [ 403.        ,    0.3741295...],
                          [ 404.        ,    0.3702996...],
                          [ 405.        ,    0.366519 ...],
                          [ 406.        ,    0.3627868...],
                          [ 407.        ,    0.3591025...],
                          [ 408.        ,    0.3554651...],
                          [ 409.        ,    0.3518740...],
                          [ 410.        ,    0.3483286...],
                          [ 411.        ,    0.344828 ...],
                          [ 412.        ,    0.3413716...],
                          [ 413.        ,    0.3379587...],
                          [ 414.        ,    0.3345887...],
                          [ 415.        ,    0.3312609...],
                          [ 416.        ,    0.3279747...],
                          [ 417.        ,    0.3247294...],
                          [ 418.        ,    0.3215245...],
                          [ 419.        ,    0.3183593...],
                          [ 420.        ,    0.3152332...],
                          [ 421.        ,    0.3121457...],
                          [ 422.        ,    0.3090962...],
                          [ 423.        ,    0.3060841...],
                          [ 424.        ,    0.3031088...],
                          [ 425.        ,    0.3001699...],
                          [ 426.        ,    0.2972668...],
                          [ 427.        ,    0.2943989...],
                          [ 428.        ,    0.2915657...],
                          [ 429.        ,    0.2887668...],
                          [ 430.        ,    0.2860017...],
                          [ 431.        ,    0.2832697...],
                          [ 432.        ,    0.2805706...],
                          [ 433.        ,    0.2779037...],
                          [ 434.        ,    0.2752687...],
                          [ 435.        ,    0.2726650...],
                          [ 436.        ,    0.2700922...],
                          [ 437.        ,    0.2675500...],
                          [ 438.        ,    0.2650377...],
                          [ 439.        ,    0.2625551...],
                          [ 440.        ,    0.2601016...],
                          [ 441.        ,    0.2576770...],
                          [ 442.        ,    0.2552807...],
                          [ 443.        ,    0.2529124...],
                          [ 444.        ,    0.2505716...],
                          [ 445.        ,    0.2482581...],
                          [ 446.        ,    0.2459713...],
                          [ 447.        ,    0.2437110...],
                          [ 448.        ,    0.2414768...],
                          [ 449.        ,    0.2392683...],
                          [ 450.        ,    0.2370851...],
                          [ 451.        ,    0.2349269...],
                          [ 452.        ,    0.2327933...],
                          [ 453.        ,    0.2306841...],
                          [ 454.        ,    0.2285989...],
                          [ 455.        ,    0.2265373...],
                          [ 456.        ,    0.2244990...],
                          [ 457.        ,    0.2224838...],
                          [ 458.        ,    0.2204912...],
                          [ 459.        ,    0.2185211...],
                          [ 460.        ,    0.2165730...],
                          [ 461.        ,    0.2146467...],
                          [ 462.        ,    0.2127419...],
                          [ 463.        ,    0.2108583...],
                          [ 464.        ,    0.2089957...],
                          [ 465.        ,    0.2071536...],
                          [ 466.        ,    0.2053320...],
                          [ 467.        ,    0.2035304...],
                          [ 468.        ,    0.2017487...],
                          [ 469.        ,    0.1999865...],
                          [ 470.        ,    0.1982436...],
                          [ 471.        ,    0.1965198...],
                          [ 472.        ,    0.1948148...],
                          [ 473.        ,    0.1931284...],
                          [ 474.        ,    0.1914602...],
                          [ 475.        ,    0.1898101...],
                          [ 476.        ,    0.1881779...],
                          [ 477.        ,    0.1865633...],
                          [ 478.        ,    0.1849660...],
                          [ 479.        ,    0.1833859...],
                          [ 480.        ,    0.1818227...],
                          [ 481.        ,    0.1802762...],
                          [ 482.        ,    0.1787463...],
                          [ 483.        ,    0.1772326...],
                          [ 484.        ,    0.1757349...],
                          [ 485.        ,    0.1742532...],
                          [ 486.        ,    0.1727871...],
                          [ 487.        ,    0.1713365...],
                          [ 488.        ,    0.1699011...],
                          [ 489.        ,    0.1684809...],
                          [ 490.        ,    0.1670755...],
                          [ 491.        ,    0.1656848...],
                          [ 492.        ,    0.1643086...],
                          [ 493.        ,    0.1629468...],
                          [ 494.        ,    0.1615991...],
                          [ 495.        ,    0.1602654...],
                          [ 496.        ,    0.1589455...],
                          [ 497.        ,    0.1576392...],
                          [ 498.        ,    0.1563464...],
                          [ 499.        ,    0.1550668...],
                          [ 500.        ,    0.1538004...],
                          [ 501.        ,    0.1525470...],
                          [ 502.        ,    0.1513063...],
                          [ 503.        ,    0.1500783...],
                          [ 504.        ,    0.1488628...],
                          [ 505.        ,    0.1476597...],
                          [ 506.        ,    0.1464687...],
                          [ 507.        ,    0.1452898...],
                          [ 508.        ,    0.1441228...],
                          [ 509.        ,    0.1429675...],
                          [ 510.        ,    0.1418238...],
                          [ 511.        ,    0.1406916...],
                          [ 512.        ,    0.1395707...],
                          [ 513.        ,    0.1384610...],
                          [ 514.        ,    0.1373624...],
                          [ 515.        ,    0.1362747...],
                          [ 516.        ,    0.1351978...],
                          [ 517.        ,    0.1341316...],
                          [ 518.        ,    0.1330759...],
                          [ 519.        ,    0.1320306...],
                          [ 520.        ,    0.1309956...],
                          [ 521.        ,    0.1299707...],
                          [ 522.        ,    0.1289559...],
                          [ 523.        ,    0.1279511...],
                          [ 524.        ,    0.1269560...],
                          [ 525.        ,    0.1259707...],
                          [ 526.        ,    0.1249949...],
                          [ 527.        ,    0.1240286...],
                          [ 528.        ,    0.1230717...],
                          [ 529.        ,    0.1221240...],
                          [ 530.        ,    0.1211855...],
                          [ 531.        ,    0.1202560...],
                          [ 532.        ,    0.1193354...],
                          [ 533.        ,    0.1184237...],
                          [ 534.        ,    0.1175207...],
                          [ 535.        ,    0.1166263...],
                          [ 536.        ,    0.1157404...],
                          [ 537.        ,    0.1148630...],
                          [ 538.        ,    0.1139939...],
                          [ 539.        ,    0.1131331...],
                          [ 540.        ,    0.1122804...],
                          [ 541.        ,    0.1114357...],
                          [ 542.        ,    0.1105990...],
                          [ 543.        ,    0.1097702...],
                          [ 544.        ,    0.1089492...],
                          [ 545.        ,    0.1081358...],
                          [ 546.        ,    0.1073301...],
                          [ 547.        ,    0.1065319...],
                          [ 548.        ,    0.1057411...],
                          [ 549.        ,    0.1049577...],
                          [ 550.        ,    0.1041815...],
                          [ 551.        ,    0.1034125...],
                          [ 552.        ,    0.1026507...],
                          [ 553.        ,    0.1018958...],
                          [ 554.        ,    0.1011480...],
                          [ 555.        ,    0.1004070...],
                          [ 556.        ,    0.0996728...],
                          [ 557.        ,    0.0989453...],
                          [ 558.        ,    0.0982245...],
                          [ 559.        ,    0.0975102...],
                          [ 560.        ,    0.0968025...],
                          [ 561.        ,    0.0961012...],
                          [ 562.        ,    0.0954062...],
                          [ 563.        ,    0.0947176...],
                          [ 564.        ,    0.0940352...],
                          [ 565.        ,    0.0933589...],
                          [ 566.        ,    0.0926887...],
                          [ 567.        ,    0.0920246...],
                          [ 568.        ,    0.0913664...],
                          [ 569.        ,    0.0907141...],
                          [ 570.        ,    0.0900677...],
                          [ 571.        ,    0.0894270...],
                          [ 572.        ,    0.0887920...],
                          [ 573.        ,    0.0881627...],
                          [ 574.        ,    0.0875389...],
                          [ 575.        ,    0.0869207...],
                          [ 576.        ,    0.0863079...],
                          [ 577.        ,    0.0857006...],
                          [ 578.        ,    0.0850986...],
                          [ 579.        ,    0.0845019...],
                          [ 580.        ,    0.0839104...],
                          [ 581.        ,    0.0833242...],
                          [ 582.        ,    0.0827430...],
                          [ 583.        ,    0.082167 ...],
                          [ 584.        ,    0.0815959...],
                          [ 585.        ,    0.0810298...],
                          [ 586.        ,    0.0804687...],
                          [ 587.        ,    0.0799124...],
                          [ 588.        ,    0.0793609...],
                          [ 589.        ,    0.0788142...],
                          [ 590.        ,    0.0782722...],
                          [ 591.        ,    0.0777349...],
                          [ 592.        ,    0.0772022...],
                          [ 593.        ,    0.0766740...],
                          [ 594.        ,    0.0761504...],
                          [ 595.        ,    0.0756313...],
                          [ 596.        ,    0.0751166...],
                          [ 597.        ,    0.0746063...],
                          [ 598.        ,    0.0741003...],
                          [ 599.        ,    0.0735986...],
                          [ 600.        ,    0.0731012...],
                          [ 601.        ,    0.072608 ...],
                          [ 602.        ,    0.0721189...],
                          [ 603.        ,    0.0716340...],
                          [ 604.        ,    0.0711531...],
                          [ 605.        ,    0.0706763...],
                          [ 606.        ,    0.0702035...],
                          [ 607.        ,    0.0697347...],
                          [ 608.        ,    0.0692697...],
                          [ 609.        ,    0.0688087...],
                          [ 610.        ,    0.0683515...],
                          [ 611.        ,    0.0678981...],
                          [ 612.        ,    0.0674485...],
                          [ 613.        ,    0.0670026...],
                          [ 614.        ,    0.0665603...],
                          [ 615.        ,    0.0661218...],
                          [ 616.        ,    0.0656868...],
                          [ 617.        ,    0.0652555...],
                          [ 618.        ,    0.0648277...],
                          [ 619.        ,    0.0644033...],
                          [ 620.        ,    0.0639825...],
                          [ 621.        ,    0.0635651...],
                          [ 622.        ,    0.0631512...],
                          [ 623.        ,    0.0627406...],
                          [ 624.        ,    0.0623333...],
                          [ 625.        ,    0.0619293...],
                          [ 626.        ,    0.0615287...],
                          [ 627.        ,    0.0611312...],
                          [ 628.        ,    0.0607370...],
                          [ 629.        ,    0.0603460...],
                          [ 630.        ,    0.0599581...],
                          [ 631.        ,    0.0595733...],
                          [ 632.        ,    0.0591917...],
                          [ 633.        ,    0.0588131...],
                          [ 634.        ,    0.0584375...],
                          [ 635.        ,    0.0580649...],
                          [ 636.        ,    0.0576953...],
                          [ 637.        ,    0.0573286...],
                          [ 638.        ,    0.0569649...],
                          [ 639.        ,    0.0566040...],
                          [ 640.        ,    0.0562460...],
                          [ 641.        ,    0.0558909...],
                          [ 642.        ,    0.0555385...],
                          [ 643.        ,    0.0551890...],
                          [ 644.        ,    0.0548421...],
                          [ 645.        ,    0.0544981...],
                          [ 646.        ,    0.0541567...],
                          [ 647.        ,    0.053818 ...],
                          [ 648.        ,    0.0534819...],
                          [ 649.        ,    0.0531485...],
                          [ 650.        ,    0.0528176...],
                          [ 651.        ,    0.0524894...],
                          [ 652.        ,    0.0521637...],
                          [ 653.        ,    0.0518405...],
                          [ 654.        ,    0.0515198...],
                          [ 655.        ,    0.0512017...],
                          [ 656.        ,    0.0508859...],
                          [ 657.        ,    0.0505726...],
                          [ 658.        ,    0.0502618...],
                          [ 659.        ,    0.0499533...],
                          [ 660.        ,    0.0496472...],
                          [ 661.        ,    0.0493434...],
                          [ 662.        ,    0.0490420...],
                          [ 663.        ,    0.0487428...],
                          [ 664.        ,    0.0484460...],
                          [ 665.        ,    0.0481514...],
                          [ 666.        ,    0.0478591...],
                          [ 667.        ,    0.0475689...],
                          [ 668.        ,    0.0472810...],
                          [ 669.        ,    0.0469953...],
                          [ 670.        ,    0.0467117...],
                          [ 671.        ,    0.0464302...],
                          [ 672.        ,    0.0461509...],
                          [ 673.        ,    0.0458737...],
                          [ 674.        ,    0.0455986...],
                          [ 675.        ,    0.0453255...],
                          [ 676.        ,    0.0450545...],
                          [ 677.        ,    0.0447855...],
                          [ 678.        ,    0.0445185...],
                          [ 679.        ,    0.0442535...],
                          [ 680.        ,    0.0439905...],
                          [ 681.        ,    0.0437294...],
                          [ 682.        ,    0.0434703...],
                          [ 683.        ,    0.0432131...],
                          [ 684.        ,    0.0429578...],
                          [ 685.        ,    0.0427044...],
                          [ 686.        ,    0.0424529...],
                          [ 687.        ,    0.0422032...],
                          [ 688.        ,    0.0419553...],
                          [ 689.        ,    0.0417093...],
                          [ 690.        ,    0.0414651...],
                          [ 691.        ,    0.0412226...],
                          [ 692.        ,    0.0409820...],
                          [ 693.        ,    0.0407431...],
                          [ 694.        ,    0.0405059...],
                          [ 695.        ,    0.0402705...],
                          [ 696.        ,    0.0400368...],
                          [ 697.        ,    0.0398047...],
                          [ 698.        ,    0.0395744...],
                          [ 699.        ,    0.0393457...],
                          [ 700.        ,    0.0391187...],
                          [ 701.        ,    0.0388933...],
                          [ 702.        ,    0.0386696...],
                          [ 703.        ,    0.0384474...],
                          [ 704.        ,    0.0382269...],
                          [ 705.        ,    0.0380079...],
                          [ 706.        ,    0.0377905...],
                          [ 707.        ,    0.0375747...],
                          [ 708.        ,    0.0373604...],
                          [ 709.        ,    0.0371476...],
                          [ 710.        ,    0.0369364...],
                          [ 711.        ,    0.0367266...],
                          [ 712.        ,    0.0365184...],
                          [ 713.        ,    0.0363116...],
                          [ 714.        ,    0.0361063...],
                          [ 715.        ,    0.0359024...],
                          [ 716.        ,    0.0357000...],
                          [ 717.        ,    0.0354990...],
                          [ 718.        ,    0.0352994...],
                          [ 719.        ,    0.0351012...],
                          [ 720.        ,    0.0349044...],
                          [ 721.        ,    0.0347090...],
                          [ 722.        ,    0.0345150...],
                          [ 723.        ,    0.0343223...],
                          [ 724.        ,    0.0341310...],
                          [ 725.        ,    0.0339410...],
                          [ 726.        ,    0.0337523...],
                          [ 727.        ,    0.033565 ...],
                          [ 728.        ,    0.0333789...],
                          [ 729.        ,    0.0331941...],
                          [ 730.        ,    0.0330106...],
                          [ 731.        ,    0.0328284...],
                          [ 732.        ,    0.0326474...],
                          [ 733.        ,    0.0324677...],
                          [ 734.        ,    0.0322893...],
                          [ 735.        ,    0.0321120...],
                          [ 736.        ,    0.0319360...],
                          [ 737.        ,    0.0317611...],
                          [ 738.        ,    0.0315875...],
                          [ 739.        ,    0.0314151...],
                          [ 740.        ,    0.0312438...],
                          [ 741.        ,    0.0310737...],
                          [ 742.        ,    0.0309048...],
                          [ 743.        ,    0.0307370...],
                          [ 744.        ,    0.0305703...],
                          [ 745.        ,    0.0304048...],
                          [ 746.        ,    0.0302404...],
                          [ 747.        ,    0.0300771...],
                          [ 748.        ,    0.0299149...],
                          [ 749.        ,    0.0297538...],
                          [ 750.        ,    0.0295938...],
                          [ 751.        ,    0.0294349...],
                          [ 752.        ,    0.0292771...],
                          [ 753.        ,    0.0291203...],
                          [ 754.        ,    0.0289645...],
                          [ 755.        ,    0.0288098...],
                          [ 756.        ,    0.0286561...],
                          [ 757.        ,    0.0285035...],
                          [ 758.        ,    0.0283518...],
                          [ 759.        ,    0.0282012...],
                          [ 760.        ,    0.0280516...],
                          [ 761.        ,    0.0279030...],
                          [ 762.        ,    0.0277553...],
                          [ 763.        ,    0.0276086...],
                          [ 764.        ,    0.027463 ...],
                          [ 765.        ,    0.0273182...],
                          [ 766.        ,    0.0271744...],
                          [ 767.        ,    0.0270316...],
                          [ 768.        ,    0.0268897...],
                          [ 769.        ,    0.0267487...],
                          [ 770.        ,    0.0266087...],
                          [ 771.        ,    0.0264696...],
                          [ 772.        ,    0.0263314...],
                          [ 773.        ,    0.0261941...],
                          [ 774.        ,    0.0260576...],
                          [ 775.        ,    0.0259221...],
                          [ 776.        ,    0.0257875...],
                          [ 777.        ,    0.0256537...],
                          [ 778.        ,    0.0255208...],
                          [ 779.        ,    0.0253888...],
                          [ 780.        ,    0.0252576...]],
                         interpolator=SpragueInterpolator,
                         interpolator_kwargs={},
                         extrapolator=Extrapolator,
                         extrapolator_kwargs={...})
    """

    wavelengths = shape.range()
    return SpectralDistribution(
        data=dict(
            zip(
                wavelengths,
                rayleigh_optical_depth(wavelengths * 10e-8, CO2_concentration,
                                       temperature, pressure, latitude,
                                       altitude, avogadro_constant, n_s,
                                       F_air))),
        name=('Rayleigh Scattering - {0} ppm, {1} K, {2} Pa, {3} Degrees, '
              '{4} m').format(CO2_concentration, temperature, pressure,
                              latitude, altitude))
