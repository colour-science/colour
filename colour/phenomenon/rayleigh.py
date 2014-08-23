#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Rayleigh Optical Depth / Scattering in the Atmosphere
=====================================================

Implements rayleigh scattering / optical depth in the atmosphere computation:

-   :func:`scattering_cross_section`
-   :func:`rayleigh_optical_depth`
-   :func:`rayleigh_scattering`

References
----------
.. [1]  `On Rayleigh Optical Depth Calculations
        <http://journals.ametsoc.org/doi/pdf/10.1175/1520-0426(1999)016%3C1854%3AORODC%3E2.0.CO%3B2>`_
.. [2]  http://en.wikipedia.org/wiki/Rayleigh_scattering
"""

from __future__ import division, unicode_literals

import math

from colour.constants import AVOGADRO_CONSTANT

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['STANDARD_AIR_TEMPERATURE',
           'STANDARD_CO2_CONCENTRATION',
           'AVERAGE_PRESSURE_MEAN_SEA_LEVEL',
           'DEFAULT_LATITUDE',
           'DEFAULT_ALTITUDE',
           'air_refraction_index_penndorf1957',
           'air_refraction_index_edlen1966',
           'air_refraction_index_peck1972',
           'air_refraction_index_bodhaine1999',
           'N2_depolarisation',
           'O2_depolarisation',
           'F_air_penndorf1957',
           'F_air_young1981',
           'F_air_bates1984',
           'F_air_bodhaine1999',
           'molecular_density',
           'mean_molecular_weights',
           'gravity_list1968',
           'scattering_cross_section',
           'rayleigh_optical_depth',
           'rayleigh_scattering']

STANDARD_AIR_TEMPERATURE = 288.15
"""
*Standard air* temperature :math:`T[K]` in kelvin degrees (:math:`15\circ C`).

STANDARD_AIR_TEMPERATURE : numeric
"""

STANDARD_CO2_CONCENTRATION = 300
"""
*Standard air* :math:`CO_2` concentration in parts per million (ppm).

STANDARD_CO2_CONCENTRATION : numeric
"""

AVERAGE_PRESSURE_MEAN_SEA_LEVEL = 101325
"""
*Standard air* average pressure :math:`Hg` at mean sea-level in pascal (Pa).

AVERAGE_PRESSURE_MEAN_SEA_LEVEL : numeric
"""

DEFAULT_LATITUDE = 0
"""
Default latitude in degrees (equator).

DEFAULT_LATITUDE : numeric
"""

DEFAULT_ALTITUDE = 0
"""
Default altitude in meters (sea level).

DEFAULT_ALTITUDE : numeric
"""


def air_refraction_index_penndorf1957(wavelength, *args):
    """
    Returns the air refraction index :math:`n_s` from given wavelength
    :math:`\lambda` in  micrometers (:math:`\mu m`) using *Penndorf (1957)*
    method.

    Parameters
    ----------
    wavelength : numeric
        Wavelength :math:`\lambda` in micrometers (:math:`\mu m`).
    \*args : \*
        Arguments.

    Returns
    -------
    numeric
        Air refraction index :math:`n_s`.

    See Also
    --------
    air_refraction_index_edlen1966, air_refraction_index_peck1972,
    air_refraction_index_bodhaine1999

    Examples
    --------
    >>> colour.phenomenon.rayleigh.air_refraction_index_penndorf1957(0.555)
    1.000277729533864
    """

    wl = wavelength
    n = 6432.8 + 2949810 / (146 - wl ** (-2)) + 25540 / (41 - wl ** (-2))
    n = n / 1.0e8 + 1
    return n


def air_refraction_index_edlen1966(wavelength, *args):
    """
    Returns the air refraction index :math:`n_s` from given wavelength
    :math:`\lambda` in micrometers (:math:`\mu m`) using *Edlen (1966)* method.

    Parameters
    ----------
    wavelength : numeric
        Wavelength :math:`\lambda` in micrometers (:math:`\mu m`).
    \*args : \*
        Arguments.

    Returns
    -------
    numeric
        Air refraction index :math:`n_s`.

    See Also
    --------
    air_refraction_index_penndorf1957, air_refraction_index_peck1972,
    air_refraction_index_bodhaine1999

    Examples
    --------
    >>> colour.phenomenon.rayleigh.air_refraction_index_edlen1966(0.555)
    1.000277727690364
    """

    wl = wavelength
    n = 8342.13 + 2406030 / (130 - wl ** (-2)) + 15997 / (38.9 - wl ** (-2))
    n = n / 1.0e8 + 1
    return n


def air_refraction_index_peck1972(wavelength, *args):
    """
    Returns the air refraction index :math:`n_s` from given wavelength
    :math:`\lambda` in micrometers (:math:`\mu m`) using
    *Peck and Reeder (1972)* method.

    Parameters
    ----------
    wavelength : numeric
        Wavelength :math:`\lambda` in micrometers (:math:`\mu m`).
    \*args : \*
        Arguments.

    Returns
    -------
    numeric
        Air refraction index :math:`n_s`.

    See Also
    --------
    air_refraction_index_penndorf1957, air_refraction_index_edlen1966,
    air_refraction_index_bodhaine1999

    Examples
    --------
    >>> colour.phenomenon.rayleigh.air_refraction_index_peck1972(0.555)
    1.0002777265414837
    """

    wl = wavelength
    n = (8060.51 + 2480990 / (132.274 - wl ** (-2)) + 17455.7 /
         (39.32957 - wl ** (-2)))
    n = n / 1.0e8 + 1
    return n


def air_refraction_index_bodhaine1999(
        wavelength,
        CO2_concentration=STANDARD_CO2_CONCENTRATION):
    """
    Returns the air refraction index :math:`n_s` from given wavelength
    :math:`\lambda` in micrometers (:math:`\mu m`) using
    *Bodhaine, Wood, Dutton and Slusser (1999)* method.

    Parameters
    ----------
    wavelength : numeric
        Wavelength :math:`\lambda` in micrometers (:math:`\mu m`).
    CO2_concentration : numeric
        :math:`CO_2` concentration in parts per million (ppm).

    Returns
    -------
    numeric
        Air refraction index :math:`n_s`.

    See Also
    --------
    air_refraction_index_penndorf1957, air_refraction_index_edlen1966,
    air_refraction_index_peck1972

    Examples
    --------
    >>> colour.phenomenon.rayleigh.air_refraction_index_bodhaine1999(0.555)
    1.0002777265414837
    """

    wl = wavelength
    CCO2 = CO2_concentration
    n = ((1 + 0.54 * ((CCO2 * 1e-6) - 300e-6)) *
         (air_refraction_index_peck1972(wl) - 1) + 1)
    return n


def N2_depolarisation(wavelength):
    """
    Returns the depolarisation of nitrogen :math:`N_2` as function of
    wavelength :math:`\lambda` in micrometers (:math:`\mu m`).

    Parameters
    ----------
    wavelength : numeric
        Wavelength :math:`\lambda` in micrometers (:math:`\mu m`).

    Returns
    -------
    numeric
        Nitrogen :math:`N_2` depolarisation.

    Examples
    --------
    >>> colour.phenomenon.rayleigh.N2_depolarisation(0.555)
    1.0350291372453535
    """

    wl = wavelength
    N2 = 1.034 + 3.17 * 1.0e-4 * (1 / wl ** 2)

    return N2


def O2_depolarisation(wavelength):
    """
    Returns the depolarisation of oxygen :math:`O_2` as function of
    wavelength :math:`\lambda` in micrometers (:math:`\mu m`).

    Parameters
    ----------
    wavelength : numeric
        Wavelength :math:`\lambda` in micrometers (:math:`\mu m`).

    Returns
    -------
    numeric
        Oxygen :math:`O_2` depolarisation.

    Examples
    --------
    >>> colour.phenomenon.rayleigh.O2_depolarisation(0.555)
    1.1020225362010714
    """

    wl = wavelength
    O2 = (1.096 + 1.385 * 1.0e-3 * (1 / wl ** 2) +
          1.448 * 1.0e-4 * (1 / wl ** 4))

    return O2


def F_air_penndorf1957(*args):
    """
    Returns :math:`(6+3_p)/(6-7_p)`, the depolarisation term :math:`F(air)` or
    *King Factor* using *Penndorf (1957)* method.

    Parameters
    ----------
    \*args : \*
        Arguments.

    Returns
    -------
    numeric
        Air depolarisation.

    See Also
    --------
    F_air_young1981, F_air_bates1984, F_air_bodhaine1999

    Notes
    -----
    -   The argument *wavelength* is only provided for consistency with the
        other air depolarisation methods but is actually not used as this
        definition is essentially a constant in its current implementation.

    Examples
    --------
    >>> colour.phenomenon.rayleigh.F_air_penndorf1957(0.555)
    1.0608
    """

    return 1.0608


def F_air_young1981(*args):
    """
    Returns :math:`(6+3_p)/(6-7_p)`, the depolarisation term :math:`F(air)` or
    *King Factor* using *Young (1981)* method.

    Parameters
    ----------
    \*args : \*
        Arguments.

    Returns
    -------
    numeric
        Air depolarisation.

    See Also
    --------
    F_air_penndorf1957, F_air_bates1984, F_air_bodhaine1999

    Notes
    -----
    -   The argument *wavelength* is only provided for consistency with the
        other air depolarisation methods but is actually not used as this
        definition is essentially a constant in its current implementation.

    Examples
    --------
    >>> colour.phenomenon.rayleigh.F_air_young1981(0.555)
    1.0480
    """

    return 1.0480


def F_air_bates1984(wavelength):
    """
    Returns :math:`(6+3_p)/(6-7_p)`, the depolarisation term :math:`F(air)` or
    *King Factor* as function of wavelength :math:`\lambda` in micrometers
    (:math:`\mu m`) using *Bates (1984)* method.

    Parameters
    ----------
    wavelength : numeric
        Wavelength :math:`\lambda` in micrometers (:math:`\mu m`).

    Returns
    -------
    numeric
        Air depolarisation.

    See Also
    --------
    F_air_penndorf1957, F_air_young1981, F_air_bodhaine1999

    Examples
    --------
    >>> colour.phenomenon.rayleigh.F_air_bates1984(0.555)
    1.048153579718658
    """

    wl = wavelength
    O2 = O2_depolarisation(wl)
    N2 = N2_depolarisation(wl)
    Ar = 1.00
    CO2 = 1.15

    F_air = ((78.084 * N2 + 20.946 * O2 + CO2 + Ar) /
             (78.084 + 20.946 + Ar + CO2))

    return F_air


def F_air_bodhaine1999(wavelength,
                       CO2_concentration=STANDARD_CO2_CONCENTRATION):
    """
    Returns :math:`(6+3_p)/(6-7_p)`, the depolarisation term :math:`F(air)` or
    *King Factor* as function of wavelength :math:`\lambda` in micrometers
    (:math:`\mu m`) and :math:`CO_2` concentration in parts per million (ppm)
    using *Bodhaine, Wood, Dutton and Slusser (1999)* method.

    Parameters
    ----------
    wavelength : numeric
        Wavelength :math:`\lambda` in micrometers (:math:`\mu m`).
    CO2_concentration : numeric, optional
        :math:`CO_2` concentration in parts per million (ppm).

    Returns
    -------
    numeric
        Air depolarisation.

    See Also
    --------
    F_air_penndorf1957, F_air_young1981, F_air_bates1984

    Examples
    --------
    >>> colour.phenomenon.rayleigh.F_air_bodhaine1999(0.555)
    1.1246916702401561
    """

    wl = wavelength
    O2 = O2_depolarisation(wl)
    N2 = N2_depolarisation(wl)
    CO2_c = CO2_concentration

    F_air = ((78.084 * N2 + 20.946 * O2 + 0.934 * 1 + CO2_c * 1.15) /
             (78.084 + 20.946 + 0.934 + CO2_c))

    return F_air


def molecular_density(temperature=STANDARD_AIR_TEMPERATURE,
                      avogadro_constant=AVOGADRO_CONSTANT):
    """
    Returns the molecular density :math:`N_s` (molecules :math:`cm^{-3}`)
    as function of air temperature :math:`T[K]` in kelvin degrees.

    Parameters
    ----------
    temperature : numeric, optional
        Air temperature :math:`T[K]` in kelvin degrees.
    avogadro_constant : numeric, optional
        *Avogadro*'s number (molecules :math:`mol^{-1}`).

    Returns
    -------
    numeric
        Molecular density :math:`N_s` (molecules :math:`cm^{-3}`).

    Notes
    -----
    -   The *Avogadro*'s number used in this implementation is the one given by
        by the Committee on Data for Science and Technology (CODATA):
        :math:`6.02214179x10^{23}`, which is different from the reference [1]_
        value :math:`6.0221367x10^{23}`.

    Examples
    --------
    >>> colour.phenomenon.rayleigh.molecular_density(288.15)
    2.546902105274093e+19
    >>> colour.phenomenon.rayleigh.molecular_density(288.15, 6.0221367e23)
    2.5468999525961638e+19
    """

    T = temperature
    N_s = (avogadro_constant / 22.4141) * (273.15 / T) * (1 / 1000)
    return N_s


def mean_molecular_weights(CO2_concentration=STANDARD_CO2_CONCENTRATION):
    """
    Returns the mean molecular weights :math:`m_a` for dry air as function of
    :math:`CO_2` concentration in parts per million (ppm).

    Parameters
    ----------
    CO2_concentration : numeric, optional
        :math:`CO_2` concentration in parts per million (ppm).

    Returns
    -------
    numeric
        Mean molecular weights :math:`m_a` for dry air.

    Examples
    --------
    >>> colour.phenomenon.rayleigh.mean_molecular_weights()
    28.964016679999997
    """

    CO2_c = CO2_concentration * 1.0e-6

    m_a = 15.0556 * CO2_c + 28.9595
    return m_a


def gravity_list1968(latitude=DEFAULT_LATITUDE, altitude=DEFAULT_ALTITUDE):
    """
    Returns the gravity :math:`g` in :math:`cm/s_2` (gal) representative of the
    mass-weighted column of air molecules above the site of given latitude and
    altitude using *List (1968)* method.

    Parameters
    ----------
    latitude : numeric, optional
        Latitude of the site in degrees.
    altitude : numeric, optional
        Altitude of the site in meters.

    Returns
    -------
    numeric
        Gravity :math:`g` in :math:`cm/s_2` (gal).

    Examples
    --------
    >>> colour.phenomenon.rayleigh.gravity_list1968()
    978.0356070576
    >>> colour.phenomenon.rayleigh.gravity_list1968(0, 1500)
    977.5726106461989

    Gravity :math:`g` for Paris:

    >>> colour.phenomenon.rayleigh.gravity_list1968(48.8567, 35)
    980.9524178426182
    """

    cos2phi = math.cos(2 * math.radians(latitude))

    # Sea level acceleration of gravity.
    g0 = 980.6160 * (1 - 0.0026373 * cos2phi + 0.0000059 * cos2phi ** 2)

    g = (g0 - (3.085462e-4 + 2.27e-7 * cos2phi) * altitude +
         (7.254e-11 + 1.0e-13 * cos2phi) * altitude ** 2 -
         (1.517e-17 + 6e-20 * cos2phi) * altitude ** 3)
    return g


def scattering_cross_section(wavelength,
                             CO2_concentration=STANDARD_CO2_CONCENTRATION,
                             temperature=STANDARD_AIR_TEMPERATURE,
                             avogadro_constant=AVOGADRO_CONSTANT,
                             n_s=air_refraction_index_bodhaine1999,
                             F_air=F_air_bodhaine1999):
    """
    Returns the scattering cross section per molecule :math:`\sigma` of dry air
    as function of wavelength :math:`\lambda` in centimeters (cm) using given
    :math:`CO_2` concentration in parts per million (ppm) and temperature
    :math:`T[K]` in kelvin degrees following *Van de Hulst (1957)* method.

    Parameters
    ----------
    wavelength : numeric
        Wavelength :math:`\lambda` in centimeters (cm).
    CO2_concentration : numeric, optional
        :math:`CO_2` concentration in parts per million (ppm).
    temperature : numeric, optional
        Air temperature :math:`T[K]` in kelvin degrees.
    avogadro_constant : numeric, optional
        *Avogadro*'s number (molecules :math:`mol^{-1}`).
    n_s : object
        Air refraction index :math:`n_s` computation method.
    F_air : object
        :math:`(6+3_p)/(6-7_p)`, the depolarisation term :math:`F(air)` or
        *King Factor* computation method.

    Returns
    -------
    numeric
        Scattering cross section per molecule :math:`\sigma` of dry air.

    Warning
    -------
    Unlike most objects of :mod:`colour.phenomenon.rayleigh` module,
    :func:`colour.phenomenon.rayleigh.scattering_cross_section` expects
    wavelength :math:`\lambda` to be expressed in centimeters (cm).

    Examples
    --------
    >>> colour.phenomenon.rayleigh.scattering_cross_section(555 * 10e-8)
    4.661330902337604e-27
    """

    wl = wavelength
    wl_micrometers = wl * 10e3
    n_s = n_s(wl_micrometers)
    N_s = molecular_density(temperature, avogadro_constant)
    F_air = F_air(wl_micrometers, CO2_concentration)

    sigma = (24 * math.pi ** 3 * (n_s ** 2 - 1) ** 2 /
             (wl ** 4 * N_s ** 2 * (n_s ** 2 + 2) ** 2))
    sigma *= F_air

    return sigma


def rayleigh_optical_depth(wavelength,
                           P=AVERAGE_PRESSURE_MEAN_SEA_LEVEL,
                           latitude=DEFAULT_LATITUDE,
                           altitude=DEFAULT_ALTITUDE,
                           CO2_concentration=STANDARD_CO2_CONCENTRATION,
                           temperature=STANDARD_AIR_TEMPERATURE,
                           avogadro_constant=AVOGADRO_CONSTANT,
                           n_s=air_refraction_index_bodhaine1999,
                           F_air=F_air_bodhaine1999):
    """
    Returns the rayleigh optical depth :math:`T_r(\lambda)` as function of
    wavelength :math:`\lambda` in centimeters (cm).

    Parameters
    ----------
    wavelength : numeric
        Wavelength :math:`\lambda` in centimeters (cm).
    P : numeric
        Surface pressure :math:`P` of the measurement site.
    latitude : numeric, optional
        Latitude of the site in degrees.
    altitude : numeric, optional
        Altitude of the site in meters.
    CO2_concentration : numeric, optional
        :math:`CO_2` concentration in parts per million (ppm).
    temperature : numeric, optional
        Air temperature :math:`T[K]` in kelvin degrees.
    avogadro_constant : numeric, optional
        *Avogadro*'s number (molecules :math:`mol^{-1}`).
    n_s : object
        Air refraction index :math:`n_s` computation method.
    F_air : object
        :math:`(6+3_p)/(6-7_p)`, the depolarisation term :math:`F(air)` or
        *King Factor* computation method.

    Returns
    -------
    numeric
        Rayleigh optical depth :math:`T_r(\lambda)`

    Warning
    -------
    Unlike most objects of :mod:`colour.phenomenon.rayleigh` module,
    :func:`colour.phenomenon.rayleigh.rayleigh_optical_depth` expects
    wavelength :math:`\lambda` to be expressed in centimeters (cm).

    Examples
    --------
    >>> colour.rayleigh_optical_depth(555 * 10e-8)
    0.10040701772896546
    """

    CO2_c = CO2_concentration
    sigma = scattering_cross_section(wavelength,
                                     CO2_c,
                                     temperature,
                                     avogadro_constant,
                                     n_s,
                                     F_air)

    # Conversion from pascal to dyne/cm2.
    P = P * 10
    m_a = mean_molecular_weights(CO2_c)
    g = gravity_list1968(latitude, altitude)

    T_R = sigma * (P * avogadro_constant) / (m_a * g)

    return T_R


rayleigh_scattering = rayleigh_optical_depth