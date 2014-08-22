#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Rayleigh Optical Depth / Scattering in the Atmosphere
=====================================================

Implements rayleigh scattering / optical depth in the atmosphere computation.

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
           'molecular_density_bodhaine1999',
           'mean_molecular_weight',
           'gravity_list1968',
           'scattering_cross_section',
           'rayleigh_optical_depth',
           'rayleigh_scattering']

STANDARD_AIR_TEMPERATURE = 288.15
"""
*Standard air* temperature :math:`T[K]` in kelvin degrees (:math:`15\circ C`).

STANDARD_AIR_TEMPERATURE : float
"""

STANDARD_CO2_CONCENTRATION = 300
"""
*Standard air* :math:`CO2` concentration in parts per million (ppm).

STANDARD_CO2_CONCENTRATION : float
"""

AVERAGE_PRESSURE_MEAN_SEA_LEVEL = 101325
"""
*Standard air* average pressure :math:`Hg` at mean sea-level in pascal (Pa).

AVERAGE_PRESSURE_MEAN_SEA_LEVEL : float
"""

DEFAULT_LATITUDE = 0
"""
Default latitude in degrees (equator).

DEFAULT_LATITUDE : float
"""

DEFAULT_ALTITUDE = 0
"""
Default altitude in meters (sea level).

DEFAULT_ALTITUDE : float
"""


def air_refraction_index_penndorf1957(wavelength):
    """
    Returns the air refraction index from given wavelength :math:`\lambda` in
    micrometers (:math:`\mu m`) using *Penndorf (1957)* method.

    Parameters
    ----------
    wavelength : float
        Wavelength :math:`\lambda` in micrometers (:math:`\mu m`).

    Returns
    -------
    float
        Air refraction index.

    Examples
    --------
    >>> colour.phenomenon.rayleigh.air_refraction_index_penndorf1957(555)
    1.0002725990541006
    """

    wl = wavelength
    n = 6432.8 + 2949810 / (146 - wl ** (-2)) + 25540 / (41 - wl ** (-2))
    n = n / 1.0e8 + 1
    return n


def air_refraction_index_edlen1966(wavelength):
    """
    Returns the air refraction index from given wavelength :math:`\lambda` in
    micrometers (:math:`\mu m`) using *Edlen (1966)* method.

    Parameters
    ----------
    wavelength : float
        Wavelength :math:`\lambda` in micrometers (:math:`\mu m`).

    Returns
    -------
    float
        Air refraction index.

    Examples
    --------
    >>> colour.phenomenon.rayleigh.air_refraction_index_edlen1966(555)
    1.000272612875066
    """

    wl = wavelength
    n = 8342.13 + 2406030 / (130 - wl ** (-2)) + 15997 / (38.9 - wl ** (-2))
    n = n / 1.0e8 + 1
    return n


def air_refraction_index_peck1972(wavelength):
    """
    Returns the air refraction index from given wavelength :math:`\lambda` in
    micrometers (:math:`\mu m`) using *Peck and Reeder (1972)* method.

    Parameters
    ----------
    wavelength : float
        Wavelength :math:`\lambda` in micrometers (:math:`\mu m`).

    Returns
    -------
    float
        Air refraction index.

    Examples
    --------
    >>> colour.phenomenon.rayleigh.air_refraction_index_peck1972(555)
    1.000272607869001
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
    Returns the air refraction index from given wavelength :math:`\lambda` in
    micrometers (:math:`\mu m`) using
    *Bodhaine, Wood, Dutton and Slusser (1999)* method.

    Parameters
    ----------
    wavelength : float
        Wavelength :math:`\lambda` in micrometers (:math:`\mu m`).

    Returns
    -------
    float
        Air refraction index.

    Examples
    --------
    >>> colour.phenomenon.rayleigh.air_refraction_index_bodhaine1999(555)
    1.000272607869001
    """

    wl = wavelength
    CCO2 = CO2_concentration
    n = ((1 + 0.54 * ((CCO2 * 1e-6) - 300e-6)) *
         (air_refraction_index_peck1972(wl) - 1) + 1)
    return n


def N2_depolarisation(wavelength):
    """Depolarisation of N2.

    On Rayleigh Optical Depth Calculations"""

    wl = wavelength
    N2 = 1.034 + 3.17 * 1.0e-4 * (1 / wl ** 2)

    return N2


def O2_depolarisation(wavelength):
    """Depolarisation of O2.

    On Rayleigh Optical Depth Calculations"""

    wl = wavelength
    O2 = (1.096 + 1.385 * 1.0e-3 * (1 / wl ** 2) +
           1.448 * 1.0e-4 * (1 / wl ** 4))

    return O2


def F_air_penndorf1957(wavelength):
    """On Rayleigh Optical Depth Calculations"""
    return 1.0608


def F_air_young1981(wavelength):
    """On Rayleigh Optical Depth Calculations"""
    return 1.0480


def F_air_bates1984(wavelength):
    """On Rayleigh Optical Depth Calculations"""

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
    """On Rayleigh Optical Depth Calculations"""

    wl = wavelength
    O2 = O2_depolarisation(wl)
    N2 = N2_depolarisation(wl)
    CO2_c = CO2_concentration

    F_air = ((78.084 * N2 + 20.946 * O2 + 0.934 * 1 + CO2_c * 1.15) /
             (78.084 + 20.946 + 0.934 + CO2_c))

    return F_air


def molecular_density_bodhaine1999(temperature=STANDARD_AIR_TEMPERATURE,
                                   avogadro_constant=AVOGADRO_CONSTANT):
    """
    temperature in K
    On Rayleigh Optical Depth Calculations"""

    T = temperature
    N_s = (avogadro_constant / 22.4141) * (273.15 / T) * (1 / 1000)
    return N_s


def mean_molecular_weight(CO2_concentration=STANDARD_CO2_CONCENTRATION):
    CO2_c = CO2_concentration * 1.0e-6

    m_a = 15.0556 * CO2_c + 28.9595
    return m_a


def gravity_list1968(latitude=DEFAULT_LATITUDE, altitude=DEFAULT_ALTITUDE):
    """
    gravity in cm s-2
    """

    cos2phi = math.cos(2 * math.radians(latitude))

    # Sea level acceleration of gravity.
    g0 = 980.6160 * (1 - 0.0026373 * cos2phi + 0.0000059 * cos2phi ** 2)

    g = (g0 - (3.085462e-4 + 2.27e-7 * cos2phi) * altitude +
         (7.254e-11 + 1.0e-13 * cos2phi) * altitude ** 2 -
         (1.517e-17 + 6e-20 * cos2phi) * altitude ** 3)
    return g


def scattering_cross_section(wavelength,
                             temperature=STANDARD_AIR_TEMPERATURE,
                             CO2_concentration=STANDARD_CO2_CONCENTRATION,
                             avogadro_constant=AVOGADRO_CONSTANT,
                             n_s=air_refraction_index_bodhaine1999,
                             N_s=molecular_density_bodhaine1999,
                             F_air=F_air_bodhaine1999):
    """*Van de Hulst (1957)*, wavelength in centimeters.
    """

    wl = wavelength
    wl_micrometers = wl * 10e3
    n_s = n_s(wl_micrometers)
    N_s = N_s(temperature, avogadro_constant)
    F_air = F_air(wl_micrometers, CO2_concentration)

    sigma = (24 * math.pi ** 3 * (n_s ** 2 - 1) ** 2 /
             (wl ** 4 * N_s ** 2 * (n_s ** 2 + 2) ** 2))
    sigma *= F_air

    return sigma


def rayleigh_optical_depth(wavelength,
                           P=AVERAGE_PRESSURE_MEAN_SEA_LEVEL,
                           latitude=DEFAULT_LATITUDE,
                           altitude=DEFAULT_ALTITUDE,
                           temperature=STANDARD_AIR_TEMPERATURE,
                           CO2_concentration=STANDARD_CO2_CONCENTRATION,
                           avogadro_constant=AVOGADRO_CONSTANT,
                           n_s=air_refraction_index_bodhaine1999,
                           N_s=molecular_density_bodhaine1999,
                           F_air=F_air_bodhaine1999):
    """wavelength in centimeters
    On Rayleigh Optical Depth Calculations"""
    CO2_c = CO2_concentration
    sigma = scattering_cross_section(wavelength,
                                     temperature,
                                     CO2_c,
                                     avogadro_constant,
                                     n_s,
                                     N_s,
                                     F_air)

    # Conversion from pascal to dyne/cm2.
    P = P * 10
    m_a = mean_molecular_weight(CO2_c)
    g = gravity_list1968(latitude, altitude)

    T_R = sigma * (P * avogadro_constant) / (m_a * g)

    return T_R


rayleigh_scattering = rayleigh_optical_depth

print(rayleigh_scattering(5.32 * 10e-6))
print(scattering_cross_section(532 * 10e-8))
# 0.11933547195409362
# 5.540072155775108e-27