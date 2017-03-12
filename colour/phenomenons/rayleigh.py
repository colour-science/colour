#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Rayleigh Optical Depth - Scattering in the Atmosphere
=====================================================

Implements *Rayleigh* scattering / optical depth in the atmosphere computation:

-   :func:`scattering_cross_section`
-   :func:`rayleigh_optical_depth`
-   :func:`rayleigh_scattering`

See Also
--------
`Rayleigh Optical Depth - Scattering in the Atmosphere Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/phenomenons/rayleigh.ipynb>`_

References
----------
.. [1]  Bodhaine, B. A., Wood, N. B., Dutton, E. G., & Slusser, J. R. (1999).
        On Rayleigh optical depth calculations. Journal of Atmospheric …,
        16(11 PART 2), 1854–1861.
        doi:10.1175/1520-0426(1999)016%3C1854:ORODC%3E2.0.CO;2
.. [2]  Wikipedia. (n.d.). Rayleigh scattering. Retrieved September 23, 2014,
        from http://en.wikipedia.org/wiki/Rayleigh_scattering
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import (
    DEFAULT_SPECTRAL_SHAPE,
    SpectralPowerDistribution)
from colour.constants import AVOGADRO_CONSTANT
from colour.utilities import filter_kwargs

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['STANDARD_AIR_TEMPERATURE',
           'STANDARD_CO2_CONCENTRATION',
           'AVERAGE_PRESSURE_MEAN_SEA_LEVEL',
           'DEFAULT_LATITUDE',
           'DEFAULT_ALTITUDE',
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


def air_refraction_index_Penndorf1957(wavelength):
    """
    Returns the air refraction index :math:`n_s` from given wavelength
    :math:`\lambda` in  micrometers (:math:`\mu m`) using *Penndorf (1957)*
    method.

    Parameters
    ----------
    wavelength : numeric or array_like
        Wavelength :math:`\lambda` in micrometers (:math:`\mu m`).

    Returns
    -------
    numeric or ndarray
        Air refraction index :math:`n_s`.

    See Also
    --------
    air_refraction_index_Edlen1966, air_refraction_index_Peck1972,
    air_refraction_index_Bodhaine1999

    Examples
    --------
    >>> air_refraction_index_Penndorf1957(0.555)  # doctest: +ELLIPSIS
    1.0002777...
    """

    wl = np.asarray(wavelength)

    n = 6432.8 + 2949810 / (146 - wl ** (-2)) + 25540 / (41 - wl ** (-2))
    n /= 1.0e8
    n += + 1

    return n


def air_refraction_index_Edlen1966(wavelength):
    """
    Returns the air refraction index :math:`n_s` from given wavelength
    :math:`\lambda` in micrometers (:math:`\mu m`) using *Edlen (1966)* method.

    Parameters
    ----------
    wavelength : numeric or array_like
        Wavelength :math:`\lambda` in micrometers (:math:`\mu m`).

    Returns
    -------
    numeric or ndarray
        Air refraction index :math:`n_s`.

    See Also
    --------
    air_refraction_index_Penndorf1957, air_refraction_index_Peck1972,
    air_refraction_index_Bodhaine1999

    Examples
    --------
    >>> air_refraction_index_Edlen1966(0.555)  # doctest: +ELLIPSIS
    1.0002777...
    """

    wl = np.asarray(wavelength)

    n = 8342.13 + 2406030 / (130 - wl ** (-2)) + 15997 / (38.9 - wl ** (-2))
    n /= 1.0e8
    n += + 1

    return n


def air_refraction_index_Peck1972(wavelength):
    """
    Returns the air refraction index :math:`n_s` from given wavelength
    :math:`\lambda` in micrometers (:math:`\mu m`) using
    *Peck and Reeder (1972)* method.

    Parameters
    ----------
    wavelength : numeric or array_like
        Wavelength :math:`\lambda` in micrometers (:math:`\mu m`).

    Returns
    -------
    numeric or ndarray
        Air refraction index :math:`n_s`.

    See Also
    --------
    air_refraction_index_Penndorf1957, air_refraction_index_Edlen1966,
    air_refraction_index_Bodhaine1999

    Examples
    --------
    >>> air_refraction_index_Peck1972(0.555)  # doctest: +ELLIPSIS
    1.0002777...
    """

    wl = np.asarray(wavelength)

    n = (8060.51 + 2480990 / (132.274 - wl ** (-2)) + 17455.7 /
         (39.32957 - wl ** (-2)))
    n /= 1.0e8
    n += + 1

    return n


def air_refraction_index_Bodhaine1999(
        wavelength,
        CO2_concentration=STANDARD_CO2_CONCENTRATION):
    """
    Returns the air refraction index :math:`n_s` from given wavelength
    :math:`\lambda` in micrometers (:math:`\mu m`) using
    *Bodhaine, Wood, Dutton and Slusser (1999)* method.

    Parameters
    ----------
    wavelength : numeric or array_like
        Wavelength :math:`\lambda` in micrometers (:math:`\mu m`).
    CO2_concentration : numeric or array_like
        :math:`CO_2` concentration in parts per million (ppm).

    Returns
    -------
    numeric or ndarray
        Air refraction index :math:`n_s`.

    See Also
    --------
    air_refraction_index_Penndorf1957, air_refraction_index_Edlen1966,
    air_refraction_index_Peck1972

    Examples
    --------
    >>> air_refraction_index_Bodhaine1999(0.555)  # doctest: +ELLIPSIS
    1.0002777...
    """

    wl = np.asarray(wavelength)
    CO2_c = np.asarray(CO2_concentration)

    n = ((1 + 0.54 * ((CO2_c * 1e-6) - 300e-6)) *
         (air_refraction_index_Peck1972(wl) - 1) + 1)

    return n


def N2_depolarisation(wavelength):
    """
    Returns the depolarisation of nitrogen :math:`N_2` as function of
    wavelength :math:`\lambda` in micrometers (:math:`\mu m`).

    Parameters
    ----------
    wavelength : numeric or array_like
        Wavelength :math:`\lambda` in micrometers (:math:`\mu m`).

    Returns
    -------
    numeric or ndarray
        Nitrogen :math:`N_2` depolarisation.

    Examples
    --------
    >>> N2_depolarisation(0.555)  # doctest: +ELLIPSIS
    1.0350291...
    """

    wl = np.asarray(wavelength)

    N2 = 1.034 + 3.17 * 1.0e-4 * (1 / wl ** 2)

    return N2


def O2_depolarisation(wavelength):
    """
    Returns the depolarisation of oxygen :math:`O_2` as function of
    wavelength :math:`\lambda` in micrometers (:math:`\mu m`).

    Parameters
    ----------
    wavelength : numeric or array_like
        Wavelength :math:`\lambda` in micrometers (:math:`\mu m`).

    Returns
    -------
    numeric or ndarray
        Oxygen :math:`O_2` depolarisation.

    Examples
    --------
    >>> O2_depolarisation(0.555)  # doctest: +ELLIPSIS
    1.1020225...
    """

    wl = np.asarray(wavelength)

    O2 = (1.096 + 1.385 * 1.0e-3 * (1 / wl ** 2) + 1.448 * 1.0e-4 *
          (1 / wl ** 4))

    return O2


def F_air_Penndorf1957(wavelength):
    """
    Returns :math:`(6+3_p)/(6-7_p)`, the depolarisation term :math:`F(air)` or
    *King Factor* using *Penndorf (1957)* method.

    Parameters
    ----------
    wavelength : numeric or array_like
        Wavelength :math:`\lambda` in micrometers (:math:`\mu m`).

    Returns
    -------
    numeric or ndarray
        Air depolarisation.

    See Also
    --------
    F_air_Young1981, F_air_Bates1984, F_air_Bodhaine1999

    Notes
    -----
    -   The argument *wavelength* is only provided for consistency with the
        other air depolarisation methods but is actually not used as this
        definition is essentially a constant in its current implementation.

    Examples
    --------
    >>> F_air_Penndorf1957(0.555)
    array(1.0608)
    """

    wl = np.asarray(wavelength)

    return np.resize(np.array([1.0608]), wl.shape)


def F_air_Young1981(wavelength):
    """
    Returns :math:`(6+3_p)/(6-7_p)`, the depolarisation term :math:`F(air)` or
    *King Factor* using *Young (1981)* method.

    Parameters
    ----------
    wavelength : numeric or array_like
        Wavelength :math:`\lambda` in micrometers (:math:`\mu m`).

    Returns
    -------
    numeric or ndarray
        Air depolarisation.

    See Also
    --------
    F_air_Penndorf1957, F_air_Bates1984, F_air_Bodhaine1999

    Notes
    -----
    -   The argument *wavelength* is only provided for consistency with the
        other air depolarisation methods but is actually not used as this
        definition is essentially a constant in its current implementation.

    Examples
    --------
    >>> F_air_Young1981(0.555)
    array(1.048)
    """

    wl = np.asarray(wavelength)

    return np.resize(np.array([1.0480]), wl.shape)


def F_air_Bates1984(wavelength):
    """
    Returns :math:`(6+3_p)/(6-7_p)`, the depolarisation term :math:`F(air)` or
    *King Factor* as function of wavelength :math:`\lambda` in micrometers
    (:math:`\mu m`) using *Bates (1984)* method.

    Parameters
    ----------
    wavelength : numeric or array_like
        Wavelength :math:`\lambda` in micrometers (:math:`\mu m`).

    Returns
    -------
    numeric or ndarray
        Air depolarisation.

    See Also
    --------
    F_air_Penndorf1957, F_air_Young1981, F_air_Bodhaine1999

    Examples
    --------
    >>> F_air_Bates1984(0.555)  # doctest: +ELLIPSIS
    1.0481535...
    """

    O2 = O2_depolarisation(wavelength)
    N2 = N2_depolarisation(wavelength)
    Ar = 1.00
    CO2 = 1.15

    F_air = ((78.084 * N2 + 20.946 * O2 + CO2 + Ar) /
             (78.084 + 20.946 + Ar + CO2))

    return F_air


def F_air_Bodhaine1999(wavelength,
                       CO2_concentration=STANDARD_CO2_CONCENTRATION):
    """
    Returns :math:`(6+3_p)/(6-7_p)`, the depolarisation term :math:`F(air)` or
    *King Factor* as function of wavelength :math:`\lambda` in micrometers
    (:math:`\mu m`) and :math:`CO_2` concentration in parts per million (ppm)
    using *Bodhaine, Wood, Dutton and Slusser (1999)* method.

    Parameters
    ----------
    wavelength : numeric or array_like
        Wavelength :math:`\lambda` in micrometers (:math:`\mu m`).
    CO2_concentration : numeric or array_like, optional
        :math:`CO_2` concentration in parts per million (ppm).

    Returns
    -------
    numeric or ndarray
        Air depolarisation.

    See Also
    --------
    F_air_Penndorf1957, F_air_Young1981, F_air_Bates1984

    Examples
    --------
    >>> F_air_Bodhaine1999(0.555)  # doctest: +ELLIPSIS
    1.1246916...
    """

    O2 = O2_depolarisation(wavelength)
    N2 = N2_depolarisation(wavelength)
    CO2_c = np.asarray(CO2_concentration)

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
        :math:`6.02214179x10^{23}`, which is different from the reference [1]_
        value :math:`6.0221367x10^{23}`.

    Examples
    --------
    >>> molecular_density(288.15)  # doctest: +ELLIPSIS
    2.5469021...e+19
    >>> molecular_density(288.15, 6.0221367e23)  # doctest: +ELLIPSIS
    2.5468999...e+19
    """

    T = np.asarray(temperature)

    N_s = (avogadro_constant / 22.4141) * (273.15 / T) * (1 / 1000)

    return N_s


def mean_molecular_weights(CO2_concentration=STANDARD_CO2_CONCENTRATION):
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


def gravity_List1968(latitude=DEFAULT_LATITUDE, altitude=DEFAULT_ALTITUDE):
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
    >>> gravity_List1968(0, 1500)  # doctest: +ELLIPSIS
    977.5726106...

    Gravity :math:`g` for Paris:

    >>> gravity_List1968(48.8567, 35)  # doctest: +ELLIPSIS
    980.9524178...
    """

    latitude = np.asarray(latitude)
    altitude = np.asarray(altitude)

    cos2phi = np.cos(2 * np.radians(latitude))

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
                             n_s=air_refraction_index_Bodhaine1999,
                             F_air=F_air_Bodhaine1999):
    """
    Returns the scattering cross section per molecule :math:`\sigma` of dry air
    as function of wavelength :math:`\lambda` in centimeters (cm) using given
    :math:`CO_2` concentration in parts per million (ppm) and temperature
    :math:`T[K]` in kelvin degrees following *Van de Hulst (1957)* method.

    Parameters
    ----------
    wavelength : numeric or array_like
        Wavelength :math:`\lambda` in centimeters (cm).
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
        Scattering cross section per molecule :math:`\sigma` of dry air.

    Warning
    -------
    Unlike most objects of :mod:`colour.phenomenons.rayleigh` module,
    :func:`colour.phenomenons.rayleigh.scattering_cross_section` expects
    wavelength :math:`\lambda` to be expressed in centimeters (cm).

    Examples
    --------
    >>> scattering_cross_section(555 * 10e-8)  # doctest: +ELLIPSIS
    4.6613309...e-27
    """

    wl = np.asarray(wavelength)
    CO2_c = np.asarray(CO2_concentration)
    temperature = np.asarray(temperature)

    wl_micrometers = wl * 10e3

    n_s = n_s(wl_micrometers)
    # n_s = n_s(**filter_kwargs(
    #     n_s, wavelength=wl_micrometers, CO2_concentration=CO2_c))
    N_s = molecular_density(temperature, avogadro_constant)
    F_air = F_air(**filter_kwargs(
        F_air, wavelength=wl_micrometers, CO2_concentration=CO2_c))

    sigma = (24 * np.pi ** 3 * (n_s ** 2 - 1) ** 2 /
             (wl ** 4 * N_s ** 2 * (n_s ** 2 + 2) ** 2))
    sigma *= F_air

    return sigma


def rayleigh_optical_depth(wavelength,
                           CO2_concentration=STANDARD_CO2_CONCENTRATION,
                           temperature=STANDARD_AIR_TEMPERATURE,
                           pressure=AVERAGE_PRESSURE_MEAN_SEA_LEVEL,
                           latitude=DEFAULT_LATITUDE,
                           altitude=DEFAULT_ALTITUDE,
                           avogadro_constant=AVOGADRO_CONSTANT,
                           n_s=air_refraction_index_Bodhaine1999,
                           F_air=F_air_Bodhaine1999):
    """
    Returns the *Rayleigh* optical depth :math:`T_r(\lambda)` as function of
    wavelength :math:`\lambda` in centimeters (cm).

    Parameters
    ----------
    wavelength : numeric or array_like
        Wavelength :math:`\lambda` in centimeters (cm).
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
        *Rayleigh* optical depth :math:`T_r(\lambda)`.

    Warning
    -------
    Unlike most objects of :mod:`colour.phenomenons.rayleigh` module,
    :func:`colour.phenomenons.rayleigh.rayleigh_optical_depth` expects
    wavelength :math:`\lambda` to be expressed in centimeters (cm).

    Examples
    --------
    >>> rayleigh_optical_depth(555 * 10e-8)  # doctest: +ELLIPSIS
    0.1004070...
    """

    wavelength = np.asarray(wavelength)
    CO2_c = np.asarray(CO2_concentration)
    latitude = np.asarray(latitude)
    altitude = np.asarray(altitude)
    # Conversion from pascal to dyne/cm2.
    P = np.asarray(pressure * 10)

    sigma = scattering_cross_section(wavelength,
                                     CO2_c,
                                     temperature,
                                     avogadro_constant,
                                     n_s,
                                     F_air)

    m_a = mean_molecular_weights(CO2_c)
    g = gravity_List1968(latitude, altitude)

    T_R = sigma * (P * avogadro_constant) / (m_a * g)

    return T_R


rayleigh_scattering = rayleigh_optical_depth


def rayleigh_scattering_spd(shape=DEFAULT_SPECTRAL_SHAPE,
                            CO2_concentration=STANDARD_CO2_CONCENTRATION,
                            temperature=STANDARD_AIR_TEMPERATURE,
                            pressure=AVERAGE_PRESSURE_MEAN_SEA_LEVEL,
                            latitude=DEFAULT_LATITUDE,
                            altitude=DEFAULT_ALTITUDE,
                            avogadro_constant=AVOGADRO_CONSTANT,
                            n_s=air_refraction_index_Bodhaine1999,
                            F_air=F_air_Bodhaine1999):
    """
    Returns the *Rayleigh* spectral power distribution for given spectral
    shape.

    Parameters
    ----------
    shape : SpectralShape, optional
        Spectral shape used to create the *Rayleigh* scattering spectral power
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
    SpectralPowerDistribution
        *Rayleigh* optical depth spectral power distribution.

    Examples
    --------
    >>> print(rayleigh_scattering_spd())
    SpectralPowerDistribution('Rayleigh Scattering - 300 ppm, 288.15 K, \
101325 Pa, 0 Degrees, 0 m', (360.0, 780.0, 1.0))
    """

    wavelengths = shape.range()
    return SpectralPowerDistribution(
        name=('Rayleigh Scattering - {0} ppm, {1} K, {2} Pa, {3} Degrees, '
              '{4} m').format(CO2_concentration,
                              temperature,
                              pressure,
                              latitude,
                              altitude),
        data=dict(zip(wavelengths,
                      rayleigh_optical_depth(wavelengths * 10e-8,
                                             CO2_concentration,
                                             temperature,
                                             pressure,
                                             latitude,
                                             altitude,
                                             avogadro_constant,
                                             n_s,
                                             F_air))))
