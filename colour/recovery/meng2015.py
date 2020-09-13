# -*- coding: utf-8 -*-
"""
Meng et al. (2015) - Reflectance Recovery
=========================================

Defines objects for reflectance recovery using *Meng, Simon and Hanika (2015)*
method:

-   :func:`colour.recovery.XYZ_to_sd_Meng2015`

References
----------
-   :cite:`Meng2015c` : Meng, J., Simon, F., Hanika, J., & Dachsbacher, C.
    (2015). Physically Meaningful Rendering using Tristimulus Colours. Computer
    Graphics Forum, 34(4), 31-40. doi:10.1111/cgf.12676
"""

from __future__ import division, unicode_literals

import numpy as np
from scipy.optimize import minimize

from colour.colorimetry import (MSDS_CMFS_STANDARD_OBSERVER, SDS_ILLUMINANTS,
                                SpectralDistribution, SpectralShape, sd_ones,
                                sd_to_XYZ_integration)
from colour.utilities import to_domain_1, from_range_100, runtime_warning
from colour.utilities.deprecation import handle_arguments_deprecation

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['SPECTRAL_SHAPE_MENG2015', 'XYZ_to_sd_Meng2015']

SPECTRAL_SHAPE_MENG2015 = SpectralShape(360, 780, 5)
"""
Spectral shape according to *ASTM E308-15* practise shape but using an interval
of 5.

SPECTRAL_SHAPE_MENG2015 : SpectralShape
"""


def XYZ_to_sd_Meng2015(
        XYZ,
        cmfs=MSDS_CMFS_STANDARD_OBSERVER['CIE 1931 2 Degree Standard Observer']
        .copy().align(SPECTRAL_SHAPE_MENG2015),
        illuminant=SDS_ILLUMINANTS['D65'].copy().align(
            SPECTRAL_SHAPE_MENG2015),
        optimisation_kwargs=None,
        **kwargs):
    """
    Recovers the spectral distribution of given *CIE XYZ* tristimulus values
    using *Meng et al. (2015)* method.

    Parameters
    ----------
    XYZ : array_like, (3,)
        *CIE XYZ* tristimulus values to recover the spectral distribution from.
    cmfs : XYZ_ColourMatchingFunctions
        Standard observer colour matching functions. The wavelength
        :math:`\\lambda_{i}` range interval of the colour matching functions
        affects directly the time the computations take. The current default
        interval of 5 is a good compromise between precision and time spent.
    illuminant : SpectralDistribution, optional
        Illuminant spectral distribution.
    optimisation_kwargs : dict_like, optional
        Parameters for :func:`scipy.optimize.minimize` definition.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        Keywords arguments for deprecation management.

    Returns
    -------
    SpectralDistribution
        Recovered spectral distribution.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    -   The definition used to convert spectrum to *CIE XYZ* tristimulus
        values is :func:`colour.colorimetry.spectral_to_XYZ_integration`
        definition because it processes any measurement interval opposed to
        :func:`colour.colorimetry.sd_to_XYZ_ASTME308` definition that
        handles only measurement interval of 1, 5, 10 or 20nm.

    References
    ----------
    :cite:`Meng2015c`

    Examples
    --------
    >>> from colour.utilities import numpy_print_options
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> cmfs = (
    ...     MSDS_CMFS_STANDARD_OBSERVER['CIE 1931 2 Degree Standard Observer'].
    ...     copy().align(SpectralShape(360, 780, 10))
    ... )
    >>> illuminant = SDS_ILLUMINANTS['D65'].copy().align(cmfs.shape)
    >>> sd = XYZ_to_sd_Meng2015(XYZ, cmfs, illuminant)
    >>> with numpy_print_options(suppress=True):
    ...     # Doctests skip for Python 2.x compatibility.
    ...     sd  # doctest: +SKIP
    SpectralDistribution([[ 360.        ,    0.0765153...],
                          [ 370.        ,    0.0764771...],
                          [ 380.        ,    0.0764286...],
                          [ 390.        ,    0.0764329...],
                          [ 400.        ,    0.0765863...],
                          [ 410.        ,    0.0764339...],
                          [ 420.        ,    0.0757213...],
                          [ 430.        ,    0.0733091...],
                          [ 440.        ,    0.0676493...],
                          [ 450.        ,    0.0577616...],
                          [ 460.        ,    0.0440805...],
                          [ 470.        ,    0.0284802...],
                          [ 480.        ,    0.0138019...],
                          [ 490.        ,    0.0033557...],
                          [ 500.        ,    0.       ...],
                          [ 510.        ,    0.       ...],
                          [ 520.        ,    0.       ...],
                          [ 530.        ,    0.       ...],
                          [ 540.        ,    0.0055360...],
                          [ 550.        ,    0.0317335...],
                          [ 560.        ,    0.075457 ...],
                          [ 570.        ,    0.1314930...],
                          [ 580.        ,    0.1938219...],
                          [ 590.        ,    0.2559747...],
                          [ 600.        ,    0.3122869...],
                          [ 610.        ,    0.3584363...],
                          [ 620.        ,    0.3927112...],
                          [ 630.        ,    0.4158866...],
                          [ 640.        ,    0.4305832...],
                          [ 650.        ,    0.4391142...],
                          [ 660.        ,    0.4439484...],
                          [ 670.        ,    0.4464121...],
                          [ 680.        ,    0.4475718...],
                          [ 690.        ,    0.4481182...],
                          [ 700.        ,    0.4483734...],
                          [ 710.        ,    0.4484743...],
                          [ 720.        ,    0.4485753...],
                          [ 730.        ,    0.4486474...],
                          [ 740.        ,    0.4486629...],
                          [ 750.        ,    0.4486995...],
                          [ 760.        ,    0.4486925...],
                          [ 770.        ,    0.4486794...],
                          [ 780.        ,    0.4486982...]],
                         interpolator=SpragueInterpolator,
                         interpolator_kwargs={},
                         extrapolator=Extrapolator,
                         extrapolator_kwargs={...})
    >>> sd_to_XYZ_integration(sd, cmfs, illuminant) / 100  # doctest: +ELLIPSIS
    array([ 0.2065400...,  0.1219722...,  0.0513695...])
    """

    optimisation_kwargs = handle_arguments_deprecation({
        'ArgumentRenamed': [['optimisation_parameters', 'optimisation_kwargs']
                            ],
    }, **kwargs).get('optimisation_kwargs', optimisation_kwargs)

    XYZ = to_domain_1(XYZ)

    if illuminant.shape != cmfs.shape:
        runtime_warning(
            'Aligning "{0}" illuminant shape to "{1}" colour matching '
            'functions shape.'.format(illuminant.name, cmfs.name))
        illuminant = illuminant.copy().align(cmfs.shape)

    sd = sd_ones(cmfs.shape)

    def objective_function(a):
        """
        Objective function.
        """

        return np.sum(np.diff(a) ** 2)

    def constraint_function(a):
        """
        Function defining the constraint.
        """

        sd[:] = a
        return sd_to_XYZ_integration(
            sd, cmfs=cmfs, illuminant=illuminant) - XYZ

    wavelengths = sd.wavelengths
    bins = wavelengths.size

    optimisation_settings = {
        'method': 'SLSQP',
        'constraints': {
            'type': 'eq',
            'fun': constraint_function
        },
        'bounds': np.tile(np.array([0, 1000]), (bins, 1)),
        'options': {
            'ftol': 1e-10,
        },
    }
    if optimisation_kwargs is not None:
        optimisation_settings.update(optimisation_kwargs)

    result = minimize(objective_function, sd.values, **optimisation_settings)

    if not result.success:
        raise RuntimeError(
            'Optimization failed for {0} after {1} iterations: "{2}".'.format(
                XYZ, result.nit, result.message))

    return SpectralDistribution(
        from_range_100(result.x * 100),
        wavelengths,
        name='{0} (XYZ) - Meng (2015)'.format(XYZ))
