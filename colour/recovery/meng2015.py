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

import colour.ndarray as np
from scipy.optimize import minimize

from colour.colorimetry import (MSDS_CMFS_STANDARD_OBSERVER,
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
        illuminant=sd_ones(SPECTRAL_SHAPE_MENG2015),
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
    >>> sd = XYZ_to_sd_Meng2015(XYZ, cmfs)
    >>> with numpy_print_options(suppress=True):
    ...     # Doctests skip for Python 2.x compatibility.
    ...     sd  # doctest: +SKIP
    SpectralDistribution([[ 360.        ,    0.0780114...],
                          [ 370.        ,    0.0780316...],
                          [ 380.        ,    0.0780471...],
                          [ 390.        ,    0.0780351...],
                          [ 400.        ,    0.0779702...],
                          [ 410.        ,    0.0778033...],
                          [ 420.        ,    0.0770958...],
                          [ 430.        ,    0.0748008...],
                          [ 440.        ,    0.0693230...],
                          [ 450.        ,    0.0601136...],
                          [ 460.        ,    0.0477407...],
                          [ 470.        ,    0.0334964...],
                          [ 480.        ,    0.0193352...],
                          [ 490.        ,    0.0074858...],
                          [ 500.        ,    0.0001225...],
                          [ 510.        ,    0.       ...],
                          [ 520.        ,    0.       ...],
                          [ 530.        ,    0.       ...],
                          [ 540.        ,    0.0124896...],
                          [ 550.        ,    0.0389831...],
                          [ 560.        ,    0.0775105...],
                          [ 570.        ,    0.1247947...],
                          [ 580.        ,    0.1765339...],
                          [ 590.        ,    0.2281918...],
                          [ 600.        ,    0.2751347...],
                          [ 610.        ,    0.3140115...],
                          [ 620.        ,    0.3433561...],
                          [ 630.        ,    0.3635777...],
                          [ 640.        ,    0.3765428...],
                          [ 650.        ,    0.3841726...],
                          [ 660.        ,    0.3883633...],
                          [ 670.        ,    0.3905415...],
                          [ 680.        ,    0.3916742...],
                          [ 690.        ,    0.3922554...],
                          [ 700.        ,    0.3925427...],
                          [ 710.        ,    0.3926783...],
                          [ 720.        ,    0.3927330...],
                          [ 730.        ,    0.3927586...],
                          [ 740.        ,    0.3927548...],
                          [ 750.        ,    0.3927681...],
                          [ 760.        ,    0.3927813...],
                          [ 770.        ,    0.3927840...],
                          [ 780.        ,    0.3927536...]],
                         interpolator=SpragueInterpolator,
                         interpolator_kwargs={},
                         extrapolator=Extrapolator,
                         extrapolator_kwargs={...})
    >>> sd_to_XYZ_integration(sd) / 100  # doctest: +ELLIPSIS
    array([ 0.2065812...,  0.1219752...,  0.0514132...])
    """

    optimisation_kwargs = handle_arguments_deprecation({
        'ArgumentRenamed': [['optimisation_parameters', 'optimisation_kwargs']
                            ],
    }, **kwargs).get('optimisation_kwargs', optimisation_kwargs)

    cupy = False

    if np.__name__ == 'cupy':
        XYZ = np.asnumpy(XYZ)
        np.set_ndimensional_array_backend('numpy')
        cupy = True

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

    wavelengths = np.array(sd.wavelengths)
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

    result = minimize(objective_function, np.array(sd.values),
                      **optimisation_settings)
    if not result.success:
        if cupy is True:
            np.set_ndimensional_array_backend('cupy')
        raise RuntimeError(
            'Optimization failed for {0} after {1} iterations: "{2}".'.format(
                XYZ, result.nit, result.message))

    name = 'Meng (2015) - {0}'.format(XYZ)
    returned = result.x
    if cupy is True:
        np.set_ndimensional_array_backend('cupy')
        returned = np.array(returned)
        wavelengths = np.array(wavelengths)

    return SpectralDistribution(
        from_range_100(returned * 100), wavelengths, name=name)
