#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Meng et al. (2015) - Reflectance Recovery
=========================================

Defines objects for reflectance recovery using *Meng, Simon and Hanika (2015)*
method:

-   :func:`XYZ_to_spectral_Meng2015`

See Also
--------
`Meng et al. (2015) - Reflectance Recovery Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/recovery/meng2015.ipynb>`_

References
----------
.. [1]  Meng, J., Simon, F., & Hanika, J. (2015). Physically Meaningful
        Rendering using Tristimulus Colours, 34(4). Retrieved from
        http://jo.dreggn.org/home/2015_spectrum.pdf
"""

from __future__ import division, unicode_literals

import numpy as np
from scipy.optimize import minimize

from colour import (STANDARD_OBSERVERS_CMFS, SpectralPowerDistribution,
                    SpectralShape, ones_spd, spectral_to_XYZ_integration)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['XYZ_to_spectral_Meng2015']


def XYZ_to_spectral_Meng2015(
        XYZ,
        cmfs=STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer'],
        interval=5,
        tolerance=1e-10,
        maximum_iterations=2000):
    """
    Recovers the spectral power distribution of given *CIE XYZ* tristimulus
    values using *Meng et al. (2015)* method.

    Parameters
    ----------
    XYZ : array_like, (3,)
        *CIE XYZ* tristimulus values.
    cmfs : XYZ_ColourMatchingFunctions
        Standard observer colour matching functions.
    interval : numeric, optional
        Wavelength :math:`\lambda_{i}` range interval in nm. The smaller
        `interval` is, the longer the computations will be.
    tolerance : numeric, optional
        Tolerance for termination. The lower `tolerance` is, the smoother
        the recovered spectral power distribution will be.
    maximum_iterations : int, optional
        Maximum number of iterations to perform.

    Returns
    -------
    SpectralPowerDistribution
        Recovered spectral power distribution.

    Notes
    -----
    -   The definition used to convert spectrum to *CIE XYZ* tristimulus
        values is :func:`colour.spectral_to_XYZ_integration` definition
        because it processes any measurement interval opposed to
        :func:`colour.spectral_to_XYZ_ASTME30815` definition that handles only
        measurement interval of 1, 5, 10 or 20nm.

    Examples
    --------
    >>> XYZ = np.array([0.07049534, 0.10080000, 0.09558313])
    >>> spd = XYZ_to_spectral_Meng2015(XYZ)
    >>> print(spd)
    SpectralPowerDistribution('Meng (2015) - \
[ 0.07049534  0.1008      0.09558313]', (360.0, 830.0, 5.0))
    >>> spectral_to_XYZ_integration(spd)  # doctest: +ELLIPSIS
    array([ 0.0704952...,  0.1007999...,  0.0955824...])
    """

    XYZ = np.asarray(XYZ)
    shape = SpectralShape(cmfs.shape.start, cmfs.shape.end, interval)
    cmfs = cmfs.clone().align(shape)
    illuminant = ones_spd(shape)
    spd = ones_spd(shape)

    def function_objective(a):
        """
        Objective function.
        """

        return np.sum(np.diff(a) ** 2)

    def function_constraint(a):
        """
        Function defining the constraint.
        """

        spd[:] = a
        return spectral_to_XYZ_integration(
            spd, cmfs=cmfs, illuminant=illuminant) - XYZ

    wavelengths = spd.wavelengths
    bins = wavelengths.size

    constraints = {'type': 'eq', 'fun': function_constraint}

    bounds = np.tile(np.array([0, 1000]), (bins, 1))

    result = minimize(
        function_objective,
        spd.values,
        method='SLSQP',
        constraints=constraints,
        bounds=bounds,
        options={'ftol': tolerance,
                 'maxiter': maximum_iterations})

    if not result.success:
        raise RuntimeError(
            'Optimization failed for {0} after {1} iterations: "{2}".'.format(
                XYZ, result.nit, result.message))

    return SpectralPowerDistribution('Meng (2015) - {0}'.format(XYZ),
                                     dict(zip(wavelengths, result.x)))
