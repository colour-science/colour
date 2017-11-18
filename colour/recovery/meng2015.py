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
    >>> from colour import numpy_print_options
    >>> XYZ = np.array([0.07049534, 0.10080000, 0.09558313])
    >>> spd = XYZ_to_spectral_Meng2015(XYZ)
    >>> with numpy_print_options(suppress=True):
    ...     spd  # doctest: +ELLIPSIS
    SpectralPowerDistribution([[ 360.        ,    0.0007963...],
                               [ 365.        ,    0.0007964...],
                               [ 370.        ,    0.0007965...],
                               [ 375.        ,    0.0007958...],
                               [ 380.        ,    0.0007960...],
                               [ 385.        ,    0.0007964...],
                               [ 390.        ,    0.0007969...],
                               [ 395.        ,    0.0007964...],
                               [ 400.        ,    0.0007956...],
                               [ 405.        ,    0.0007954...],
                               [ 410.        ,    0.0007957...],
                               [ 415.        ,    0.0007967...],
                               [ 420.        ,    0.0007992...],
                               [ 425.        ,    0.0008038...],
                               [ 430.        ,    0.0008126...],
                               [ 435.        ,    0.0008256...],
                               [ 440.        ,    0.0008426...],
                               [ 445.        ,    0.0008663...],
                               [ 450.        ,    0.0008950...],
                               [ 455.        ,    0.0009282...],
                               [ 460.        ,    0.0009649...],
                               [ 465.        ,    0.0010056...],
                               [ 470.        ,    0.0010488...],
                               [ 475.        ,    0.0010919...],
                               [ 480.        ,    0.0011393...],
                               [ 485.        ,    0.001188 ...],
                               [ 490.        ,    0.0012356...],
                               [ 495.        ,    0.0012807...],
                               [ 500.        ,    0.0013231...],
                               [ 505.        ,    0.0013599...],
                               [ 510.        ,    0.0013905...],
                               [ 515.        ,    0.0014128...],
                               [ 520.        ,    0.0014255...],
                               [ 525.        ,    0.0014278...],
                               [ 530.        ,    0.0014177...],
                               [ 535.        ,    0.0013958...],
                               [ 540.        ,    0.0013645...],
                               [ 545.        ,    0.0013244...],
                               [ 550.        ,    0.0012747...],
                               [ 555.        ,    0.0012171...],
                               [ 560.        ,    0.0011523...],
                               [ 565.        ,    0.0010820...],
                               [ 570.        ,    0.0010071...],
                               [ 575.        ,    0.0009273...],
                               [ 580.        ,    0.0008466...],
                               [ 585.        ,    0.0007662...],
                               [ 590.        ,    0.0006878...],
                               [ 595.        ,    0.0006146...],
                               [ 600.        ,    0.0005460...],
                               [ 605.        ,    0.0004823...],
                               [ 610.        ,    0.0004259...],
                               [ 615.        ,    0.0003761...],
                               [ 620.        ,    0.0003339...],
                               [ 625.        ,    0.0002974...],
                               [ 630.        ,    0.0002681...],
                               [ 635.        ,    0.0002448...],
                               [ 640.        ,    0.0002260...],
                               [ 645.        ,    0.0002124...],
                               [ 650.        ,    0.0002020...],
                               [ 655.        ,    0.0001943...],
                               [ 660.        ,    0.0001892...],
                               [ 665.        ,    0.0001862...],
                               [ 670.        ,    0.0001840...],
                               [ 675.        ,    0.0001820...],
                               [ 680.        ,    0.0001804...],
                               [ 685.        ,    0.0001783...],
                               [ 690.        ,    0.0001772...],
                               [ 695.        ,    0.0001762...],
                               [ 700.        ,    0.0001743...],
                               [ 705.        ,    0.0001738...],
                               [ 710.        ,    0.0001741...],
                               [ 715.        ,    0.0001737...],
                               [ 720.        ,    0.0001723...],
                               [ 725.        ,    0.0001711...],
                               [ 730.        ,    0.0001707...],
                               [ 735.        ,    0.0001701...],
                               [ 740.        ,    0.0001696...],
                               [ 745.        ,    0.0001695...],
                               [ 750.        ,    0.0001689...],
                               [ 755.        ,    0.0001690...],
                               [ 760.        ,    0.0001692...],
                               [ 765.        ,    0.0001691...],
                               [ 770.        ,    0.0001685...],
                               [ 775.        ,    0.0001682...],
                               [ 780.        ,    0.0001687...],
                               [ 785.        ,    0.0001690...],
                               [ 790.        ,    0.0001692...],
                               [ 795.        ,    0.0001694...],
                               [ 800.        ,    0.0001696...],
                               [ 805.        ,    0.0001699...],
                               [ 810.        ,    0.0001702...],
                               [ 815.        ,    0.0001705...],
                               [ 820.        ,    0.0001710...],
                               [ 825.        ,    0.0001716...],
                               [ 830.        ,    0.0001720...]],
                              interpolator=SpragueInterpolator,
                              interpolator_args={},
                              extrapolator=Extrapolator,
                              extrapolator_args={...})
    >>> spectral_to_XYZ_integration(spd)  # doctest: +ELLIPSIS
    array([ 0.0704952...,  0.1007999...,  0.0955824...])
    """

    XYZ = np.asarray(XYZ)
    shape = SpectralShape(cmfs.shape.start, cmfs.shape.end, interval)
    cmfs = cmfs.copy().align(shape)
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

    return SpectralPowerDistribution(
        dict(zip(wavelengths, result.x)), name='Meng (2015) - {0}'.format(XYZ))
