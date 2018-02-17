# -*- coding: utf-8 -*-
"""
Meng et al. (2015) - Reflectance Recovery
=========================================

Defines objects for reflectance recovery using *Meng, Simon and Hanika (2015)*
method:

-   :func:`colour.recovery.XYZ_to_spectral_Meng2015`

See Also
--------
`Meng et al. (2015) - Reflectance Recovery Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/recovery/meng2015.ipynb>`_

References
----------
-   :cite:`Meng2015c` : Meng, J., Simon, F., Hanika, J., & Dachsbacher, C.
    (2015). Physically Meaningful Rendering using Tristimulus Colours. Computer
    Graphics Forum, 34(4), 31-40. doi:10.1111/cgf.12676
"""

from __future__ import division, unicode_literals

import numpy as np
from scipy.optimize import minimize

from colour.colorimetry import (STANDARD_OBSERVERS_CMFS,
                                SpectralPowerDistribution, SpectralShape,
                                ones_spd, spectral_to_XYZ_integration)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
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
    values using *Meng et alii (2015)* method.

    Parameters
    ----------
    XYZ : array_like, (3,)
        *CIE XYZ* tristimulus values to recover the spectral power distribution
        from.
    cmfs : XYZ_ColourMatchingFunctions
        Standard observer colour matching functions.
    interval : numeric, optional
        Wavelength :math:`\lambda_{i}` range interval in nm. The smaller
        ``interval`` is, the longer the computations will be.
    tolerance : numeric, optional
        Tolerance for termination. The lower ``tolerance`` is, the smoother
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
        values is :func:`colour.colorimetry.spectral_to_XYZ_integration`
        definition because it processes any measurement interval opposed to
        :func:`colour.colorimetry.spectral_to_XYZ_ASTME30815` definition that
        handles only measurement interval of 1, 5, 10 or 20nm.

    References
    ----------
    -   :cite:`Meng2015c`

    Examples
    --------
    >>> from colour.utilities import numpy_print_options
    >>> XYZ = np.array([0.07049534, 0.10080000, 0.09558313])
    >>> spd = XYZ_to_spectral_Meng2015(XYZ, interval=10)
    >>> with numpy_print_options(suppress=True):
    ...     spd  # doctest: +ELLIPSIS
    SpectralPowerDistribution([[ 360.        ,    0.0788075...],
                               [ 370.        ,    0.0788543...],
                               [ 380.        ,    0.0788825...],
                               [ 390.        ,    0.0788714...],
                               [ 400.        ,    0.0788911...],
                               [ 410.        ,    0.07893  ...],
                               [ 420.        ,    0.0797471...],
                               [ 430.        ,    0.0813339...],
                               [ 440.        ,    0.0840145...],
                               [ 450.        ,    0.0892826...],
                               [ 460.        ,    0.0965359...],
                               [ 470.        ,    0.1053176...],
                               [ 480.        ,    0.1150921...],
                               [ 490.        ,    0.1244252...],
                               [ 500.        ,    0.1326083...],
                               [ 510.        ,    0.1390282...],
                               [ 520.        ,    0.1423548...],
                               [ 530.        ,    0.1414636...],
                               [ 540.        ,    0.1365195...],
                               [ 550.        ,    0.1277319...],
                               [ 560.        ,    0.1152622...],
                               [ 570.        ,    0.1004513...],
                               [ 580.        ,    0.0844187...],
                               [ 590.        ,    0.0686863...],
                               [ 600.        ,    0.0543013...],
                               [ 610.        ,    0.0423486...],
                               [ 620.        ,    0.0333861...],
                               [ 630.        ,    0.0273558...],
                               [ 640.        ,    0.0233407...],
                               [ 650.        ,    0.0211208...],
                               [ 660.        ,    0.0197248...],
                               [ 670.        ,    0.0187157...],
                               [ 680.        ,    0.0181510...],
                               [ 690.        ,    0.0179691...],
                               [ 700.        ,    0.0179247...],
                               [ 710.        ,    0.0178665...],
                               [ 720.        ,    0.0178005...],
                               [ 730.        ,    0.0177570...],
                               [ 740.        ,    0.0177090...],
                               [ 750.        ,    0.0175743...],
                               [ 760.        ,    0.0175058...],
                               [ 770.        ,    0.0174492...],
                               [ 780.        ,    0.0174984...],
                               [ 790.        ,    0.0175667...],
                               [ 800.        ,    0.0175657...],
                               [ 810.        ,    0.0175319...],
                               [ 820.        ,    0.0175184...],
                               [ 830.        ,    0.0175390...]],
                              interpolator=SpragueInterpolator,
                              interpolator_args={},
                              extrapolator=Extrapolator,
                              extrapolator_args={...})
    >>> spectral_to_XYZ_integration(spd) / 100  # doctest: +ELLIPSIS
    array([ 0.0705100...,  0.1007987...,  0.0956738...])
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
        dict(zip(wavelengths, result.x * 100)),
        name='Meng (2015) - {0}'.format(XYZ))
