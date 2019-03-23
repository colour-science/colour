# -*- coding: utf-8 -*-
"""
Meng et al. (2015) - Reflectance Recovery
=========================================

Defines objects for reflectance recovery using *Meng, Simon and Hanika (2015)*
method:

-   :func:`colour.recovery.XYZ_to_sd_Meng2015`

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

from colour.colorimetry import (STANDARD_OBSERVERS_CMFS, SpectralDistribution,
                                SpectralShape, sd_ones, sd_to_XYZ_integration)
from colour.utilities import to_domain_1, from_range_100

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['XYZ_to_sd_Meng2015']


def XYZ_to_sd_Meng2015(
        XYZ,
        cmfs=STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer'],
        interval=5,
        optimisation_parameters=None):
    """
    Recovers the spectral distribution of given *CIE XYZ* tristimulus values
    using *Meng et al. (2015)* method.

    Parameters
    ----------
    XYZ : array_like, (3,)
        *CIE XYZ* tristimulus values to recover the spectral distribution from.
    cmfs : XYZ_ColourMatchingFunctions
        Standard observer colour matching functions.
    interval : numeric, optional
        Wavelength :math:`\\lambda_{i}` range interval in nm. The smaller
        ``interval`` is, the longer the computations will be.
    optimisation_parameters : dict_like, optional
        Parameters for :func:`scipy.optimize.minimize` definition.

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
        :func:`colour.colorimetry.sd_to_XYZ_ASTME30815` definition that
        handles only measurement interval of 1, 5, 10 or 20nm.

    References
    ----------
    :cite:`Meng2015c`

    Examples
    --------
    >>> from colour.utilities import numpy_print_options
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> sd = XYZ_to_sd_Meng2015(XYZ, interval=10)
    >>> with numpy_print_options(suppress=True):
    ...     # Doctests skip for Python 2.x compatibility.
    ...     sd  # doctest: +SKIP
    SpectralDistribution([[ 360.        ,    0.0780368...],
                          [ 370.        ,    0.0780387...],
                          [ 380.        ,    0.0780469...],
                          [ 390.        ,    0.0780894...],
                          [ 400.        ,    0.0780285...],
                          [ 410.        ,    0.0777034...],
                          [ 420.        ,    0.0769175...],
                          [ 430.        ,    0.0746243...],
                          [ 440.        ,    0.0691410...],
                          [ 450.        ,    0.0599949...],
                          [ 460.        ,    0.04779  ...],
                          [ 470.        ,    0.0337270...],
                          [ 480.        ,    0.0196952...],
                          [ 490.        ,    0.0078056...],
                          [ 500.        ,    0.0004368...],
                          [ 510.        ,    0.0000065...],
                          [ 520.        ,    0.       ...],
                          [ 530.        ,    0.       ...],
                          [ 540.        ,    0.0124283...],
                          [ 550.        ,    0.0389186...],
                          [ 560.        ,    0.0774087...],
                          [ 570.        ,    0.1246716...],
                          [ 580.        ,    0.1765055...],
                          [ 590.        ,    0.2281652...],
                          [ 600.        ,    0.2751726...],
                          [ 610.        ,    0.3141208...],
                          [ 620.        ,    0.3434564...],
                          [ 630.        ,    0.3636521...],
                          [ 640.        ,    0.3765182...],
                          [ 650.        ,    0.3841561...],
                          [ 660.        ,    0.3884648...],
                          [ 670.        ,    0.3906975...],
                          [ 680.        ,    0.3918679...],
                          [ 690.        ,    0.3924590...],
                          [ 700.        ,    0.3927439...],
                          [ 710.        ,    0.3928570...],
                          [ 720.        ,    0.3928867...],
                          [ 730.        ,    0.3929099...],
                          [ 740.        ,    0.3928997...],
                          [ 750.        ,    0.3928827...],
                          [ 760.        ,    0.3928579...],
                          [ 770.        ,    0.3927857...],
                          [ 780.        ,    0.3927272...],
                          [ 790.        ,    0.3926867...],
                          [ 800.        ,    0.3926441...],
                          [ 810.        ,    0.3926385...],
                          [ 820.        ,    0.3926247...],
                          [ 830.        ,    0.3926105...]],
                         interpolator=SpragueInterpolator,
                         interpolator_args={},
                         extrapolator=Extrapolator,
                         extrapolator_args={...})
    >>> sd_to_XYZ_integration(sd) / 100  # doctest: +ELLIPSIS
    array([ 0.2065817...,  0.1219754...,  0.0514131...])
    """

    XYZ = to_domain_1(XYZ)
    shape = SpectralShape(cmfs.shape.start, cmfs.shape.end, interval)
    cmfs = cmfs.copy().align(shape)
    illuminant = sd_ones(shape)
    sd = sd_ones(shape)

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
            'maxiter': 2000
        },
    }
    if optimisation_parameters is not None:
        optimisation_settings.update(optimisation_parameters)

    result = minimize(objective_function, sd.values, **optimisation_settings)

    if not result.success:
        raise RuntimeError(
            'Optimization failed for {0} after {1} iterations: "{2}".'.format(
                XYZ, result.nit, result.message))

    return SpectralDistribution(
        from_range_100(result.x * 100),
        wavelengths,
        name='Meng (2015) - {0}'.format(XYZ))
