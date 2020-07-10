# -*- coding: utf-8 -*-
"""
Otsu et al. (2018) - Reflectance Recovery
==============================================

Defines objects for reflectance recovery, i.e. spectral upsampling, using
*Otsu et al. (2018)* method:

-   :func:`colour.recovery.XYZ_to_sd_Otsu2018`

References
----------
-   :cite:`Otsu2018` : Otsu, H., Yamamoto, M. & Hachisuka, T. (2018)
    Reproducing Spectral Reflectances From Tristimulus Colours. Computer
    Graphics Forum. 37(6), 370–381. doi:10.1111/cgf.13332
"""

from __future__ import division, unicode_literals

import numpy as np

from colour import ILLUMINANT_SDS
from colour.colorimetry import (STANDARD_OBSERVER_CMFS, SpectralDistribution,
                                sd_to_XYZ)
from colour.models import XYZ_to_xy
from colour.recovery import (OTSU_2018_SPECTRAL_SHAPE,
                             OTSU_2018_BASIS_FUNCTIONS, OTSU_2018_MEANS,
                             select_cluster_Otsu2018)
from colour.utilities import as_float_array

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['XYZ_to_sd_Otsu2018']


def XYZ_to_sd_Otsu2018(
        XYZ,
        cmfs=STANDARD_OBSERVER_CMFS['CIE 1931 2 Degree Standard Observer']
        .copy().align(OTSU_2018_SPECTRAL_SHAPE),
        illuminant=ILLUMINANT_SDS['D65'].copy().align(
            OTSU_2018_SPECTRAL_SHAPE),
        clip=True):
    """
    Recovers the spectral distribution of given *CIE XYZ* tristimulus values
    using *Otsu et al. (2018)* method.

    Parameters
    ----------
    XYZ : array_like, (3,)
        *CIE XYZ* tristimulus values to recover the spectral distribution from.
    cmfs : XYZ_ColourMatchingFunctions, optional
        Standard observer colour matching functions.
    illuminant : SpectralDistribution, optional
        Illuminant spectral distribution.

    Other Parameters
    ----------------
    clip : bool, optional
        If true (the default), values below zero and above unity in the
        recovered spectral distributions will be clipped. This ensures the
        returned reflectance is physical and conserves energy, but will cause
        noticeable colour differences in case of very saturated colours.

    Returns
    -------
    SpectralDistribution
        Recovered spectral distribution. Its shape is always
        ``OTSU_2018_SPECTRAL_SHAPE``.
    """

    XYZ = as_float_array(XYZ)
    xy = XYZ_to_xy(XYZ)
    cluster = select_cluster_Otsu2018(xy)

    basis_functions = OTSU_2018_BASIS_FUNCTIONS[cluster]
    mean = OTSU_2018_MEANS[cluster]

    M = np.empty((3, 3))
    for i in range(3):
        sd = SpectralDistribution(
            basis_functions[i, :],
            OTSU_2018_SPECTRAL_SHAPE.range(),
        )
        M[:, i] = sd_to_XYZ(sd, illuminant=illuminant) / 100
    M_inverse = np.linalg.inv(M)

    sd = SpectralDistribution(mean, OTSU_2018_SPECTRAL_SHAPE.range())
    XYZ_mu = sd_to_XYZ(sd, illuminant=illuminant) / 100

    weights = np.dot(M_inverse, XYZ - XYZ_mu)
    recovered_sd = np.dot(weights, basis_functions) + mean

    if clip:
        recovered_sd = np.clip(recovered_sd, 0, 1)

    return SpectralDistribution(recovered_sd, OTSU_2018_SPECTRAL_SHAPE.range())
