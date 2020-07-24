# -*- coding: utf-8 -*-
"""
Otsu et al. (2018) - Reflectance Recovery
=========================================

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
                             OTSU_2018_SELECTOR_ARRAY)
from colour.utilities import as_float_array

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['Otsu2018Dataset', 'XYZ_to_sd_Otsu2018']


class Otsu2018Dataset:
    """
    Stores all the information needed for the *Otsu et al. (2018)* spectral
    upsampling method. Datasets can be either generated and turned into
    this form using ``Otsu2018Tree.to_dataset`` or loaded from disk.

    Attributes
    ==========
    shape: SpectralShape
        Shape of the spectral data.
    basis_functions : ndarray(n, 3, m)
        Three basis functions for every cluster.
    means : ndarray(n, m)
        Mean for every cluster.
    selector_array : ndarray(k, 4)
        Array describing how to select the appropriate cluster. See
        ``Otsu2018Dataset.select`` for details.
    """

    def __init__(self,
                 shape=None,
                 basis_functions=None,
                 means=None,
                 selector_array=None):
        self.shape = shape
        self.basis_functions = basis_functions
        self.means = means
        self.selector_array = selector_array

    def select(self, xy):
        """
        Returns the cluster index appropriate for the given *CIE xy*
        coordinates.

        Parameters
        ==========
        ndarray : (2,)
            *CIE xy* chromaticity coordinates.

        Returns
        =======
        int
            Cluster index.
        """

        i = 0
        while True:
            row = self.selector_array[i, :]
            direction, origin, lesser_index, greater_index = row

            if xy[int(direction)] <= origin:
                index = int(lesser_index)
            else:
                index = int(greater_index)

            if index < 0:
                i = -index
            else:
                return index

    def cluster(self, xy):
        """
        Returns the basis functions and dataset mean for the given *CIE xy*
        coordinates.

        Parameters
        ==========
        ndarray : (2,)
            *CIE xy* chromaticity coordinates.

        Returns
        =======
        basis_functions : ndarray (3, n)
            Three basis functions.
        mean : ndarray (n,)
            Dataset mean.
        """

        index = self.select(xy)
        return self.basis_functions[index, :, :], self.means[index, :]


builtin_Otsu2018_dataset = Otsu2018Dataset(
    OTSU_2018_SPECTRAL_SHAPE, OTSU_2018_BASIS_FUNCTIONS, OTSU_2018_MEANS,
    OTSU_2018_SELECTOR_ARRAY)
"""
Builtin *Otsu et al. (2018)* dataset in the form of an ``Otsu2018Dataset``
object, usable in ``XYZ_to_sd_Otsu2018``, among others.
"""


def XYZ_to_sd_Otsu2018(
        XYZ,
        cmfs=STANDARD_OBSERVER_CMFS['CIE 1931 2 Degree Standard Observer']
        .copy().align(OTSU_2018_SPECTRAL_SHAPE),
        illuminant=ILLUMINANT_SDS['D65'].copy().align(
            OTSU_2018_SPECTRAL_SHAPE),
        dataset=builtin_Otsu2018_dataset,
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
    dataset : Otsu2018Dataset, optional
        Dataset to use for reconstruction. The default is to use the published
        data.

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

    basis_functions, mean = dataset.cluster(xy)

    M = np.empty((3, 3))
    for i in range(3):
        sd = SpectralDistribution(basis_functions[i, :], dataset.shape.range())
        M[:, i] = sd_to_XYZ(sd, illuminant=illuminant) / 100
    M_inverse = np.linalg.inv(M)

    sd = SpectralDistribution(mean, dataset.shape.range())
    XYZ_mu = sd_to_XYZ(sd, illuminant=illuminant) / 100

    weights = np.dot(M_inverse, XYZ - XYZ_mu)
    recovered_sd = np.dot(weights, basis_functions) + mean

    if clip:
        recovered_sd = np.clip(recovered_sd, 0, 1)

    return SpectralDistribution(recovered_sd, dataset.shape.range())
