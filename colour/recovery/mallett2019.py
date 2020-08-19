# -*- coding: utf-8 -*-
"""
Mallett and Yuksel (2019) - Reflectance Recovery
================================================

Defines objects for reflectance recovery, i.e. spectral upsampling, using
*Mallett and Yuksel (2019)* method:

-   :func:`colour.recovery.sRGB_to_sd_Mallett2019`

References
----------
-   :cite:`Mallett2019` : Mallett, I., & Yuksel, C. (2019). Spectral Primary
    Decomposition for Rendering with sRGB Reflectance. Eurographics Symposium
    on Rendering - DL-Only and Industry Track, 7 pages. doi:10.2312/SR.20191216
"""

from __future__ import division, print_function, unicode_literals

import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import Bounds, LinearConstraint, minimize

from colour.colorimetry import SpectralDistribution, MultiSpectralDistributions
from colour.recovery import (BASIS_sRGB_MALLETT2019,
                             SPECTRAL_SHAPE_sRGB_MALLETT2019)
from colour.utilities import to_domain_1, runtime_warning

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'spectral_primary_decomposition_Mallett2019', 'RGB_to_sd_Mallett2019',
    'sRGB_to_sd_Mallett2019'
]


def spectral_primary_decomposition_Mallett2019(colourspace,
                                               cmfs,
                                               illuminant,
                                               metric=np.linalg.norm,
                                               metric_args=tuple(),
                                               optimisation_kwargs=None):
    """
    Performs the spectral primary decomposition as described in *Mallett and
    Yuksel (2019)*.

    Parameters
    ----------
    colourspace: RGB_Colourspace
        *RGB* colourspace.
    cmfs : XYZ_ColourMatchingFunctions
        Standard observer colour matching functions.
    illuminant : SpectralDistribution
        Illuminant spectral distribution.
    metric : unicode, optional
        Function to be minimised.

            ``metric(basis, *metric_args) -> float``

        where ``basis`` is three reflectances concatenated together, each
        with a shape matching ``shape``.
    metric_args : tuple, optional
        Additional arguments passed to ``metric``.
    optimisation_kwargs : dict_like, optional
        Parameters for :func:`scipy.optimize.minimize` definition.
    """

    if illuminant.shape != cmfs.shape:
        runtime_warning(
            'Aligning "{0}" illuminant shape to "{1}" colour matching '
            'functions shape.'.format(illuminant.name, cmfs.name))
        illuminant = illuminant.copy().align(cmfs.shape)

    N = len(cmfs.shape)

    R_to_XYZ = np.transpose(
        np.expand_dims(illuminant.values, axis=1) * cmfs.values / (np.sum(
            cmfs.values[:, 1] * illuminant.values)))
    R_to_RGB = np.dot(colourspace.XYZ_to_RGB_matrix, R_to_XYZ)
    basis_to_RGB = block_diag(R_to_RGB, R_to_RGB, R_to_RGB)

    primaries = np.identity(3).reshape(9)

    # Ensure the reflectances correspond to the correct RGB colours.
    colour_match = LinearConstraint(basis_to_RGB, primaries, primaries)

    # Ensure the reflectances are bounded by [0, 1].
    energy_conservation = Bounds(np.zeros(3 * N), np.ones(3 * N))

    # Ensure the sum of the three bases is bounded by [0, 1].
    sum_matrix = np.transpose(np.tile(np.identity(N), (3, 1)))
    sum_constraint = LinearConstraint(sum_matrix, np.zeros(N), np.ones(N))

    optimisation_settings = {
        'method': 'SLSQP',
        'constraints': [colour_match, sum_constraint],
        'bounds': energy_conservation,
        'options': {
            'maxiter': 10 ** 3,
            'ftol': 1e-3
        }
    }

    if optimisation_kwargs is not None:
        optimisation_settings.update(optimisation_kwargs)

    result = minimize(
        metric, args=metric_args, x0=np.zeros(3 * N), **optimisation_settings)
    basis = np.transpose(result.x.reshape(3, N))

    return MultiSpectralDistributions(basis, cmfs.shape.range())


def RGB_to_sd_Mallett2019(RGB, basis):
    """
    Recovers the spectral distribution of given *RGB* colourspace array using
    *Mallett and Yuksel (2019)* method.

    Parameters
    ----------
    RGB : array_like, (3,)
        *RGB* colourspace array.
    basis : MultiSpectralDistributions
        Basis functions for the method. To use the built-in *sRGB* dataset,
        refer to :func:``colour.recovery.sRGB_to_sd_Mallett2019``.

    Returns
    -------
    SpectralDistribution
        Recovered reflectance.
    """

    RGB = to_domain_1(RGB)

    sd = SpectralDistribution(
        np.dot(RGB, np.transpose(basis.values)), basis.wavelengths)
    sd.name = 'Mallett (2019) - {0} (RGB)'.format(RGB)

    return sd


def sRGB_to_sd_Mallett2019(RGB):
    """
    Recovers the spectral distribution of given *sRGB* colourspace array using
    *Mallett and Yuksel (2019)* method.

    Parameters:
    -----------
    RGB : array_like, (3,)
        *sRGB* colourspace array. Do not apply a transfer function to the
        *RGB* values.

    Returns
    -------
    SpectralDistribution
        Recovered reflectance.
    """

    basis = MultiSpectralDistributions(BASIS_sRGB_MALLETT2019,
                                       SPECTRAL_SHAPE_sRGB_MALLETT2019.range())

    return RGB_to_sd_Mallett2019(RGB, basis)
