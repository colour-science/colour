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
    on Rendering - DL-Only and Industry Track. doi:10.2312/SR.20191216
"""

from __future__ import division, print_function, unicode_literals

import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
from scipy.linalg import block_diag

from colour.colorimetry import SpectralDistribution, MultiSpectralDistributions
from colour.recovery import MALLETT_2019_SRGB_SHAPE, MALLETT_2019_SRGB_BASIS

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
                                               shape,
                                               cmfs,
                                               illuminant,
                                               metric=np.linalg.norm,
                                               metric_args=tuple(),
                                               solve_separately=False,
                                               optimisation_kwargs=None):
    """
    Performs the spectral primary decomposition as described in *Mallett and
    Yuksel (2019)*.

    Parameters
    ==========
    colourspace: RGB_Colourspace
        *RGB* colourspace.
    shape : SpectralShape
        Shape of ``cmfs``, ``illuminant`` and the returned basis functions.
    cmfs : XYZ_ColourMatchingFunctions
        Standard observer colour matching functions. The shape must match the
        ``shape`` argument.
    illuminant : SpectralDistribution
        Illuminant spectral distribution. The shape must match the ``shape``
        argument.

    Other parameters
    ================
    metric : unicode, optional
        Function to be minimised.

            ``metric(basis, *metric_args) -> float``

        where ``basis`` is three reflectances concatenated together, each
        with a shapea matching ``shape``.
    metric_args : tuple, optional
        Additional arguments passed to ``metric``.
    """

    N = len(shape.range())

    R_to_XYZ = (np.expand_dims(illuminant.values, axis=1) * cmfs.values /
                (np.sum(cmfs.values[:, 1] * illuminant.values))).T
    R_to_RGB = np.dot(colourspace.XYZ_to_RGB_matrix, R_to_XYZ)
    basis_to_RGB = block_diag(R_to_RGB, R_to_RGB, R_to_RGB)

    primaries = np.identity(3).reshape(9)

    # Ensure the reflectivities correspond to the correct RGB colours
    colour_match = LinearConstraint(basis_to_RGB, primaries, primaries)

    # Ensure the reflectivities are bounded by [0, 1]
    energy_conservation = Bounds(np.zeros(3 * N), np.ones(3 * N))

    # Ensure the sum of the three bases is bounded by [0, 1]
    sum_matrix = np.tile(np.identity(N), (3, 1)).T
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

    opt = minimize(
        metric, args=metric_args, x0=np.zeros(3 * N), **optimisation_settings)
    basis = opt.x.reshape(3, N).T

    return MultiSpectralDistributions(basis, shape.range())


def RGB_to_sd_Mallett2019(RGB, basis):
    """
    Recovers the spectral distribution of given *RGB* colourspace array using
    *Mallett and Yuksel (2019)* method.

    Parameters:
    RGB : array_like, (3,)
        *RGB* colourspace array. Do not apply a transfer function to the
        *RGB* values.
    basis : MultiSpectralDistrubtions
        Basis functions for the method. To use the built-in *sRGB* dataset,
        refer to :func:``colour.recovery.sRGB_to_sd_Mallett2019``.

    Returns
    =======
    SpectralDistribution
        Reconstructed reflectivity.
    """

    sd = SpectralDistribution(np.dot(RGB, basis.values.T), basis.wavelengths)
    sd.name = 'Mallett (2019) - {0} (RGB)'.format(RGB)
    return sd


def sRGB_to_sd_Mallett2019(RGB):
    """
    Recovers the spectral distribution of given *sRGB* colourspace array using
    *Mallett and Yuksel (2019)* method.

    Parameters:
    RGB : array_like, (3,)
        *sRGB* colourspace array. Do not apply a transfer function to the
        *RGB* values.

    Returns
    =======
    SpectralDistribution
        Reconstructed reflectivity.
    """

    basis = MultiSpectralDistributions(MALLETT_2019_SRGB_BASIS,
                                       MALLETT_2019_SRGB_SHAPE.range())
    return RGB_to_sd_Mallett2019(RGB, basis)
